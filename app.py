import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fractions import Fraction
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from itertools import permutations
import streamlit as st


st.set_page_config(page_title="Parametric Shortest Paths", page_icon="🌀", layout="wide")


def default_weights() -> np.ndarray:
    """Provide a simple strongly connected 4x4 weight matrix."""
    return np.array(
        [
            [0.0, 1.0, 4.0, 1.0],
            [1.0, 0.0, 2.0, 3.0],
            [3.0, 1.0, 0.0, 2.0],
            [2.0, 3.0, 1.0, 0.0],
        ]
    )


def floyd_warshall(weights: np.ndarray) -> tuple[np.ndarray, bool]:
    """All-pairs shortest paths; returns (distances, has_negative_cycle)."""
    n = weights.shape[0]
    dist = weights.astype(float).copy()
    np.fill_diagonal(dist, 0.0)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                via = dist[i, k] + dist[k, j]
                if via < dist[i, j]:
                    dist[i, j] = via

    has_neg_cycle = np.any(np.diag(dist) < 0)
    return dist, has_neg_cycle


class ParametricPaths:
    """Encapsulate trajectory logic with simple shortest paths and Karp bound."""

    def __init__(self, weights: np.ndarray):
        import copy

        n = weights.shape[0]
        self.n = n
        self.matrices: list[tuple[Fraction, list[list[Fraction]], list[list[Fraction]]]] = []
        self.freeze_times = [
            [Fraction(0) if i == j else None for j in range(n)] for i in range(n)
        ]
        weights_frac = [[Fraction(weights[i][j]) for j in range(n)] for i in range(n)]

        t = Fraction(0)
        mat = copy.deepcopy(weights_frac)
        dmat = [[-Fraction(1) if i!=j else Fraction(0) for j in range(n)] for i in range(n)]

        while True:
            # Floyd–Warshall style relaxation with lexicographic (mat, dmat)
            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        cand = (mat[i][k] + mat[k][j], (dmat[i][k] + dmat[k][j]))
                        cur = (mat[i][j], dmat[i][j])
                        if cand < cur:
                            mat[i][j] = mat[i][k] + mat[k][j]
                            dmat[i][j] = dmat[i][k] + dmat[k][j]

            # Zero cycles freeze slope
            for i in range(n):
                for j in range(n):
                    if mat[i][j] + mat[j][i] == 0:
                        if self.freeze_times[i][j] is None:
                            self.freeze_times[i][j] = t
                        if self.freeze_times[j][i] is None:
                            self.freeze_times[j][i] = t
                        dmat[i][j] = Fraction(0)
                        dmat[j][i] = Fraction(0)
                    else:
                        dmat[i][j] = -Fraction(1)
                        dmat[j][i] = -Fraction(1)

            for k in range(n):
                for i in range(n):
                    for j in range(n):
                        cand = (mat[i][k] + mat[k][j], (dmat[i][k] + dmat[k][j]))
                        cur = (mat[i][j], dmat[i][j])
                        if cand < cur:
                            mat[i][j] = mat[i][k] + mat[k][j]
                            dmat[i][j] = dmat[i][k] + dmat[k][j]

            self.matrices.append((t, copy.deepcopy(mat), copy.deepcopy(dmat)))
            if all(dmat[i][j] == 0 for i in range(n) for j in range(n)):
                break

            dt = Fraction(10**18)
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        denom = dmat[i][j] - dmat[i][k] - dmat[k][j]
                        if denom <= 0:
                            continue
                        num = mat[i][k] + mat[k][j] - mat[i][j]
                        dt = min(dt, num / denom)

            for i in range(n):
                for j in range(n):
                    mat[i][j] = mat[i][j] + dt * dmat[i][j]
            t += dt

    def max_t(self) -> float:
        return float(self.matrices[-1][0])

    def freeze_time_matrix(self) -> np.ndarray:
        out = np.full((self.n, self.n), np.nan, dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if self.freeze_times[i][j] is not None:
                    out[i, j] = float(self.freeze_times[i][j])
        return out

    def trajectories(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t_frac = Fraction(t)
        chosen = self.matrices[0]
        for tvalue, mat, dmat in self.matrices:
            if t_frac < tvalue:
                break
            chosen = (tvalue, mat, dmat)

        t0, mat, dmat = chosen
        mat_np = np.array(mat, dtype=float) + float(t_frac - t0) * np.array(dmat, dtype=float)
        out = mat_np
        inn = -mat_np.T
        return out, inn

    @staticmethod
    def _fmt_fraction(fr: Fraction) -> str:
        if fr.denominator == 1:
            return str(fr.numerator)
        return f"{fr.numerator}/{fr.denominator}"

    def symbolic_matrix(self, t: float) -> list[list[str]]:
        """Return matrix entries as affine forms in t (e.g., '3-5t')."""
        t_frac = Fraction(t)
        chosen = self.matrices[0]
        for tvalue, mat, dmat in self.matrices:
            if t_frac < tvalue:
                break
            chosen = (tvalue, mat, dmat)
        t0, mat, dmat = chosen
        result: list[list[str]] = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                base = mat[i][j] - t0 * dmat[i][j]
                slope = dmat[i][j]
                if slope == 0:
                    row.append(self._fmt_fraction(base))
                else:
                    base_str = "" if base == 0 else self._fmt_fraction(base)
                    slope_str = self._fmt_fraction(abs(slope))
                    sign = "-" if slope < 0 else "+"
                    if base == 0:
                        row.append(f"{'-' if slope<0 else ''}{slope_str}t")
                    else:
                        row.append(f"{base_str}{sign}{slope_str}t")
            result.append(row)
        return result

def sum_zero_basis() -> np.ndarray:
    """Orthonormal basis of the hyperplane x0+x1+x2+x3 = 0."""
    base = np.array(
        [
            [1.0, -1.0, 0.0, 0.0],
            [1.0, 1.0, -2.0, 0.0],
            [1.0, 1.0, 1.0, -3.0],
        ]
    )
    basis: list[np.ndarray] = []
    for vec in base:
        v = vec.copy()
        for b in basis:
            v -= np.dot(v, b) * b
        norm = np.linalg.norm(v)
        basis.append(v / norm)
    return np.stack(basis)


BASIS = sum_zero_basis()
TO3 = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
    ]
)


def project_point(point4: np.ndarray) -> np.ndarray:
    """Project a 4D point to 3D along the (1,1,1,1) normal direction."""
    centered = point4 - np.mean(point4)
    return BASIS @ centered


def compute_trajectories(
    base_weights: np.ndarray, steps: int
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[list[list[str]]],
]:
    """
    Compute blue/red trajectories by sampling t and using shortest paths.
    """

    n = 4
    dist0, _ = floyd_warshall(base_weights)
    engine = ParametricPaths(base_weights)
    t_max = engine.max_t()
    freeze_times = engine.freeze_time_matrix()

    blue_paths: list[list[np.ndarray]] = [[] for _ in range(n)]
    red_paths: list[list[np.ndarray]] = [[] for _ in range(n)]
    t_values: list[float] = []

    for t in np.linspace(0.0, t_max, steps):
        t_values.append(t)
        out_dists, in_dists = engine.trajectories(t)
        for i in range(n):
            blue_paths[i].append(project_point(out_dists[i, :]))
            red_paths[i].append(project_point(in_dists[i, :]))

    transition_ts = [entry[0] for entry in engine.matrices]
    interval_labels: list[str] = []
    interval_symbolic: list[list[list[str]]] = []
    for idx, start in enumerate(transition_ts):
        end = transition_ts[idx + 1] if idx + 1 < len(transition_ts) else None
        start_str = ParametricPaths._fmt_fraction(start)
        if end is None:
            label = f"{start_str} <= t"
        else:
            end_str = ParametricPaths._fmt_fraction(end)
            label = f"{start_str} <= t <= {end_str}"
        interval_labels.append(label)
        interval_symbolic.append(engine.symbolic_matrix(float(start)))

    t_arr = np.array(t_values, dtype=float)
    blue_arrays = [np.vstack(path) if path else np.empty((0, 3)) for path in blue_paths]
    red_arrays = [np.vstack(path) if path else np.empty((0, 3)) for path in red_paths]
    return blue_arrays, red_arrays, dist0, freeze_times, t_arr, interval_labels, interval_symbolic


def build_halfspaces(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create A, b for inequalities A y <= b in 3D (sum=0 reduced space)."""
    rows = []
    rhs = []
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            a4 = np.zeros(4)
            a4[j] = 1.0
            a4[i] = -1.0
            a3 = a4 @ TO3  # reduce using x3 = -x0 - x1 - x2
            rows.append(a3)
            rhs.append(weights[i, j])
    A = np.vstack(rows)
    b = np.array(rhs)
    return A, b


def find_interior_point(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Find an interior point for the halfspaces via LP with slack variable."""
    m = A.shape[0]
    c = np.array([0.0, 0.0, 0.0, -1.0])  # maximize t => minimize -t
    A_ub = np.hstack([A, np.ones((m, 1))])
    b_ub = b
    bounds = [(None, None)] * 4
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x[-1] < -1e-7:
        return None
    return res.x[:3]


def polytope_geometry(weights: np.ndarray):
    """
    Build the feasible polytope for x[j]-x[i] <= c[i,j] in the sum-zero slice.
    Returns a dict with constraints, vertices (4D and projected), and edges.
    """
    dist, has_neg = floyd_warshall(weights)
    if has_neg:
        return None
    A, b = build_halfspaces(dist)
    interior = find_interior_point(A, b)
    if interior is None:
        return None

    halfspaces = np.hstack([A, -b[:, None]])  # a·x + b' <= 0
    try:
        hs = HalfspaceIntersection(halfspaces, interior)
    except Exception:
        return None

    verts3 = hs.intersections
    if len(verts3) == 0:
        return None

    hull = ConvexHull(verts3)
    hull_vertex_indices = np.array(hull.vertices, dtype=int)
    verts3_unique = verts3[hull_vertex_indices]

    # Edges from hull ridges only, filter coplanar diagonals
    normals = hull.equations[:, :3]
    edge_normals: dict[tuple[int, int], list[np.ndarray]] = {}
    for simplex_idx, simplex in enumerate(hull.simplices):
        for u, v in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((simplex[u], simplex[v]))
            edge_normals.setdefault((i, j), []).append(normals[simplex_idx])

    edges_set: set[tuple[int, int]] = set()
    cos_tol = 1 - 1e-6
    for (i, j), ns in edge_normals.items():
        keep = False
        if len(ns) == 1:
            keep = True
        else:
            for a in range(len(ns)):
                for b in range(a + 1, len(ns)):
                    cos_angle = np.dot(ns[a], ns[b]) / (
                        np.linalg.norm(ns[a]) * np.linalg.norm(ns[b]) + 1e-12
                    )
                    if cos_angle < cos_tol:
                        keep = True
        if keep:
            edges_set.add((i, j))

    verts4 = np.column_stack([verts3_unique, -verts3_unique.sum(axis=1)])
    projected = np.array([project_point(v) for v in verts4])

    index_map = {old: new for new, old in enumerate(hull_vertex_indices)}
    edges = [(index_map[i], index_map[j]) for i, j in edges_set if i in index_map and j in index_map]

    return {
        "constraints": (A, b),
        "vertices4": verts4,
        "vertices3": projected,
        "edges": edges,
        "hull_raw": hull,
    }


def plot_trajectories(
    blue_paths: list[np.ndarray],
    red_paths: list[np.ndarray],
    t_values: np.ndarray,
    poly_vertices: np.ndarray | None,
    poly_edges: list[tuple[int, int]] | None,
    axis_dirs: np.ndarray,
    show_out: bool,
    show_in: bool,
) -> go.Figure:
    blue_color = "#4C78A8"  # muted blue
    red_color = "#F28E2B"  # muted orange
    fig = go.Figure()

    all_pts: list[np.ndarray] = []

    if show_out:
        for idx, path in enumerate(blue_paths):
            if len(path) == 0:
                continue
            all_pts.append(path)
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    name=f"Out-distance {idx}",
                    line=dict(color=blue_color, width=4),
                    marker=dict(size=4, color=blue_color),
                    hoverinfo="skip",
                )
            )

    if show_in:
        for idx, path in enumerate(red_paths):
            if len(path) == 0:
                continue
            all_pts.append(path)
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    name=f"In-distance {idx}",
                    line=dict(color=red_color, width=4, dash="dash"),
                    marker=dict(size=4, color=red_color),
                    hoverinfo="skip",
                )
            )

    if poly_vertices is not None and poly_edges is not None:
        all_pts.append(poly_vertices)
        fig.add_trace(
            go.Scatter3d(
                x=poly_vertices[:, 0],
                y=poly_vertices[:, 1],
                z=poly_vertices[:, 2],
                mode="markers",
                name="Polytope vertices",
                marker=dict(size=6, color="#444", opacity=0.9, symbol="diamond"),
                hoverinfo="skip",
            )
        )
        for i, j in poly_edges:
            pts = poly_vertices[[i, j]]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    name="Polytope edges",
                    line=dict(color="#888", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # Axes for e0..e3 (centered to sum=0)
    axis_scale = 1.5
    for idx, direction in enumerate(axis_dirs):
        vec = direction * axis_scale
        all_pts.append(np.vstack([[0, 0, 0], vec]))
        fig.add_trace(
            go.Scatter3d(
                x=[0, vec[0]],
                y=[0, vec[1]],
                z=[0, vec[2]],
                mode="lines+text",
                name=f"e{idx} axis",
                line=dict(color="#222", width=2, dash="dot"),
                text=[None, f"e{idx}"],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if all_pts:
        pts = np.vstack(all_pts)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        center = (mins + maxs) / 2
        span = max((maxs - mins).max(), 1e-6)
        half = span / 2
        xr = [center[0] - half, center[0] + half]
        yr = [center[1] - half, center[1] + half]
        zr = [center[2] - half, center[2] + half]
    else:
        xr = yr = zr = [-1, 1]

    fig.update_layout(
        scene=dict(
            xaxis_title="e₁",
            yaxis_title="e₂",
            zaxis_title="e₃",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(projection=dict(type="orthographic")),
            xaxis=dict(showgrid=False, showbackground=False, zeroline=False, range=xr),
            yaxis=dict(showgrid=False, showbackground=False, zeroline=False, range=yr),
            zaxis=dict(showgrid=False, showbackground=False, zeroline=False, range=zr),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


st.title("Zonotope Trajectory Visualizer")
st.write(
    "Explore the 4-vertex weighted directed graph, its shortest-path vertices, "
    "and how they move as weights are shifted by a parameter t."
)


with st.sidebar:
    st.header("Graph Weights")
    initial = pd.DataFrame(
        default_weights(),
        columns=["v_0", "v_1", "v_2", "v_3"],
        index=["v_0", "v_1", "v_2", "v_3"],
    )
    edited_df = st.data_editor(
        initial,
        key="weights",
        num_rows="fixed",
        width=320,
    )

    weights = edited_df.to_numpy(dtype=float)
    np.fill_diagonal(weights, 0.0)

    st.header("Visibility")
    show_out = st.checkbox("Show out-trajectories", value=True)
    show_in = st.checkbox("Show in-trajectories", value=True)

steps = 160


blue_paths, red_paths, dist0, freeze_times, t_samples, interval_labels, interval_symbolic = compute_trajectories(
    weights, steps
)

neg_cycle_at_zero = np.any(np.diag(dist0) < 0)

# Polytope geometry at t = 0 (only if feasible)
poly_geom = None if neg_cycle_at_zero else polytope_geometry(weights)
poly_vertices_proj = None
poly_edges = None
if poly_geom is not None:
    poly_vertices_proj = poly_geom["vertices3"]
    poly_edges = poly_geom["edges"]

# Axes directions projected (centered to sum=0)
canonical_dirs = []
for k in range(4):
    e = np.zeros(4)
    e[k] = 1.0
    centered = e - 0.25
    canonical_dirs.append(project_point(centered))
axis_dirs = np.vstack(canonical_dirs)

status_col, meta_col = st.columns([3, 1])
with status_col:
    if neg_cycle_at_zero:
        st.error("The graph already has a negative cycle at t = 0. The polytope is empty.")

    fig = plot_trajectories(
        blue_paths,
        red_paths,
        t_samples,
        poly_vertices_proj,
        poly_edges,
        axis_dirs,
        show_out,
        show_in,
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True, height=800)

with meta_col:
    st.subheader("Parametric distances")
    if len(interval_labels) > 0:
        seg_choice = st.selectbox(
            "Interval",
            options=list(range(len(interval_labels))),
            format_func=lambda idx: interval_labels[idx],
        )
        st.dataframe(
            pd.DataFrame(interval_symbolic[seg_choice], index=["v0", "v1", "v2", "v3"], columns=["v0", "v1", "v2", "v3"]),
            width="stretch",
        )

    st.markdown("---")
    st.subheader("Freeze times")
    ft_df = pd.DataFrame(freeze_times, columns=["v0", "v1", "v2", "v3"], index=["v0", "v1", "v2", "v3"])
    st.dataframe(ft_df, width="stretch")

    if poly_geom:
        st.subheader("Polytope debug")
        A, b = poly_geom["constraints"]
        A = np.atleast_2d(np.asarray(A, dtype=float))
        b = np.atleast_1d(np.asarray(b, dtype=float))
        st.caption(f"{A.shape[0]} inequalities in sum-zero slice, {len(poly_edges) if poly_edges else 0} edges.")
        with st.expander("Show A | b"):
            if b.shape[0] == A.shape[0]:
                stacked = np.hstack([A, b[:, None]])
                st.dataframe(pd.DataFrame(stacked, columns=["a0", "a1", "a2", "b"]))
            else:
                st.write("Constraint shapes differ; showing separately.")
                st.dataframe(pd.DataFrame(A, columns=["a0", "a1", "a2"]))
                st.dataframe(pd.DataFrame(b, columns=["b"]))

    st.subheader("Notes")
    st.markdown(
        "- Out-distance i: shortest paths from i to every j.\n"
        "- In-distance i: shortest paths from every j to i (sign-flipped for display).\n"
        "- Projection: orthogonal to (1,1,1,1) onto an isometric 3D frame.\n"
        "- Grey vertices/edges: the feasible polytope (with sum(x)=0 slice).\n"
        "- Dotted axes show projected directions of the canonical basis e0..e3."
    )
