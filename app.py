import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fractions import Fraction
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection
from itertools import permutations
import streamlit as st


st.set_page_config(page_title="Parametric Shortest Paths", page_icon="🌀", layout="wide")


def default_weights() -> np.ndarray:
    """Provide a simple strongly connected 4x4 weight matrix."""
    return np.array(
        [
            [0.0, 1.0, 0.5, 1.0],
            [1.0, 0.0, 99, 99],
            [1.0, 1.0, 0.0, 2.0],
            [2.0, 99, 1.0, 0.0],
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
        out = np.full((self.n, self.n), "", dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                if self.freeze_times[i][j] is not None:
                    out[i, j] = self._fmt_fraction(self.freeze_times[i][j])
        return out

    # This returns first outdistances (blue, out[i]=max(x_j-x_i))
    # then the indistances (orange, in[j]=-min(x_i-x_j))
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
                    slope_abs = abs(slope)
                    slope_str = "" if slope_abs == 1 else self._fmt_fraction(slope_abs)
                    sign = "-" if slope < 0 else "+"
                    if base == 0:
                        row.append(f"{'-' if slope<0 else ''}{slope_str}t" if slope_str else ("-t" if slope<0 else "t"))
                    else:
                        row.append(f"{base_str}{sign}{slope_str}t" if slope_str else f"{base_str}{sign}t")
            result.append(row)
        return result

    def trajectory_tree_length(self, t_values: np.ndarray) -> float:
        ans=0

        def tropdist(x,y):
          v= [xx-yy for xx,yy in zip(x,y)]

          return max(v)-min(v)

        for i in range(len(self.matrices)-1):
          _, m1, _ = self.matrices[i]
          _, m2, _ = self.matrices[i+1]

          for j in range(4):
            if not any(tropdist(m1[j],m1[k])==0 for k in range(j)):
              ans+=tropdist(m1[j],m2[j])
        
        return ans

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
    list[tuple[str, dict | None]],
    ParametricPaths,
]:
    """
    Compute out/in trajectories by sampling t and using shortest paths.
    """

    n = 4
    dist0, _ = floyd_warshall(base_weights)
    engine = ParametricPaths(base_weights)
    t_max = engine.max_t()
    freeze_times = engine.freeze_time_matrix()

    out_paths_list: list[list[np.ndarray]] = [[] for _ in range(n)]
    in_paths_list: list[list[np.ndarray]] = [[] for _ in range(n)]
    t_values: list[float] = []

    for t in np.linspace(0.0, t_max, steps):
        t_values.append(t)
        out_dists, in_dists = engine.trajectories(t)
        for i in range(n):
            out_paths_list[i].append(project_point(out_dists[i, :]))
            in_paths_list[i].append(project_point(in_dists[i, :]))

    transition_ts = [entry[0] for entry in engine.matrices]
    interval_labels: list[str] = []
    interval_symbolic: list[list[list[str]]] = []
    poly_info: list[tuple[str, dict | None]] = []
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
        out_dists, _ = engine.trajectories(float(start))
        poly_info.append((start_str, polytope_geometry(out_dists)))

    t_arr = np.array(t_values, dtype=float)
    out_paths = [np.vstack(path) if path else np.empty((0, 3)) for path in out_paths_list]
    in_paths = [np.vstack(path) if path else np.empty((0, 3)) for path in in_paths_list]
    return (
        out_paths,
        in_paths,
        dist0,
        freeze_times,
        t_arr,
        interval_labels,
        interval_symbolic,
        poly_info,
        engine,
    )


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


def zonotropal_summands(polytrope: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from itertools import combinations

    n = polytrope.shape[0]

    antizono= polytrope.copy()
    zono= np.zeros((n,n))
    length=0

    for k in range(n):
      for i in range(n):
        for j in range(n):
          antizono[i][j]=min(antizono[i][j], antizono[i][k]+antizono[k][j])

    for i in range(1,n-1):
      for s in combinations(range(n), i):
        t= [x for x in range(n) if x not in s]
        from math import inf
        mu= inf
        for a in s:
          for b in s:
            for c in t:
              for d in t:
                mu=min(mu, antizono[b][c]-antizono[d][c]+antizono[d][a]-antizono[b][a])
        if mu>0:
          for a in s:
            for c in t:
              antizono[a][c]-=mu/2
              antizono[c][a]-=mu/2
              zono[a][c]+=mu/2
              zono[c][a]+=mu/2
          length+=mu
    
    return zono, antizono, length


def metric_tight_span(polytrope: np.ndarray, show_full_pd: bool = False) -> list[np.ndarray]:
    n= len(polytrope)

    for k in range(n):
      for i in range(n):
        for j in range(n):
          assert(polytrope[i][j]<=polytrope[i][k]+polytrope[k][j])
          assert(polytrope[i][j]==polytrope[j][i]), "non symmetric distance"

    polytropes=[]

    for i in range(n):
      for j in range(n):
        # let us compute the boundary of the facet x_i+x_j>= polytrope[i][j]
        # first the boundaries of the form x_i-x_j are the same as polytrope

        polytrope2=None
        if show_full_pd:
          polytrope2 = polytrope.copy()
        else:
          polytrope2 = np.full_like(polytrope, 100000.0)
          np.fill_diagonal(polytrope2, 0.0)
        
        # now we compute the ridges with another facet $x_ii+x_jj$
        # They can be $x_ii + x_jj -x_i - x_j >= d(ii,jj)-d(i,j)$
        # then:
        # x_k-x_j >= d(k,i)-d(i,j) si i=jj, k=ii. We can swap ii and jj but not i and j
        # x_k-x_i >= d(j,k)-d(i,j) si j=ii, k=jj
        
        for k in range(n):
          polytrope2[k][j]=min(polytrope2[k][j], polytrope[i][j]-polytrope[k][i])
          polytrope2[k][i]=min(polytrope2[k][i], polytrope[i][j]-polytrope[j][k])
        
        for k in range(n):
          for ii in range(n):
            for jj in range(n):
              polytrope2[ii][jj]=min(polytrope2[ii][jj],polytrope2[ii][k]+polytrope2[k][jj])
        
        polytropes.append(polytrope2)

    return polytropes

def steiner_tree_edges(
    polytrope: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float | None]:
    n= len(polytrope)
    assert(n==4)

    for k in range(n):
      for i in range(n):
        for j in range(n):
          assert(polytrope[i][j]<=polytrope[i][k]+polytrope[k][j])

    i=0

    besttree: list[tuple[np.ndarray, np.ndarray]] | None = None
    from math import inf
    bestlength= inf

    for j in range(1,4):
      # i,j are in a side of the partition, 
      # k,l are the other side
      k,l= {1:(2,3), 2:(1,3), 3:(1,2)}[j]

      # i,j connect to x
      # k,l connect to y

      # receives two np vectors
      def tropdist(x,y):
        v= x-y
        return max(v)-min(v)

      def f(x,y):
        return tropdist(x,polytrope[i])+tropdist(x,polytrope[j])\
              +tropdist(x,y)\
              +tropdist(y,polytrope[k])+tropdist(y,polytrope[l])

      x0 = 0.5 * (polytrope[i] + polytrope[j])
      y0 = 0.5 * (polytrope[k] + polytrope[l])
      def f_flat(v):
        return f(v[:n], v[n:])
      res = minimize(
        f_flat,
        np.hstack([x0, y0]),
        method="Powell",
        options={"maxiter": 2000, "xtol": 1e-6, "ftol": 1e-6},
      )
      x, y = res.x[:n], res.x[n:]

      length = f(x, y)
      if length < bestlength:
        besttree = [
          (x, polytrope[i]),
          (x, polytrope[j]),
          (x, y),
          (y, polytrope[k]),
          (y, polytrope[l]),
        ]
        bestlength = length

    return besttree or [], (None if bestlength == inf else bestlength)

def nudge_off_diagonal(mat: np.ndarray, eps: float = 0.001) -> np.ndarray:
    """Add a small epsilon to off-diagonal entries to avoid degenerate plots."""
    nudged = mat.copy()
    for i in range(nudged.shape[0]):
        for j in range(nudged.shape[1]):
            if i != j:
                nudged[i, j] += eps
    return nudged


def plot_trajectories(
    out_paths: list[np.ndarray],
    in_paths: list[np.ndarray],
    t_values: np.ndarray,
    polytopes: list[tuple[str, np.ndarray, list[tuple[int, int]]]],
    axis_dirs: np.ndarray,
    show_out: bool,
    show_in: bool,
    tight_span_overlays: list[tuple[np.ndarray, list[tuple[int, int]]]],
    pd_overlays: list[tuple[np.ndarray, list[tuple[int, int]]]],
    steiner_edges: list[tuple[np.ndarray, np.ndarray]],
    tight_span_edge_threshold: float,
    labels_3d: list[tuple[np.ndarray, str]],
) -> go.Figure:
    out_color = "#4C78A8"  # blue for out-distances
    in_color = "#F28E2B"  # orange for in-distances
    tight_span_color = "#00C2D1"
    steiner_color = "#59A14F"
    fig = go.Figure()

    all_pts: list[np.ndarray] = []

    def to_plot_point(pt: np.ndarray) -> np.ndarray:
        arr = np.asarray(pt, dtype=float)
        if arr.shape == (4,):
            return project_point(arr)
        return arr

    if show_out:
        for idx, path in enumerate(out_paths):
            if len(path) == 0:
                continue
            all_pts.append(path)
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    name="out[i]= (max(x_j-x_i) | j=0..3)" if idx == 0 else None,
                    line=dict(color=out_color, width=4),
                    marker=dict(size=4, color=out_color),
                    hoverinfo="skip",
                    showlegend=(idx == 0),
                )
            )

    if show_in:
        for idx, path in enumerate(in_paths):
            if len(path) == 0:
                continue
            all_pts.append(path)
            fig.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    name="in[j]= (max(x_j-x_i) | i=0..3)" if idx == 0 else None,
                    line=dict(color=in_color, width=4, dash="dash"),
                    marker=dict(size=4, color=in_color),
                    hoverinfo="skip",
                    showlegend=(idx == 0),
                )
            )

    if polytopes:
        colors = ["#555", "#777", "#999", "#BBB", "#333", "#AAA"]
        for idx, (label, poly_vertices, poly_edges) in enumerate(polytopes):
            if poly_vertices is None or poly_edges is None:
                continue
            color = colors[idx % len(colors)]
            all_pts.append(poly_vertices)
            for i, j in poly_edges:
                pts = poly_vertices[[i, j]]
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        name=f"Polytope edges t={label}",
                        line=dict(color=color, width=3),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    if tight_span_overlays:
        show_legend = True
        for poly_vertices, poly_edges in tight_span_overlays:
            if poly_vertices is None or poly_edges is None:
                continue
            for i, j in poly_edges:
                pts = poly_vertices[[i, j]]
                if (
                    np.linalg.norm(pts[0]) > tight_span_edge_threshold
                    or np.linalg.norm(pts[1]) > tight_span_edge_threshold
                ):
                    continue
                all_pts.append(pts)
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        name="Metric tight span",
                        line=dict(color=tight_span_color, width=6),
                        hoverinfo="skip",
                        showlegend=show_legend,
                    )
                )
                show_legend = False

    if steiner_edges:
        show_legend = True
        for start, end in steiner_edges:
            seg = np.vstack([to_plot_point(start), to_plot_point(end)])
            all_pts.append(seg)
            fig.add_trace(
                go.Scatter3d(
                    x=seg[:, 0],
                    y=seg[:, 1],
                    z=seg[:, 2],
                    mode="lines",
                    name="out-Steiner Tree",
                    line=dict(color=steiner_color, width=5),
                    hoverinfo="skip",
                    showlegend=show_legend,
                )
            )
            show_legend = False

    if pd_overlays:
        show_legend = True
        for poly_vertices, poly_edges in pd_overlays:
            if poly_vertices is None or poly_edges is None:
                continue
            for i, j in poly_edges:
                pts = poly_vertices[[i, j]]
                if (
                    np.linalg.norm(pts[0]) > tight_span_edge_threshold
                    or np.linalg.norm(pts[1]) > tight_span_edge_threshold
                ):
                    continue
                all_pts.append(pts)
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="lines",
                        name="P_d",
                        line=dict(color=tight_span_color, width=5),
                        hoverinfo="skip",
                        showlegend=show_legend,
                    )
                )
                show_legend = False

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

    in_points = [(p, txt) for p, txt, kind in labels_3d if kind == "in"]
    out_points = [(p, txt) for p, txt, kind in labels_3d if kind == "out"]

    for pts, color, shift_dir in (
        (out_points, out_color, np.array([1.0, 0.0, 0.0])),
        (in_points, in_color, np.array([-1.0, 0.0, 0.0])),
    ):
        if not pts:
            continue
        xs, ys, zs, texts = [], [], [], []
        for p, txt in pts:
            offset = shift_dir * 0.05
            xs.append(p[0] + offset[0])
            ys.append(p[1] + offset[1])
            zs.append(p[2] + offset[2])
            texts.append(txt)
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="text",
                text=texts,
                textfont=dict(color=color, size=12),
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    if all_pts:
        pts = np.vstack(all_pts)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        max_abs = max(np.abs(mins).max(), np.abs(maxs).max(), 1e-6)
        xr = yr = zr = [-max_abs, max_abs]
    else:
        xr = yr = zr = [-1, 1]

    fig.update_layout(
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            dragmode="orbit",
            camera=dict(projection=dict(type="orthographic"), center=dict(x=0, y=0, z=0)),
            xaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=xr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
            yaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=yr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
            zaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=zr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=0, r=0, b=0, t=30),
        height=675,
    )
    return fig


def plot_polytope_only(
    poly_vertices: np.ndarray | None,
    poly_edges: list[tuple[int, int]] | None,
    axis_dirs: np.ndarray,
    title: str,
) -> go.Figure:
    """Render a single polytope with just its edges and the canonical axes."""
    fig = go.Figure()
    all_pts: list[np.ndarray] = []

    if poly_vertices is not None and poly_edges is not None:
        all_pts.append(poly_vertices)
        for i, j in poly_edges:
            pts = poly_vertices[[i, j]]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines",
                    name="Polytope edge",
                    line=dict(color="#555", width=4),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

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
        max_abs = max(np.abs(mins).max(), np.abs(maxs).max(), 1e-6)
        xr = yr = zr = [-max_abs, max_abs]
    else:
        xr = yr = zr = [-1, 1]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            dragmode="orbit",
            camera=dict(projection=dict(type="orthographic"), center=dict(x=0, y=0, z=0)),
            xaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=xr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
            yaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=yr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
            zaxis=dict(
                showgrid=False,
                showbackground=False,
                zeroline=False,
                range=zr,
                showticklabels=False,
                ticks="",
                showspikes=False,
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=50),
        height=480,
    )
    return fig


st.title("Polytrope Trajectory Visualizer")
st.write(
    "Explore the 4-vertex weighted directed graph, its shortest-path vertices, "
    "and how they move as weights are shifted by a parameter t."
)


with st.sidebar:
    st.header("Graph Weights")
    initial = pd.DataFrame(
        default_weights(),
        columns=["v0", "v1", "v2", "v3"],
        index=["v0", "v1", "v2", "v3"],
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
    show_base_polytope = st.checkbox("Show base polytope", value=True)
    show_inner_polytopes = st.checkbox("Show inner polytropes", value=True)
    show_metric_tight_span = st.checkbox("Show metric tight span", value=False)
    show_pd = st.checkbox("Show P_d", value=False)
    show_steiner_tree = st.checkbox("Show out-Steiner Tree", value=False)
    tight_span_edge_threshold = 1000.0

steps = 160


(
    out_paths,
    in_paths,
    dist0,
    freeze_times,
    t_samples,
    interval_labels,
    interval_symbolic,
    poly_info,
    param_engine,
) = compute_trajectories(
    weights, steps
)

neg_cycle_at_zero = np.any(np.diag(dist0) < 0)

# Axes directions projected (centered to sum=0)
canonical_dirs = []
for k in range(4):
    e = np.zeros(4)
    e[k] = 1.0
    centered = e - 0.25
    canonical_dirs.append(project_point(centered))
axis_dirs = np.vstack(canonical_dirs)

poly_overlays: list[tuple[str, np.ndarray, list[tuple[int, int]]]] = []
poly_debug_items: list[tuple[str, dict]] = []
label_points: list[tuple[np.ndarray, str, str]] = []
first_poly_added = False
for label, geom in poly_info:
    if geom is None:
        continue
    if not first_poly_added:
        if show_base_polytope:
            poly_overlays.append((label, geom["vertices3"], geom["edges"]))
            poly_debug_items.append((label, geom))
        first_poly_added = True
        if not show_inner_polytopes:
            break
        continue
    if show_inner_polytopes:
        poly_overlays.append((label, geom["vertices3"], geom["edges"]))
        poly_debug_items.append((label, geom))

base_geom = None
for lbl, geom in poly_info:
    if lbl == "0" and geom is not None:
        base_geom = geom
        break
if base_geom is None and poly_overlays:
    # fallback to the first available
    first_label, first_vertices, _ = poly_overlays[0]
    base_geom = {"vertices3": first_vertices}

if base_geom is not None:
    verts = np.asarray(base_geom["vertices3"])
    for j in range(4):
        direction = axis_dirs[j]
        dots = verts @ direction
        max_idx = int(np.argmax(dots))
        min_idx = int(np.argmin(dots))
        max_pt = verts[max_idx]
        min_pt = verts[min_idx]
        label_points.append((max_pt, f"{j}", "in"))
        label_points.append((min_pt, f"{j}", "out"))

metric_tight_span_overlays: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
if show_metric_tight_span:
    try:
        for poly in metric_tight_span(dist0, show_full_pd=False):
            geom = polytope_geometry(poly)
            if geom is None:
                continue
            metric_tight_span_overlays.append((geom["vertices3"], geom["edges"]))
    except NotImplementedError:
        st.warning("Define metric_tight_span to see the metric tight span.")
    except Exception as exc:  # pragma: no cover - defensive for user edits
        st.warning(f"Metric tight span failed: {exc}")

pd_overlays: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
if show_pd:
    try:
        for poly in metric_tight_span(dist0, show_full_pd=True):
            geom = polytope_geometry(poly)
            if geom is None:
                continue
            pd_overlays.append((geom["vertices3"], geom["edges"]))
    except NotImplementedError:
        st.warning("Define metric_tight_span to see P_d.")
    except Exception as exc:  # pragma: no cover - defensive for user edits
        st.warning(f"P_d failed: {exc}")

steiner_edges: list[tuple[np.ndarray, np.ndarray]] = []
steiner_length: float | None = None
if show_steiner_tree:
    try:
        steiner_edges, steiner_length = steiner_tree_edges(dist0)
    except NotImplementedError:
        st.warning("Define steiner_tree_edges to see the out-Steiner Tree.")
    except Exception as exc:  # pragma: no cover - defensive for user edits
        st.warning(f"out-Steiner Tree failed: {exc}")

zonotropal_polytope = None
antizonotropal_polytope = None
zonotropal_length: float | None = None
try:
    zonotropal_polytope, antizonotropal_polytope, zonotropal_length = zonotropal_summands(dist0)
except NotImplementedError:
    st.warning("Define zonotropal_summands to see zonotropal/antizonotropal plots.")
except Exception as exc:  # pragma: no cover - defensive for user edits
    st.warning(f"Zonotropal decomposition failed: {exc}")

zono_display = nudge_off_diagonal(zonotropal_polytope) if zonotropal_polytope is not None else None
anti_display = nudge_off_diagonal(antizonotropal_polytope) if antizonotropal_polytope is not None else None

zono_geom = polytope_geometry(zono_display) if zono_display is not None else None
anti_geom = polytope_geometry(anti_display) if anti_display is not None else None

status_col, meta_col = st.columns([3, 1])
with status_col:
    if neg_cycle_at_zero:
        st.error("The graph already has a negative cycle at t = 0. The polytope is empty.")

    fig = plot_trajectories(
        out_paths,
        in_paths,
        t_samples,
        poly_overlays,
        axis_dirs,
        show_out,
        show_in,
        metric_tight_span_overlays,
        pd_overlays,
        steiner_edges,
        tight_span_edge_threshold,
        label_points,
    )
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)
    if show_steiner_tree:
        if steiner_length is None:
            st.caption("out-Steiner Tree length: (not available)")
        else:
            st.caption(f"out-Steiner Tree length: {steiner_length:.4f}")
        try:
            trajectory_length = param_engine.trajectory_tree_length(t_samples)
            st.caption(f"Trajectory tree length: {trajectory_length:.4f}")
        except Exception as exc:  # pragma: no cover - defensive for user edits
            st.caption(f"Trajectory tree length failed: {exc}")

    zono_vertices = zono_geom["vertices3"] if zono_geom else None
    zono_edges = zono_geom["edges"] if zono_geom else None
    #st.subheader("Zonotropal summand")
    zono_fig = plot_polytope_only(zono_vertices, zono_edges, axis_dirs, "Zonotropal summand")
    st.plotly_chart(zono_fig, config={"displayModeBar": False}, use_container_width=True)
    if zonotropal_length is not None:
        st.caption(f"Zonotropal generator length sum: {zonotropal_length:.4f}")

    anti_vertices = anti_geom["vertices3"] if anti_geom else None
    anti_edges = anti_geom["edges"] if anti_geom else None
    #st.subheader("Antizonotropal summand")
    anti_fig = plot_polytope_only(anti_vertices, anti_edges, axis_dirs, "Antizonotropal summand")
    st.plotly_chart(anti_fig, config={"displayModeBar": False}, use_container_width=True)

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

    if zonotropal_polytope is not None and antizonotropal_polytope is not None:
        st.subheader("Zono / Antizono matrices")
        st.markdown("Zonotropal summand")
        st.dataframe(
            pd.DataFrame(zono_display, columns=["v0", "v1", "v2", "v3"], index=["v0", "v1", "v2", "v3"]),
            width="stretch",
        )
        st.markdown("Antizonotropal summand")
        st.dataframe(
            pd.DataFrame(anti_display, columns=["v0", "v1", "v2", "v3"], index=["v0", "v1", "v2", "v3"]),
            width="stretch",
        )

    if poly_debug_items:
        st.subheader("Polytope debug")
        for label, geom in poly_debug_items:
            A, b = geom["constraints"]
            A = np.atleast_2d(np.asarray(A, dtype=float))
            b = np.atleast_1d(np.asarray(b, dtype=float))
            edges = geom.get("edges", [])
            st.caption(f"t = {label}: {A.shape[0]} inequalities, {len(edges)} edges.")
            with st.expander(f"Show A | b (t={label})"):
                if b.shape[0] == A.shape[0]:
                    stacked = np.hstack([A, b[:, None]])
                    st.dataframe(pd.DataFrame(stacked, columns=["a0", "a1", "a2", "b"]))
                else:
                    st.write("Constraint shapes differ; showing separately.")
                    st.dataframe(pd.DataFrame(A, columns=["a0", "a1", "a2"]))
                    st.dataframe(pd.DataFrame(b, columns=["b"]))

    if (show_out or show_in) and not show_steiner_tree:
        try:
            trajectory_length = param_engine.trajectory_tree_length(t_samples)
            st.caption(f"Trajectory tree length: {trajectory_length:.4f}")
        except Exception as exc:  # pragma: no cover - defensive for user edits
            st.caption(f"Trajectory tree length failed: {exc}")

    st.subheader("Notes")
    st.markdown(
        "- Out-distance i: shortest paths from i to every j.\n"
        "- In-distance i: shortest paths from every j to i (sign-flipped).\n"
        "- Projection: orthogonal to (1,1,1,1) onto an isometric 3D frame.\n"
        "- Grey vertices/edges: the feasible polytope (with sum(x)=0 slice).\n"
        "- Dotted axes show projected directions of the canonical basis e0..e3."
    )
