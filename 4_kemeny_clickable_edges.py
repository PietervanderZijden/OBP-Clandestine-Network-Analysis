"""Kemeny Constant DSS Page (Task 4)

Features
- Supports undirected/directed and unweighted/weighted graphs.
- Undirected mode: symmetrizes by average A <- (A + A^T)/2.
- Builds random-walk transition P = D_out^{-1} A with dangling-node self-loops.
- Computes Kemeny constant on:
    * full graph if connected (undirected) / strongly connected (directed)
    * otherwise on the largest connected component (LCC) / largest strongly connected component (LSCC)
- Stateful multi-edge removal.
- Objective-aware recommendations (Disrupt vs Improve) and component-shrink suggestions.
- Read-only graph visualization (now **3D Plotly**).

Notes
- The main control for removals is the edge table (robust). The graph view is visual only.
- Kemeny formula used: K = tr( (I - P + 1*pi)^(-1) ).
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import streamlit as st
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components

from ui_components import (
    apply_tactical_theme,
    COLOR_WIRE,
    COLOR_STEEL,
    COLOR_ALERT,
    COLOR_TEXT,
)


# -------------------------
# Settings
# -------------------------
@dataclass(frozen=True)
class GraphSettings:
    directed: bool
    keep_weights: bool
    lazy_alpha: float


# -------------------------
# Graph loading + preprocessing
# -------------------------
@st.cache_data(show_spinner=False)
def load_mtx_as_sparse(path: str) -> sp.csr_matrix:
    A = mmread(path)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()
    return A


def preprocess_adjacency(A_raw: sp.csr_matrix, settings: GraphSettings) -> sp.csr_matrix:
    """Return CSR adjacency suitable for random-walk transition construction."""
    A = A_raw.tocsr(copy=True).astype(float)

    # Remove self-loops
    A.setdiag(0.0)
    A.eliminate_zeros()

    # Enforce nonnegative weights
    if A.data.size and np.any(A.data < 0):
        raise ValueError("Adjacency/weight matrix has negative entries; cannot form probabilities.")

    if not settings.directed:
        # Symmetrize by average
        A = (A + A.T) * 0.5
        A = A.tocsr()
        A.eliminate_zeros()

    if not settings.keep_weights:
        # Binarize
        if A.data.size:
            A.data[:] = 1.0
        A.eliminate_zeros()

    return A


def edges_from_adjacency(A: sp.csr_matrix, directed: bool) -> list[tuple[int, int]]:
    """Extract edges list (u,v) from nonzeros of A, excluding self-loops."""
    coo = A.tocoo()
    if directed:
        edges = [(int(u), int(v)) for u, v in zip(coo.row, coo.col) if u != v]
        edges = sorted(set(edges))
        return edges
    # undirected: keep u < v
    edges_set: set[tuple[int, int]] = set()
    for u, v in zip(coo.row, coo.col):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges_set.add((a, b))
    return sorted(edges_set)


def build_nx_graph(A: sp.csr_matrix, directed: bool) -> nx.Graph:
    """Build a NetworkX graph from adjacency."""
    if directed:
        G = nx.from_scipy_sparse_array(A, create_using=nx.DiGraph)
    else:
        G = nx.from_scipy_sparse_array(A)
    # remove self-loops (just in case)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# -------------------------
# Random-walk transition + stationary distribution
# -------------------------

def random_walk_transition(A: sp.csr_matrix, lazy_alpha: float = 0.0) -> sp.csr_matrix:
    """Row-stochastic P from adjacency/weights A using out-strength. Adds self-loops to dangling nodes."""
    A = A.tocsr(copy=True)
    row_sum = np.array(A.sum(axis=1)).ravel()
    dangling = np.where(row_sum == 0)[0]
    if dangling.size:
        A[dangling, dangling] = 1.0
        A.eliminate_zeros()
        row_sum = np.array(A.sum(axis=1)).ravel()

    inv_row = np.zeros_like(row_sum, dtype=float)
    inv_row[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    P = (sp.diags(inv_row) @ A).tocsr()

    if lazy_alpha and lazy_alpha > 0:
        n = P.shape[0]
        P = (1.0 - lazy_alpha) * P + lazy_alpha * sp.eye(n, format="csr")

    return P


def stationary_distribution_power(P: sp.csr_matrix, tol: float = 1e-12, max_iter: int = 200_000) -> np.ndarray:
    """Compute stationary distribution for a (finite) Markov chain using power iteration on P^T."""
    P = P.tocsr()
    n = P.shape[0]
    if n == 1:
        return np.array([1.0])

    # Ensure row-stochastic (tolerant)
    rs = np.array(P.sum(axis=1)).ravel()
    if not np.allclose(rs, 1.0, atol=1e-10):
        raise ValueError("Transition matrix P is not row-stochastic.")

    pi = np.full(n, 1.0 / n, dtype=float)
    PT = P.transpose().tocsr()

    for _ in range(max_iter):
        pi_next = PT @ pi
        s = float(pi_next.sum())
        if s <= 0:
            raise ValueError("Numerical issue: stationary vector sum <= 0.")
        pi_next /= s
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            return pi_next
        pi = pi_next

    raise RuntimeError("stationary distribution did not converge (try increasing lazy_alpha).")


def stationary_distribution_undirected_from_strength(A: sp.csr_matrix) -> np.ndarray:
    """Closed-form stationary distribution for undirected random walk: pi_i proportional to strength/degree."""
    n = A.shape[0]
    if n == 1:
        return np.array([1.0])
    s = np.array(A.sum(axis=1)).ravel().astype(float)
    total = float(s.sum())
    if total <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return s / total


# -------------------------
# Kemeny computation with component selection
# -------------------------

def kemeny_constant(P: sp.csr_matrix, pi: np.ndarray) -> float:
    """K = tr( (I - P + 1*pi)^(-1) ). Dense solve is fine for n ~ 62."""
    n = P.shape[0]
    if n <= 1:
        return 0.0
    Pd = P.toarray()
    I = np.eye(n)
    one_pi = np.ones((n, 1)) @ pi.reshape(1, -1)
    M = I - Pd + one_pi
    # solve M Z = I
    Z = np.linalg.solve(M, I)
    return float(np.trace(Z))


def main_component_nodes(A: sp.csr_matrix, directed: bool) -> tuple[np.ndarray, bool, str]:
    """Return (nodes, full_used, label) where label is Full/LCC/LSCC."""
    if directed:
        n_comp, labels = connected_components(A, directed=True, connection="strong", return_labels=True)
        comp_label = "LSCC"
    else:
        n_comp, labels = connected_components(A, directed=False, return_labels=True)
        comp_label = "LCC"

    if n_comp == 1:
        return np.arange(A.shape[0]), True, "Full graph"

    sizes = np.bincount(labels)
    lab = int(np.argmax(sizes))
    nodes = np.where(labels == lab)[0]
    return nodes, False, comp_label


def compute_kemeny_on_component(A_comp: sp.csr_matrix, directed: bool, lazy_alpha: float) -> float:
    P = random_walk_transition(A_comp, lazy_alpha=lazy_alpha)
    if directed:
        pi = stationary_distribution_power(P)
    else:
        # Undirected case (after symmetrization by average), use degree/strength theorem
        pi = stationary_distribution_undirected_from_strength(A_comp)
    return kemeny_constant(P, pi)


def kemeny_score(A: sp.csr_matrix, directed: bool, lazy_alpha: float) -> tuple[str, float, int]:
    nodes, full_used, label = main_component_nodes(A, directed=directed)
    A_comp = A[nodes, :][:, nodes].tocsr()
    K = compute_kemeny_on_component(A_comp, directed=directed, lazy_alpha=lazy_alpha)
    if full_used:
        scope = "Full graph"
    else:
        scope = label
    return scope, float(K), int(nodes.size)


# -------------------------
# Edge removal state
# -------------------------

def normalize_edge(u: int, v: int, directed: bool) -> tuple[int, int]:
    if directed:
        return int(u), int(v)
    a, b = (int(u), int(v)) if u < v else (int(v), int(u))
    return a, b


def apply_edge_removals(A0: sp.csr_matrix, removed_set: set[tuple[int, int]], directed: bool) -> sp.csr_matrix:
    if not removed_set:
        return A0
    B = A0.tolil(copy=True)
    for (u, v) in removed_set:
        B[u, v] = 0.0
        if not directed:
            B[v, u] = 0.0
    B = B.tocsr()
    B.eliminate_zeros()
    return B


def remaining_edges_df_from_edges0(edges0: list[tuple[int, int]], removed_set: set[tuple[int, int]]) -> pd.DataFrame:
    rem = [e for e in edges0 if e not in removed_set]
    return pd.DataFrame(rem, columns=["u", "v"])


# -------------------------
# Visualization (3D, read-only)
# -------------------------
@st.cache_data(show_spinner=False)
def fixed_layout3d_from_A(A_data, A_indices, A_indptr, shape, seed: int = 42) -> dict[int, tuple[float, float, float]]:
    """Fixed 3D layout for the baseline graph.

    We compute positions once on the baseline (undirected view for stability), then reuse
    them for every rerun so nodes don't jump around when edges are removed.
    """
    A = sp.csr_matrix((A_data, A_indices, A_indptr), shape=shape)

    # Undirected view for layout stability
    Au = (A + A.T) * 0.5
    Au = Au.tocsr()
    Au.setdiag(0.0)
    Au.eliminate_zeros()

    G = nx.from_scipy_sparse_array(Au)

    # Spring layout in 3D is stable for small graphs; dim=3 gives (x, y, z).
    pos = nx.spring_layout(G, seed=seed, k=1.2, dim=3)

    # Scale up so the 3D plot isn't cramped.
    return {int(n): (float(x) * 1000.0, float(y) * 1000.0, float(z) * 1000.0) for n, (x, y, z) in pos.items()}


def _edge_segments_from_adjacency(
    A: sp.csr_matrix,
    directed: bool,
    layout3d: dict[int, tuple[float, float, float]],
) -> tuple[list[float], list[float], list[float]]:
    """Return x/y/z arrays suitable for a single Plotly line trace (with None separators)."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    coo = A.tocoo()

    if directed:
        for u, v in zip(coo.row, coo.col):
            u = int(u)
            v = int(v)
            if u == v:
                continue
            x0, y0, z0 = layout3d.get(u, (0.0, 0.0, 0.0))
            x1, y1, z1 = layout3d.get(v, (0.0, 0.0, 0.0))
            xs += [x0, x1, None]
            ys += [y0, y1, None]
            zs += [z0, z1, None]
        return xs, ys, zs

    # Undirected: keep each edge once (u < v)
    seen: set[tuple[int, int]] = set()
    for u, v in zip(coo.row, coo.col):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        x0, y0, z0 = layout3d.get(a, (0.0, 0.0, 0.0))
        x1, y1, z1 = layout3d.get(b, (0.0, 0.0, 0.0))
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]
    return xs, ys, zs


def _edge_trace_points(
    u: int,
    v: int,
    layout3d: dict[int, tuple[float, float, float]],
    steps: int = 7,
) -> tuple[list[float], list[float], list[float]]:
    """Sample points along an edge so clicks are reliable (markers are invisible)."""
    x0, y0, z0 = layout3d.get(int(u), (0.0, 0.0, 0.0))
    x1, y1, z1 = layout3d.get(int(v), (0.0, 0.0, 0.0))
    xs = np.linspace(x0, x1, steps).tolist()
    ys = np.linspace(y0, y1, steps).tolist()
    zs = np.linspace(z0, z1, steps).tolist()
    return xs, ys, zs


def render_graph_panel_3d_interactive(A0: sp.csr_matrix, A_state: sp.csr_matrix, settings: GraphSettings) -> bool:
    """3D graph with clickable edges.

    Returns True if interactive clicking is available, else False (caller can show fallback UI).
    """
    st.subheader("Network view (current state) — 3D (click edges)")
    st.caption("Drag to rotate, scroll to zoom. Click an edge to select it, then remove from the buttons below.")

    show_removed = st.checkbox("Show removed edges (faded)", value=False, key="kemeny_show_removed_edges")

    # Session state
    if "selected_edges" not in st.session_state:
        st.session_state.selected_edges = set()

    try:
        import plotly.graph_objects as go
    except Exception as e:
        st.error("Plotly is required for the 3D graph. Install with: pip install plotly")
        st.exception(e)
        return False

    # Click capture (optional dependency)
    try:
        from streamlit_plotly_events import plotly_events
    except Exception:
        plotly_events = None

    # Fixed 3D layout from baseline (nodes don't jump)
    layout3d = fixed_layout3d_from_A(A0.data, A0.indices, A0.indptr, A0.shape)

    n = A_state.shape[0]

    # Degree for hover text (computed on the current STATE graph)
    G_state_local = build_nx_graph(A_state, directed=settings.directed)
    degree_map = dict(G_state_local.degree())

    # Current active edges (removed edges are not in A_state)
    edges_state = edges_from_adjacency(A_state, directed=settings.directed)

    selected_edges: set[tuple[int, int]] = set(st.session_state.selected_edges)

    fig = go.Figure()

    # --- Clickable edges (one trace per edge) ---
    for (u, v) in edges_state:
        is_sel = (u, v) in selected_edges
        xs, ys, zs = _edge_trace_points(u, v, layout3d=layout3d, steps=7)

        if settings.directed:
            hover = f"Edge {u+1} → {v+1}"
        else:
            hover = f"Edge {u+1} – {v+1}"

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines+markers",
                line=dict(color=(COLOR_ALERT if is_sel else COLOR_WIRE), width=(6 if is_sel else 2)),
                marker=dict(size=6, opacity=0.0),  # invisible markers make clicking stable
                hovertemplate=hover + "<extra></extra>",
                showlegend=False,
            )
        )

    # --- Removed edges (optional, faded) ---
    if show_removed:
        removed_set = st.session_state.get("removed_set", set())
        rx: list[float] = []
        ry: list[float] = []
        rz: list[float] = []
        for (u, v) in removed_set:
            u = int(u)
            v = int(v)
            if u == v:
                continue
            x0, y0, z0 = layout3d.get(u, (0.0, 0.0, 0.0))
            x1, y1, z1 = layout3d.get(v, (0.0, 0.0, 0.0))
            rx += [x0, x1, None]
            ry += [y0, y1, None]
            rz += [z0, z1, None]

        if rx:
            fig.add_trace(
                go.Scatter3d(
                    x=rx,
                    y=ry,
                    z=rz,
                    mode="lines",
                    line=dict(color=COLOR_ALERT, width=6),
                    hoverinfo="skip",
                    opacity=0.25,
                    showlegend=False,
                )
            )

    # --- Nodes (spheres) + labels (1-based) ---
    node_x: list[float] = []
    node_y: list[float] = []
    node_z: list[float] = []
    node_text: list[str] = []

    for i in range(n):
        x, y, z = layout3d.get(i, (0.0, 0.0, 0.0))
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_text.append(f"Node {i+1}<br>Degree: {degree_map.get(i, 0)}")

    fig.add_trace(
        go.Scatter3d(
            x=node_x,
            y=node_y,
            z=node_z,
            mode="markers+text",
            text=[str(i + 1) for i in range(n)],
            textposition="middle center",
            textfont=dict(color="#FFFFFF", size=10),
            marker=dict(size=14, color=COLOR_STEEL),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        # Preserve camera when Streamlit reruns
        uirevision="kemeny-3d",
        paper_bgcolor="#d9d9d9",
        plot_bgcolor="#d9d9d9",
        scene=dict(
            bgcolor="#d9d9d9",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        height=750,
    )

    if plotly_events is None:
        # No click-capture available -> show read-only figure; caller should provide fallback removal UI.
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
        st.warning(
            "Clickable edges are disabled (missing optional dependency 'streamlit-plotly-events'). "
            "Use the fallback removal UI below, or install it for full interaction."
        )
        return False

    # Render + capture click events
    events = plotly_events(
        fig,
        click_event=True,
        select_event=False,
        hover_event=False,
        override_height=750,
    )

    if events:
        evt = events[0]
        curve = evt.get("curveNumber")
        if isinstance(curve, int) and 0 <= curve < len(edges_state):
            e = edges_state[curve]
            if e in selected_edges:
                selected_edges.remove(e)
            else:
                selected_edges.add(e)
            st.session_state.selected_edges = selected_edges
            st.rerun()

    return True


#---helper for node display numbering
def show_node(i: int) -> int:
    return int(i) + 1


# -------------------------
# Streamlit Page
# -------------------------
apply_tactical_theme()

st.title("Kemeny Constant — Edge Removal Analysis")
st.caption("Stateful multi-edge removal with Kemeny recomputation. Supports directed/weighted graphs.")

file_path = "data/clandestine_network_example.mtx"

with st.expander("Graph settings", expanded=False):
    directed = st.checkbox("Directed graph", value=False)
    keep_weights = st.checkbox("Use weights (do not binarize)", value=False)
    lazy_alpha = st.slider("Lazy random walk alpha (stability)", 0.0, 0.5, 0.0, 0.05)
    st.caption(
        "Undirected mode symmetrizes by average: A <- (A + A^T)/2. "
        "Lazy alpha > 0 can improve numerical stability for periodic chains."
    )

settings = GraphSettings(directed=directed, keep_weights=keep_weights, lazy_alpha=float(lazy_alpha))

if not os.path.exists(file_path):
    st.error(f"Data file not found: {file_path}")
    st.stop()

A_raw = load_mtx_as_sparse(file_path)
A0 = preprocess_adjacency(A_raw, settings=settings)

# Baseline edges list
edges0 = edges_from_adjacency(A0, directed=settings.directed)

# Initialize session state for removals
if "removed_set" not in st.session_state:
    st.session_state.removed_set = set()
if "removed_edges" not in st.session_state:
    st.session_state.removed_edges = []

# Build current adjacency/graph from removals
A_state = apply_edge_removals(A0, st.session_state.removed_set, directed=settings.directed)
G_state = build_nx_graph(A_state, directed=settings.directed)

# Connectivity status
if settings.directed:
    connected_now = nx.is_strongly_connected(G_state) if G_state.number_of_nodes() > 0 else False
    conn_label = "Strongly connected"
else:
    connected_now = nx.is_connected(G_state) if G_state.number_of_nodes() > 0 else False
    conn_label = "Connected"

# Baseline K0 always computed on baseline A0 with same settings
scope0, K0, n0 = kemeny_score(A0, directed=settings.directed, lazy_alpha=settings.lazy_alpha)

# Current K on A_state
scope_now, K_now, n_now = kemeny_score(A_state, directed=settings.directed, lazy_alpha=settings.lazy_alpha)

# Top metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nodes", str(G_state.number_of_nodes()))
col2.metric("Edges (current)", str(G_state.number_of_edges()))
col3.metric(conn_label, "Yes" if connected_now else "No")
col4.metric("Scope used for K", f"{scope_now} (n={n_now})")

kcol1, kcol2 = st.columns(2)
with kcol1:
    st.metric("Baseline Kemeny constant K0", f"{K0:.6f}")
with kcol2:
    st.metric("Current Kemeny constant", f"{K_now:.6f}", delta=f"{(K_now - K0):+.6f}")



# Graph panel — 3D (clickable edges)
interactive_ok = render_graph_panel_3d_interactive(A0=A0, A_state=A_state, settings=settings)

# --- Selection + removal controls ---
selected_edges: set[tuple[int, int]] = set(st.session_state.get("selected_edges", set()))

st.write("**Selected edges:**")
if selected_edges:
    pretty_sel = [f"({u+1}, {v+1})" for (u, v) in sorted(selected_edges)]
    st.markdown(" ".join([f"`{e}`" for e in pretty_sel]))
else:
    st.caption("(none — click edges in the graph)")

b1, b2, b3 = st.columns([1, 1, 1])
with b1:
    remove_clicked = st.button("Remove selected edges", disabled=(not selected_edges))
with b2:
    clear_clicked = st.button("Clear selection", disabled=(not selected_edges))
with b3:
    reset_clicked = st.button("Reset all removals")

if clear_clicked:
    st.session_state.selected_edges = set()
    st.rerun()

if reset_clicked:
    st.session_state.removed_set = set()
    st.session_state.removed_edges = []
    st.session_state.selected_edges = set()
    st.rerun()

if remove_clicked and selected_edges:
    # Update removed_set
    removed_set = set(st.session_state.removed_set)
    for e in selected_edges:
        removed_set.add(normalize_edge(e[0], e[1], directed=settings.directed))

    # Maintain ordered list, allow deletions later
    removed_edges_order = list(st.session_state.removed_edges)
    for e in selected_edges:
        e = normalize_edge(e[0], e[1], directed=settings.directed)
        if e not in removed_edges_order:
            removed_edges_order.append(e)

    st.session_state.removed_set = removed_set
    st.session_state.removed_edges = removed_edges_order
    st.session_state.selected_edges = set()
    st.rerun()

# --- Removed edges list with per-edge Undo ---
st.write("**Removed edges (click to undo):**")
removed_edges_order: list[tuple[int, int]] = list(st.session_state.get("removed_edges", []))
if removed_edges_order:
    # Left-to-right layout in rows
    cols_per_row = 6
    for i in range(0, len(removed_edges_order), cols_per_row):
        row = removed_edges_order[i : i + cols_per_row]
        cols = st.columns(len(row))
        for c, (u, v) in zip(cols, row):
            label = f"↩ {u+1}-{v+1}" if not settings.directed else f"↩ {u+1}→{v+1}"
            if c.button(label, key=f"undo_edge_{u}_{v}"):
                st.session_state.removed_set.discard((u, v))
                st.session_state.removed_edges = [e for e in st.session_state.removed_edges if e != (u, v)]
                # If it was also selected, unselect
                sel = set(st.session_state.get("selected_edges", set()))
                sel.discard((u, v))
                st.session_state.selected_edges = sel
                st.rerun()
else:
    st.caption("(none)")

# --- Fallback removal UI (insurance) ---
with st.expander("Fallback: edge table removal (only needed if edge clicks don't work)", expanded=(not interactive_ok)):
    st.caption(
        "This table is a backup control. If your environment can't capture Plotly clicks, use this and your demo still works."
    )

    edges_all = pd.DataFrame(edges0, columns=["u", "v"])
    edges_all["u_show"] = edges_all["u"] + 1
    edges_all["v_show"] = edges_all["v"] + 1
    edges_all["Removed"] = [(e in st.session_state.removed_set) for e in edges0]

    edited = st.data_editor(
        edges_all[["u_show", "v_show", "Removed"]],
        use_container_width=True,
        hide_index=True,
        key="edge_editor_fallback",
        column_config={
            "u_show": st.column_config.NumberColumn("u", disabled=True),
            "v_show": st.column_config.NumberColumn("v", disabled=True),
            "Removed": st.column_config.CheckboxColumn("Removed ✅"),
        },
        height=350,
    )

    old_set = set(st.session_state.removed_set)
    new_set = {edges0[i] for i, flag in enumerate(edited["Removed"].tolist()) if bool(flag)}

    added = new_set - old_set
    removed = old_set - new_set

    if added or removed:
        st.session_state.removed_set = new_set

        # Update ordered list
        for e in edges0:
            if e in added and e not in st.session_state.removed_edges:
                st.session_state.removed_edges.append(e)
        if removed:
            st.session_state.removed_edges = [e for e in st.session_state.removed_edges if e not in removed]

        st.rerun()


st.subheader("Next-step recommendations (Top 5)")

objective = st.radio(
    "Objective",
    ["Disrupt communication (maximize K)", "Improve connectivity (minimize K)"],
    horizontal=True,
    index=0,
)
want_disrupt = objective.startswith("Disrupt")

st.caption(
    "Recommendations are computed for the NEXT removal given the current removed set. "
    "K is always computed on Full graph if possible; otherwise on LCC/LSCC."
)

# Candidate edges
edges_df = remaining_edges_df_from_edges0(edges0, st.session_state.removed_set)
if edges_df.empty:
    st.info("No candidates remaining.")
else:
    rows = []
    # Evaluate each candidate as next removal
    for _, r in edges_df.iterrows():
        e = (int(r["u"]), int(r["v"]))
        e = normalize_edge(e[0], e[1], directed=settings.directed)

        removed_next = set(st.session_state.removed_set)
        removed_next.add(e)

        A_next = apply_edge_removals(A0, removed_next, directed=settings.directed)
        scope_next, K_next, n_next = kemeny_score(
            A_next, directed=settings.directed, lazy_alpha=settings.lazy_alpha
        )

        rows.append(
            {
                "u": e[0],
                "v": e[1],
                "K_next": float(K_next),
                "dK_next": float(K_next - K_now),
                "dK_vs_K0": float(K_next - K0),
                "scope_next": scope_next,
                "comp_size_next": int(n_next),
            }
        )

    df = pd.DataFrame(rows)

    # Sort for recommendation direction
    df_sorted = df.sort_values("dK_next", ascending=(not want_disrupt))
    df_reco = df_sorted.head(5)
    df_opp = df_sorted.tail(5).sort_values("dK_next", ascending=(not want_disrupt))

    # Component shrink list (useful when removals isolate nodes)
    df_shrink = df.sort_values(["comp_size_next", "dK_next"], ascending=[True, False]).head(5)

    reco_title = "Recommended next edges" + (" (increase K most)" if want_disrupt else " (decrease K most)")
    opp_title = "Opposite-effect edges" + (" (decrease K most)" if want_disrupt else " (increase K most)")

    st.markdown(f"**{reco_title}**")
    st.dataframe(
        df_reco[["u", "v", "dK_next", "dK_vs_K0", "K_next", "scope_next", "comp_size_next"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(f"**{opp_title}**")
    st.dataframe(
        df_opp[["u", "v", "dK_next", "dK_vs_K0", "K_next", "scope_next", "comp_size_next"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Edges most likely to isolate actors / shrink the main component**")
    st.dataframe(
        df_shrink[["u", "v", "dK_next", "dK_vs_K0", "K_next", "scope_next", "comp_size_next"]],
        hide_index=True,
        use_container_width=True,
    )

# Interpretation (objective-aware)
st.subheader("Interpretation")
if want_disrupt:
    st.write(
        "- **Positive dK**: communication becomes slower (desired for disruption).\n"
        "- **Negative dK**: communication becomes faster (opposite of disruption).\n"
        "- **Smaller component size**: isolates actors / fragments the network."
    )
else:
    st.write(
        "- **Negative dK**: communication becomes faster (desired for improvement).\n"
        "- **Positive dK**: communication becomes slower (opposite of improvement).\n"
        "- **Smaller component size**: indicates fragmentation (usually undesirable)."
    )

st.divider()    

