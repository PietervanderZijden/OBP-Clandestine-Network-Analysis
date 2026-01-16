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
- Read-only graph visualization (edges disappear or show faded when removed).

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

from streamlit_agraph import agraph, Node, Edge, Config

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
# Visualization (read-only graph)
# -------------------------
@st.cache_data(show_spinner=False)
def fixed_layout_from_A(A_data, A_indices, A_indptr, shape, seed: int = 42) -> dict[int, tuple[float, float]]:
    """Fixed 2D layout for baseline graph. Uses an undirected view for stable positions."""
    A = sp.csr_matrix((A_data, A_indices, A_indptr), shape=shape)
    # undirected view for layout
    Au = (A + A.T) * 0.5
    Au = Au.tocsr()
    Au.setdiag(0.0)
    Au.eliminate_zeros()
    G = nx.from_scipy_sparse_array(Au)
    pos = nx.spring_layout(G, seed=seed, k=1.2)
    return {int(n): (float(x) * 1000.0, float(y) * 1000.0) for n, (x, y) in pos.items()}


def render_graph_panel(A0: sp.csr_matrix, A_state: sp.csr_matrix, settings: GraphSettings):
    st.subheader("Network view (current state)")
    show_removed = st.checkbox("Show removed edges (faded)", value=False)

    # Prepare layout (baseline)
    layout = fixed_layout_from_A(A0.data, A0.indices, A0.indptr, A0.shape)

    n = A_state.shape[0]
    nodes_viz: list[Node] = []
    for i in range(n):
        x, y = layout.get(i, (0.0, 0.0))
        nodes_viz.append(
            Node(
                id=str(i),
                label=str(i),
                x=x,
                y=y,
                size=18,
                color=COLOR_STEEL,
                font={"color": "white", "face": "monospace", "size": 14},
            )
        )

    # Active edges from current adjacency
    edges_viz: list[Edge] = []
    coo = A_state.tocoo()
    if settings.directed:
        for u, v in zip(coo.row, coo.col):
            u = int(u)
            v = int(v)
            if u == v:
                continue
            edges_viz.append(
                Edge(
                    source=str(u),
                    target=str(v),
                    color=COLOR_WIRE,
                    width=1,
                    opacity=0.6,
                    type="STRAIGHT",
                )
            )
    else:
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
            edges_viz.append(
                Edge(
                    source=str(a),
                    target=str(b),
                    color=COLOR_WIRE,
                    width=1,
                    opacity=0.6,
                    type="STRAIGHT",
                )
            )

    # Removed edges (optional, faded)
    if show_removed:
        removed_set = st.session_state.get("removed_set", set())
        for (u, v) in removed_set:
            edges_viz.append(
                Edge(
                    source=str(int(u)),
                    target=str(int(v)),
                    color=COLOR_ALERT,
                    width=5,
                    opacity=0.55,
                    type="sTRIAIGHT",
                )
            )

    cfg = Config(
        width=1100,
        height=520,
        directed=settings.directed,
        physics=False,
        staticGraph=True,
    )
    agraph(nodes=nodes_viz, edges=edges_viz, config=cfg)


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


# Graph panel (read-only)
render_graph_panel(A0=A0, A_state=A_state, settings=settings)

st.write("**Removed edges (ordered):**")
if st.session_state.removed_edges:
    st.code("\n".join([str(e) for e in st.session_state.removed_edges]), language=None)
else:
    st.caption("(none)")



# Two-column layout: left selection, right recommendations
left, right = st.columns([0.6, 1.4], gap="large")

with left:
    st.subheader("Remove a communication link (table selection)")
    st.caption("Select an edge to remove. The graph above updates; recommendations update on the right.")

    edges_df = remaining_edges_df_from_edges0(edges0, st.session_state.removed_set)
    if edges_df.empty:
        st.info("No edges left to remove.")
    else:
        event = st.dataframe(
            edges_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            height=900,
        )
        sel = event.selection.get("rows", [])
        if sel:
            idx = sel[0]
            u = int(edges_df.loc[idx, "u"])
            v = int(edges_df.loc[idx, "v"])
            e = normalize_edge(u, v, directed=settings.directed)

            if e not in st.session_state.removed_set:
                st.session_state.removed_set.add(e)
                st.session_state.removed_edges.append(e)
                st.rerun()



    b1, b2 = st.columns(2)
    with b1:
        if st.button(
            "Undo last removal",
            use_container_width=True,
            disabled=(len(st.session_state.removed_edges) == 0),
        ):
            last = st.session_state.removed_edges.pop()
            st.session_state.removed_set.remove(last)
            st.rerun()
    with b2:
        if st.button("Reset (original network)", use_container_width=True):
            st.session_state.removed_edges = []
            st.session_state.removed_set = set()
            st.rerun()


with right:
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
            df_shrink[["u", "v", "comp_size_next", "dK_next", "dK_vs_K0", "K_next", "scope_next"]],
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
