import os
import numpy as np
import scipy.sparse as sp
import networkx as nx
import streamlit as st
from scipy.io import mmread
from scipy.sparse.csgraph import connected_components

from ui_components import apply_tactical_theme


# -------------------------
# Core math
# -------------------------
def random_walk_transition(A: sp.csr_matrix) -> sp.csr_matrix:
    """
    P = D^{-1}A (row-stochastic). Add self-loop to dangling nodes (safety).
    """
    A = A.tocsr().copy()
    row_sum = np.array(A.sum(axis=1)).ravel()
    dangling = np.where(row_sum == 0)[0]
    if dangling.size > 0:
        A[dangling, dangling] = 1.0
        A.eliminate_zeros()
        row_sum = np.array(A.sum(axis=1)).ravel()

    inv_row = np.zeros_like(row_sum, dtype=float)
    inv_row[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    return (sp.diags(inv_row) @ A).tocsr()


def stationary_distribution(P: sp.csr_matrix,
                            tol: float = 1e-12,
                            max_iter: int = 200_000) -> np.ndarray:
    """
    Compute stationary distribution pi numerically for a (finite) Markov chain.
    We find pi such that pi = pi P, sum(pi)=1, pi>=0.
    Numerically we run power iteration on P^T (equivalently on pi as a row vector).
    Assumes:
      - P is row-stochastic (rows sum to 1)
      - chain is irreducible/ergodic in the component considered (or at least convergent)
    """
    if not sp.isspmatrix_csr(P):
        P = P.tocsr()

    n = P.shape[0]
    # start from uniform distribution
    pi = np.full(n, 1.0 / n, dtype=float)

    PT = P.transpose().tocsr()

    for _ in range(max_iter):
        pi_next = PT @ pi  # column update for pi^T
        s = pi_next.sum()
        if s <= 0:
            raise ValueError("Numerical issue: stationary vector sum <= 0.")
        pi_next /= s

        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            pi = pi_next
            return pi_next

        pi = pi_next

    raise RuntimeError("stationary_distribution did not converge; check connectivity/aperiodicity.")


def kemeny_constant(P: sp.csr_matrix, pi: np.ndarray) -> float:
    """
    K = tr( (I - P + 1*pi)^(-1) )
    Here pi is a 1D array; 1*pi is outer product of ones (column) with pi (row).
    Uses dense inversion because n=62 is small.
    """
    n = P.shape[0]
    I = np.eye(n)
    Pd = P.toarray()
    one_pi = np.ones((n, 1)) @ pi.reshape(1, -1)
    Z = np.linalg.inv(I - Pd + one_pi)
    return float(np.trace(Z))


def compute_kemeny(A: sp.csr_matrix) -> float:
    P = random_walk_transition(A)
    pi = stationary_distribution(A)
    return kemeny_constant(P, pi)


def remove_edge(A: sp.csr_matrix, u: int, v: int) -> sp.csr_matrix:
    """
    Remove undirected edge (u,v).
    """
    B = A.tolil(copy=True)
    B[u, v] = 0
    B[v, u] = 0
    B = B.tocsr()
    B.eliminate_zeros()
    return B


def largest_component_nodes(A: sp.csr_matrix) -> np.ndarray:
    n_comp, labels = connected_components(A, directed=False, return_labels=True)
    sizes = np.bincount(labels)
    lab = int(np.argmax(sizes))
    return np.where(labels == lab)[0]


# -------------------------
# Data loading
# -------------------------
@st.cache_data
def load_graph_from_mtx(path: str):
    A = mmread(path)
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()

    # Build undirected graph
    G = nx.from_scipy_sparse_array(A)
    # Ensure no self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    A = nx.to_scipy_sparse_array(G, format="csr")

    edges = sorted({tuple(sorted((int(u), int(v)))) for (u, v) in G.edges()})
    return A, G, edges



# -------------------------
# Streamlit Page
# -------------------------
apply_tactical_theme()

st.title("Kemeny Constant — Edge Removal Analysis")
st.caption("Remove a communication link (edge) and observe how global connectivity changes.")

file_path = "data/clandestine_network_example.mtx"
if not os.path.exists(file_path):
    st.error(f"Data file not found: {file_path}")
    st.stop()

A, G, edges = load_graph_from_mtx(file_path)

colA, colB, colC = st.columns(3)
colA.metric("Nodes", str(G.number_of_nodes()))
colB.metric("Edges", str(G.number_of_edges()))
colC.metric("Connected", "Yes" if nx.is_connected(G) else "No")

# Baseline
K0 = compute_kemeny(A)
st.metric("Baseline Kemeny constant K0", f"{K0:.6f}")

st.divider()

# Edge selector (robust UI)
edge_labels = [f"({u}, {v})" for (u, v) in edges]
chosen = st.selectbox("Select an edge to remove", edge_labels, index=0)
u, v = edges[edge_labels.index(chosen)]

A_removed = remove_edge(A, u, v)
G_removed = nx.from_scipy_sparse_array(A_removed)
connected = nx.is_connected(G_removed)

if connected:
    K_new = compute_kemeny(A_removed)
    st.success("Graph remains connected after removing this edge.")
    st.metric("Kemeny after removal", f"{K_new:.6f}", delta=f"{(K_new - K0):+.6f}")
else:
    st.warning("Removing this edge disconnects the network.")
    nodes = largest_component_nodes(A_removed)
    A_lcc = A_removed[nodes, :][:, nodes].tocsr()
    K_lcc = compute_kemeny(A_lcc)
    st.metric("Kemeny on largest connected component", f"{K_lcc:.6f}", delta=f"{(K_lcc - K0):+.6f}")
    st.write(f"Largest component size after removal: **{nodes.size}**")

st.subheader("Interpretation")
st.write(
    "- **Positive ΔK** → slower information flow (connectivity worse)\n"
    "- **Negative ΔK** → faster information flow (connectivity better)\n"
    "- **Disconnect** → operationally critical link (isolates actors)"
)


