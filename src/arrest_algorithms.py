from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import math
import networkx as nx
from scipy.io import mmread
from pathlib import Path



Node = int
Dept = int  # Dept A = 0, Dept B = 1

#DATA INIT

def _resolve_path(path: str):
    path = Path(path)
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / path).resolve()

def load_graph(path: str):
    #File not found error fix
    data_path = _resolve_path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"File does not exist: {data_path}")

    #Read data
    data = mmread(str(data_path)).tocsr()
    if data.shape[0] != data.shape[1]:
        raise ValueError(f"Matrix not square")

    #Making sure graph is undirected
    data = data.maximum(data.T)
    #Creates the nx graph with edges having weights
    graph = nx.from_scipy_sparse_array(data, edge_attribute='weight')

    return graph

#HELPER FUNCTIONS

def edge_penalty(u, v, original_w, comm_id ,same_comm_multiplier = 2.0,): #TODO original weight is techincally always 1 so might be deleted
    """Function returning higher penalty if nodes are in the same community"""
    if comm_id[u] == comm_id[v]:
        return original_w * same_comm_multiplier
    return original_w

def compute_regret(graph, assignment, comm_id, same_comm_multiplier = 2.0):
    """Function that computes regret per connection between nodes (members)"""
    R = 0.0
    for u, v, data in graph.edges(data=True):
        w = float(data.get("weight", 1.0))
        w_eff = edge_penalty(u, v, w, comm_id, same_comm_multiplier)
        if assignment[u] != assignment[v]:
            R += w_eff
    return R

#COMMMUNIT ASSIGNMENT PER DEPARTMENT

@dataclass
class AssignResult: #TODO renaming maybe idk
    assignment: Dict[Node, Dept]
    sizeA: int
    sizeB: int
    capacity: int

def build_comm_id(communities):
    comm_id = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            comm_id[n] = cid
    return comm_id

def community_first_assignment(graph, communities, capacity): #TODO capacity var to delete
    num_of_nodes = graph.number_of_nodes()
    cap = num_of_nodes / 2

    #sort comms and change to a list of sets
    comms = sorted((set(c) for c in communities), key=len, reverse=True)

    #Dept init
    A: Set[Node] = set()
    B: Set[Node] = set()

    for comm in comms:
        # Choose the less used dept
        preferred = A if len(A) <= len(B) else B
        other = B if preferred is A else A

        # Try to fit whole community
        if len(preferred) + len(comm) <= cap:
            preferred |= comm #add all nodes
            continue
        if len(other) + len(comm) <= cap:
            other |= comm
            continue

        # Otherwise split the community, low internal degree nodes first
        internal_deg = []
        for u in comm:
            deg_in = sum(1 for v in graph.neighbors(u) if v in comm) #counting neighbours
            internal_deg.append((deg_in, u))
        internal_deg.sort()  # lowest internal degree first

        for _, u in internal_deg:
            # keep balance if possible, but never exceed cap
            if len(A) < cap and (len(A) <= len(B) or len(B) >= cap):
                A.add(u)
            elif len(B) < cap:
                B.add(u)
            else:
                raise RuntimeError("Capacity met before assigning all nodes.")

    # Build assignment dict
    assignment: Dict[Node, Dept] = {u: 0 for u in A}
    assignment.update({u: 1 for u in B})

    return AssignResult(assignment=assignment, sizeA=len(A), sizeB=len(B), capacity=cap)

#NODE SWAP IMPROVEMENT

def delta_regret_if_flip(graph, node, assignment: Dict[Node, Dept], comm_id, same_comm_multiplier= 2.0):
    """
    Compute ΔR if node u is flipped to the other department.
    Only edges incident to u can change.
    """
    dept_u = assignment[node]
    dept_u_flipped = 1 - dept_u
    dR = 0.0

    for v, data in graph[node].items():
        weight = float(data.get("weight", 1.0))
        effective_weight  = edge_penalty(node, v, weight, comm_id, same_comm_multiplier)

        before = 1.0 if assignment[node] != assignment[v] else 0.0
        after = 1.0 if dept_u_flipped != assignment[v] else 0.0
        dR += effective_weight  * (after - before)

    return dR

#TODO these are not yet done

def improve_with_greedy_moves(graph, assignment: Dict[Node, Dept], comm_id: Dict[Node, int], capacity,same_comm_multiplier, max_iters):
    """
    Greedy node flips that reduce regret while respecting capacity.
    """
    sizeA = sum(1 for node in assignment if assignment[node] == 0)
    sizeB = len(assignment) - sizeA

    for _ in range(max_iters):
        best_node = None
        best_dR = 0.0  # only accept flips with dR < 0

        for node in graph.nodes():
            dR = delta_regret_if_flip(graph, node, assignment, comm_id, same_comm_multiplier) # dR = change in regret
            if dR >= best_dR:
                continue

            dept_node = assignment[node]

            # If node moves A->B: sizeA-1, sizeB+1 if B->A: sizeA+1, sizeB-1
            deltaA = -1 if dept_node == 0 else 1
            newA = sizeA + deltaA
            newB = sizeB - deltaA

            if newA <= capacity and newB <= capacity:
                best_node = node
                best_dR = dR

        if best_node is None:
            break

        dept_node = assignment[best_node]
        deltaA = -1 if dept_node == 0 else 1

        assignment[best_node] = 1 - dept_node
        sizeA += deltaA
        sizeB -= deltaA

    return assignment


def improve_with_balanced_swaps(
    G: nx.Graph,
    assignment: Dict[Node, Dept],
    comm_id: Dict[Node, int],
    same_comm_multiplier: float = 2.0,
    max_iters: int = 100,
    candidate_k: int = 12
) -> Dict[Node, Dept]:
    """
    Swap-based refinement: swap one node in A with one node in B (keeps sizes fixed).
    We restrict to boundary nodes and consider top contributors to regret.
    """

    def boundary_nodes(dept: Dept) -> List[Node]:
        out = []
        for u in G.nodes():
            if assignment[u] != dept:
                continue
            # boundary if it has a neighbor in the other dept
            if any(assignment[v] != dept for v in G.neighbors(u)):
                out.append(u)
        return out

    def top_cost_nodes(nodes: List[Node]) -> List[Node]:
        costs = []
        for u in nodes:
            c = 0.0
            for v, data in G[u].items():
                if assignment[u] != assignment[v]:
                    w = float(data.get("weight", 1.0))
                    c += edge_penalty(u, v, w, comm_id, same_comm_multiplier)
            costs.append((c, u))
        costs.sort(reverse=True)
        return [u for _, u in costs[:candidate_k]]

    for _ in range(max_iters):
        A_nodes = top_cost_nodes(boundary_nodes(0))
        B_nodes = top_cost_nodes(boundary_nodes(1))

        best_pair = None
        best_dR = 0.0

        for u in A_nodes:
            dR_u = delta_regret_if_flip(G, u, assignment, comm_id, same_comm_multiplier)
            for v in B_nodes:
                dR_v = delta_regret_if_flip(G, v, assignment, comm_id, same_comm_multiplier)

                dR_pair = dR_u + dR_v

                #
                if G.has_edge(u, v):
                    w = float(G[u][v].get("weight", 1.0))
                    w_eff = edge_penalty(u, v, w, comm_id, same_comm_multiplier)
                    dR_pair += 2.0 * w_eff

                if dR_pair < best_dR:
                    best_dR = dR_pair
                    best_pair = (u, v)

        if best_pair is None:
            break

        u, v = best_pair
        assignment[u] = 1
        assignment[v] = 0

    return assignment


#Put together

def run_part5_pipeline(
    mtx_path: str,
    same_comm_multiplier: float = 2.0,
    seed: int = 42
) -> None:
    graph = load_graph(mtx_path)
    N = graph.number_of_nodes()
    cap = math.ceil(N / 2) #max department capacity

    # Clustering logic
    #communities = get_communities_placeholder(G, k=2, seed=seed) #TODO Change to louvain clusters
    communities = nx.community.louvain_communities(graph, seed=123, weight="weight")

    comm_id = build_comm_id(communities)

    init = community_first_assignment(graph, communities, capacity=cap)
    assignment = init.assignment

    #Assignment process
    R0 = compute_regret(graph, assignment, comm_id, same_comm_multiplier)

    assignment = improve_with_greedy_moves(graph, assignment, comm_id, cap, same_comm_multiplier, max_iters=200)
    R1 = compute_regret(graph, assignment, comm_id, same_comm_multiplier)

    assignment = improve_with_balanced_swaps(graph, assignment, comm_id, same_comm_multiplier, max_iters=100)
    R2 = compute_regret(graph, assignment, comm_id, same_comm_multiplier)

    sizeA = sum(1 for u in assignment if assignment[u] == 0)
    sizeB = N - sizeA

    print(f"N={N}, capacity per dept={cap}")
    print(f"Initial:  sizeA={init.sizeA}, sizeB={init.sizeB}, regret={R0:.2f}")
    print(f"Moves:    sizeA={sum(1 for u in assignment if assignment[u]==0)}, sizeB={N-sum(1 for u in assignment if assignment[u]==0)}, regret={R1:.2f}")
    print(f"Swaps:    sizeA={sizeA}, sizeB={sizeB}, regret={R2:.2f}")

    # Example: list 10 highest-risk cross edges for dashboard highlighting
    # cross_edges = []
    # for u, v, data in graph.edges(data=True):
    #     if assignment[u] != assignment[v]:
    #         w = float(data.get("weight", 1.0))
    #         w_eff = edge_penalty(u, v, w, comm_id, same_comm_multiplier)
    #         cross_edges.append((w_eff, u, v))
    # cross_edges.sort(reverse=True)
    # print("Top cross-department edges (penalized weight, u, v):")
    # for row in cross_edges[:10]:
    #     print(row)


if __name__ == "__main__":
    run_part5_pipeline("data/clandestine_network_example.mtx")