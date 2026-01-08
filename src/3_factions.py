import os

import networkx as nx
from scipy.io import mmread


def faction_detection_louvain():
    file_path = "data/clandestine_network_example.mtx"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find path: {file_path}")
    A = mmread(file_path).tocsr()

    G = nx.from_scipy_sparse_array(A)

    G.remove_edges_from(nx.selfloop_edges(G))

    communities = nx.community.louvain_communities(G, seed=123, weight="weight")

    print(f"n_communities={len(communities)}")
    for i, c in enumerate(communities, start=1):
        print(f"Community {i} (size={len(c)}): {sorted(c)}")


if __name__ == "__main__":
    faction_detection_louvain()
