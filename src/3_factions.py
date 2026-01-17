import os
import random
from collections import defaultdict

import igraph as ig
import leidenalg as la
import networkx as nx
import streamlit as st
from infomap import Infomap
from networkx.algorithms.community import girvan_newman, modularity
from pyvis.network import Network
from scipy.io import mmread
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

FILE_PATH = "data/clandestine_network_example.mtx"
HEIGH_PX = 650


@st.cache_resource
def load_graph(file_path: str) -> nx.Graph:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find path: {file_path}")

    A = mmread(file_path).tocsr()
    G = nx.from_scipy_sparse_array(A)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def run_louvain(G: nx.Graph, resolution: float):
    return nx.community.louvain_communities(G, resolution=resolution, seed=42)


def run_leiden(G: nx.Graph, resolution: float):
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G.edges()]

    igG = ig.Graph(n=len(nodes), edges=edges, directed=False)

    part = la.find_partition(
        igG,
        la.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=42,
    )

    communities = []
    for block in part:
        communities.append({int(idx_to_node[i]) for i in block})

    return communities


def run_spectral(G: nx.Graph, k: int, assign_labels: str = "kmeans"):
    A = nx.to_scipy_sparse_array(G, format="csr")
    A.indices = A.indices.astype("int32", copy=False)
    A.indptr = A.indptr.astype("int32", copy=False)
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=42,
    )

    labels = sc.fit_predict(A)
    nodes = list(G.nodes())

    comms = defaultdict(set)
    for n, c in zip(nodes, labels):
        comms[int(c)].add(int(n))

    return list(comms.values())


def run_girvan_newman(G: nx.Graph, k: int):
    if k < 2:
        raise ValueError("k must be >= 2 for Girvan-Newman")

    comp_gen = girvan_newman(G)
    communities_tuple = None
    for _ in range(k - 1):
        communities_tuple = next(comp_gen)
    return [set(map(int, c)) for c in communities_tuple]


def run_infomap(G: nx.Graph):
    flags = "--two-level --silent"

    im = Infomap(flags)

    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        im.add_link(int(u), int(v), w)

    im.run()
    comm_to_nodes = defaultdict(set)
    for node in im.iterTree():
        if node.isLeaf():
            comm_to_nodes[node.moduleIndex()].add(int(node.physicalId))

    return list(comm_to_nodes.values())


def communities_to_membership(communities):
    membership = {}
    for cid, nodes in enumerate(communities):
        for n in nodes:
            membership[int(n)] = int(cid)
    return membership


def perturb_graph(G: nx.Graph, frac: float = 0.05, seed=42):
    random.seed(seed)
    Gp = G.copy()
    edges = list(Gp.edges())
    k = int(len(edges) * frac)
    remove = random.sample(edges, k)
    Gp.remove_edges_from(remove)
    return Gp


def compute_modularity(G, communities):
    return modularity(G, communities)


def communities_to_labels(communities, nodes):
    label_map = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            label_map[n] = cid
    return [label_map[n] for n in nodes]


def compute_nmi_ari(communities1, communities2, nodes):
    labels1 = communities_to_labels(communities1, nodes)
    labels2 = communities_to_labels(communities2, nodes)

    nmi = normalized_mutual_info_score(labels1, labels2)
    ari = adjusted_rand_score(labels1, labels2)
    return nmi, ari


def validate_partition_robustness(
    G: nx.Graph,
    algo_name: str,
    communities,
    frac: float = 0.05,
    seed: int = 42,
    resolution: float | None = None,
    k: int | None = None,
):
    nodes = list(G.nodes())

    initial_Q = compute_modularity(G, communities)

    Gp = perturb_graph(G, frac=frac, seed=seed)

    if algo_name == "Louvain":
        communities_p = run_louvain(
            Gp, resolution=resolution if resolution is not None else 0.4
        )
    elif algo_name == "Leiden":
        communities_p = run_leiden(
            Gp, resolution=resolution if resolution is not None else 0.4
        )
    elif algo_name == "Infomap":
        communities_p = run_infomap(Gp)
    elif algo_name == "Spectral":
        communities_p = run_spectral(
            Gp, k=k if k is not None else 2, assign_labels="kmeans"
        )
    elif algo_name == "Girvan-Newman":
        communities_p = run_girvan_newman(Gp, k=k if k is not None else 2)

    perturbation_Q = compute_modularity(Gp, communities_p)
    nmi, ari = compute_nmi_ari(communities, communities_p, nodes)

    return {
        "edges_removed": int(len(G.edges()) * frac),
        "Q_before": initial_Q,
        "Q_after": perturbation_Q,
        "delta_Q": initial_Q - perturbation_Q,
        "NMI": nmi,
        "ARI": ari,
        "communities_perturbed": communities_p,
        "G_perturbed": Gp,
    }


# APP


def colour_palette():
    return [
        "#4E79A7",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#59A14F",
        "#EDC949",
        "#AF7AA1",
        "#FF9DA7",
        "#9C755F",
        "#BAB0AC",
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ]


def build_pyvis_html(
    G: nx.Graph,
    membership: dict[int, int] | None,
    height_px: int = HEIGH_PX,
    show_labels: bool = False,
):
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#000000",
        font_color="#ffffff",
        directed=False,
    )

    net.barnes_hut()
    net.set_options("""
    {
      "layout": {
        "improvedLayout": true
      },
      "physics": {
        "enabled": false,
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      }
    }
    """)
    net.from_nx(G)

    colours = colour_palette()

    for node in net.nodes:
        nid = int(node["id"])

        if membership is None:
            node["color"] = "#DDDDDD"
            node["title"] = f"Node {nid}"
            node["opacity"] = 1
        else:
            cid = membership.get(nid, 0)
            node["color"] = colours[cid % len(colours)]
            node["title"] = f"Node {nid}<br>Community {cid}"

        node["label"] = str(nid)
        node["font"] = {"color": "#ffffff", "size": 18}

        node["size"] = 15
        node["borderWidth"] = 0
        node["borderWidthSelected"] = 0
        node["shadow"] = False
    for e in net.edges:
        e["color"] = "#AAAAAA"
        e["width"] = 1.5

    html = net.generate_html()

    html = html.replace(
        "</head>",
        """
        <style>
          html, body {
            background: #000 !important;
            margin: 0 !important;
            padding: 0 !important;
          }
          * {
            border: none !important;
            outline: none !important;
            box-shadow: none !important;
          }
          #mynetwork, #network, .vis-network {
            background: #000 !important;
          }
          canvas {
            background: #000 !important;
          }
        </style>
        </head>
        """,
    )

    html = html.replace(
        "<body>", "<body style='margin:0; padding:0; background:#000;'>"
    )
    return html


if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True


st.title("FactionSelection")
if st.session_state.show_tutorial:
    with st.expander("📘 Quick guide", expanded=True):
        st.markdown(
            """
            **Step 1 — Choose an algorithm (left panel)**  
            Pick a method to detect factions in the network.

            **Step 2 — Adjust settings (optional)**  
            For **Louvain** and **Leiden**, resolution controls the level of detail of the communities.
            Lower values produce larger communities, while higher values produce smaller communities.  
            For **Spectral** and **Girvan–Newman**, k controls the number of communities \(k\).

            **Step 3 — Click Run**  
            The network will update with color-coded factions.

            **Step 4 — Explore details**  
            Open the “Community details” section below the graph to inspect members.
            """
        )

G = load_graph(FILE_PATH)

if "communities" not in st.session_state:
    st.session_state.communities = None
    st.session_state.membership = None
    st.session_state.algo_used = None
if "disturbance_results" not in st.session_state:
    st.session_state.disturbance_results = None

st.sidebar.header("Controls")
algo = st.sidebar.selectbox(
    "Algorithm",
    ["Louvain", "Leiden", "Infomap", "Spectral", "Girvan-Newman"],
    help="Choose the community detection method used to identify factions.",
)

algo_descriptions = {
    "Louvain": "Groups nodes that interact more with each other than with the rest of the network.",
    "Leiden": "Improves Louvain by producing more stable and better connected groups.",
    "Infomap": "Groups nodes based on how information flows through the network.",
    "Spectral": "Divides the network into a chosen number of groups based on overall structure.",
    "Girvan-Newman": "Finds groups by cutting the most important links between them.",
}

st.sidebar.caption(algo_descriptions[algo])

if algo in ("Louvain", "Leiden"):
    resolution = st.sidebar.slider(
        "Resolution (granularity)",
        0.2,
        1.0,
        0.4,
        0.1,
        help="Higher values produce smaller communities. Lower values produce larger communities.",
    )
elif algo == "Spectral":
    k = st.sidebar.slider("k", 2, 10, 2, 1, help="Number of Communities")
    assign_labels = "kmeans"
elif algo == "Girvan-Newman":
    k = st.sidebar.slider("k", 2, 10, 2, 1, help="Number of Communities")
run_clicked = st.sidebar.button("Run", type="primary")
run_disturbance = False
show_perturbed_graph = False
if st.session_state.get("communities") is not None:
    st.sidebar.divider()

    run_disturbance = st.sidebar.button(
        "Run disturbance test (remove 5%)",
        help="Removes 5% of links and re-evaluates faction stability.",
    )

    show_perturbed_graph = st.sidebar.checkbox(
        "Show perturbed graph",
        value=False,
        help="Display the network after 5% of links are removed.",
    )


if run_clicked:
    if algo == "Louvain":
        communities = run_louvain(G, resolution=resolution)
    elif algo == "Leiden":
        communities = run_leiden(G, resolution=resolution)
    elif algo == "Infomap":
        communities = run_infomap(G)
    elif algo == "Spectral":
        communities = run_spectral(G, k=k, assign_labels=assign_labels)
    elif algo == "Girvan-Newman":
        communities = run_girvan_newman(G, k=k)
    st.session_state.communities = communities
    st.session_state.membership = communities_to_membership(communities)
    st.session_state.algo_used = algo
    st.session_state.disturbance_results = None
    st.rerun()

if st.session_state.communities is None:
    st.info("Pick an algorithm and click **Run**.")
    html = build_pyvis_html(
        G,
        membership=None,
        height_px=HEIGH_PX,
    )
    st.components.v1.html(html, height=HEIGH_PX, scrolling=False)
else:
    communities = st.session_state.communities
    membership = st.session_state.membership
    Q_before = float(modularity(G, communities))
    st.success(
        "Factions detected. Open **Community details** below to inspect members."
    )
    st.subheader("Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("Modularity (Q)", f"{Q_before:.3f}")
    c2.metric("Detected factions", f"{len(communities)}")
    c3.metric("Largest faction", f"{max(len(c) for c in communities)}")

    st.subheader(f"{st.session_state.algo_used}")
    html = build_pyvis_html(
        G,
        membership=membership,
        height_px=HEIGH_PX,
    )
    st.components.v1.html(html, height=HEIGH_PX, scrolling=False)
    st.subheader("Robustness check (optional)")

    st.caption(
        "This test removes 5% of links to simulate missing information and checks whether the detected factions remain stable."
    )

    if run_disturbance:
        st.session_state.disturbance_results = validate_partition_robustness(
            G,
            algo_name=st.session_state.algo_used,
            communities=communities,
            frac=0.05,
            seed=42,
            resolution=resolution
            if (st.session_state.algo_used in ("Louvain", "Leiden"))
            else None,
            k=k
            if (st.session_state.algo_used in ("Spectral", "Girvan-Newman"))
            else None,
        )

    if st.session_state.disturbance_results is None:
        st.info(
            "Click **Run disturbance test (remove 5%)** in the sidebar to evaluate stability."
        )
    else:
        results = st.session_state.disturbance_results
        st.subheader("Stability summary")

        if results["NMI"] >= 0.8 and results["ARI"] >= 0.8:
            st.success(
                "Very stable: factions barely change after removing 5% of links."
            )
        elif results["NMI"] >= 0.6 and results["ARI"] >= 0.6:
            st.warning(
                "Moderately stable: some reshuffling after removing 5% of links."
            )
        else:
            st.error("Unstable: factions change a lot under missing links.")

        col1, col2, col3 = st.columns(3)
        col1.metric("Modularity Q (before)", f"{results['Q_before']:.3f}")
        col2.metric("Modularity Q (after)", f"{results['Q_after']:.3f}")
        col3.metric("ΔQ", f"{results['delta_Q']:.3f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("NMI", f"{results['NMI']:.3f}")
        col5.metric("ARI", f"{results['ARI']:.3f}")
        col6.metric("Edges removed", f"{results['edges_removed']}")

        st.caption(
            "Higher NMI/ARI means factions stay similar after disturbance. "
            "A smaller ΔQ means the overall structure is stable under missing links."
        )

        if show_perturbed_graph:
            st.subheader("Perturbed network")
            membership_p = communities_to_membership(results["communities_perturbed"])
            html_p = build_pyvis_html(
                results["G_perturbed"], membership_p, height_px=HEIGH_PX
            )
            st.components.v1.html(html_p, height=HEIGH_PX, scrolling=False)

    with st.expander("↓ Community details", expanded=False):
        st.write(f"Number of communities: **{len(communities)}**")
        sizes = sorted([len(c) for c in communities], reverse=True)
        st.write("Sizes:", sizes)

        idx = st.selectbox("Select a community", list(range(len(communities))))
        nodes = sorted(list(communities[idx]))

        search = st.text_input("Search node ID")
        if search.strip().isdigit():
            nid = int(search)
            st.write(f"Node {nid} is in community **{membership.get(nid, 'N/A')}**")

        st.dataframe({"node_id": nodes})
##### VALIDATION #####
