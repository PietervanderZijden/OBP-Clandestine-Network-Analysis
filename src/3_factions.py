import os
from collections import defaultdict

import igraph as ig
import leidenalg as la
import networkx as nx
import streamlit as st
from infomap import Infomap
from networkx.algorithms.community import girvan_newman
from pyvis.network import Network
from scipy.io import mmread
from sklearn.cluster import SpectralClustering

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
            cid = membership[nid]
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

if "communities" not in st.session_state:
    st.session_state.communities = None
    st.session_state.membership = None
    st.session_state.algo_used = None

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

    st.success(
        "Factions detected. Open **Community details** below to inspect members."
    )

    st.subheader(f"{st.session_state.algo_used}")
    html = build_pyvis_html(
        G,
        membership=membership,
        height_px=HEIGH_PX,
    )
    st.components.v1.html(html, height=HEIGH_PX, scrolling=False)

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
