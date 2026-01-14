import os
from collections import defaultdict

import networkx as nx
import streamlit as st
from infomap import Infomap
from pyvis.network import Network
from scipy.io import mmread

FILE_PATH = "data/clandestine_network_example.mtx"


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


def run_infomap(G: nx.Graph, two_level: bool):
    flags = "--silent"
    if two_level:
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
    height_px: int = 650,
    # physics: bool = True,
    show_labels: bool = False,
):
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#000000",
        font_color="#ffffff",
        directed=False,
    )
    # net.toggle_physics(physics)
    net.barnes_hut()
    net.set_options("""
    {
      "layout": {
        "improvedLayout": true
      },
      "physics": {
        "enabled": true,
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
            # no communities yet: grey nodes
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

    # # html = net.generate_html()
    # toggle_js = """
    # <button onclick="toggleLabels()"
    # style="
    # position:fixed;
    # top:10px;
    # right:10px;
    # z-index:999;
    # background:#222;
    # color:white;
    # border:1px solid white;
    # padding:6px 10px;
    # cursor:pointer;">
    # Toggle Labels
    # </button>

    # <script>
    # var labelsVisible = true;

    # function toggleLabels() {
    #   labelsVisible = !labelsVisible;

    #   network.body.data.nodes.forEach(function(n){
    #     if(labelsVisible){
    #       n.label = String(n.id);
    #     } else {
    #       n.label = "";
    #     }
    #   });

    #   network.body.data.nodes.update(network.body.data.nodes.get());
    # }
    # </script>
    # """
    html = net.generate_html()

    html = html.replace(
        "<body>", "<body style='margin:0; padding:0; background:#000;'>"
    )

    html = html.replace(
        "network = new vis.Network(container, data, options);",
        """
    network = new vis.Network(container, data, options);
    network.fit();

    // --- UI button injected into the same container as the canvas ---
    container.style.position = "relative";
    var btn = document.createElement("button");
    btn.innerText = "Toggle Labels";
    btn.style.position = "absolute";
    btn.style.top = "12px";
    btn.style.right = "12px";
    btn.style.zIndex = "999999";
    btn.style.background = "#222";
    btn.style.color = "#fff";
    btn.style.border = "1px solid #fff";
    btn.style.padding = "8px 12px";
    btn.style.cursor = "pointer";
    btn.style.borderRadius = "8px";
    btn.style.fontSize = "14px";
    container.appendChild(btn);

    var labelsVisible = true;
    btn.onclick = function () {
      labelsVisible = !labelsVisible;

      network.body.data.nodes.forEach(function(n){
        n.label = labelsVisible ? String(n.id) : "";
      });

      network.body.data.nodes.update(network.body.data.nodes.get());
    };
    """,
    )

    return html


st.title("FactionSelection")

G = load_graph(FILE_PATH)

st.sidebar.header("Controls")
algo = st.sidebar.selectbox("Algorithm", ["Louvain", "Infomap"])
height_px = st.sidebar.slider("Graph height", 400, 1000, 650, 50)

if algo == "Louvain":
    resolution = st.sidebar.slider("Resolution (granularity)", 0.2, 2.0, 1.0, 0.1)
else:
    two_level = st.sidebar.checkbox("Two-level partition", value=True)

run_clicked = st.sidebar.button("Run", type="primary")

if "communities" not in st.session_state:
    st.session_state.communities = None
    st.session_state.membership = None
    st.session_state.algo_used = None

if run_clicked:
    if algo == "Louvain":
        communities = run_louvain(G, resolution=resolution)
    else:
        communities = run_infomap(G, two_level=two_level)

    st.session_state.communities = communities
    st.session_state.membership = communities_to_membership(communities)
    st.session_state.algo_used = algo


if st.session_state.communities is None:
    st.info("Pick an algorithm and click **Run**.")
    html = build_pyvis_html(
        G,
        membership=None,
        height_px=height_px,
        # physics=physics,
        # show_labels=show_labels,
    )
    st.components.v1.html(html, height=height_px, scrolling=False)

else:
    communities = st.session_state.communities
    membership = st.session_state.membership

    st.subheader(f"Interactive graph — {st.session_state.algo_used}")
    html = build_pyvis_html(
        G,
        membership=membership,
        height_px=height_px,
        # physics=physics,
        # show_labels=show_labels,
    )
    st.components.v1.html(html, height=height_px, scrolling=False)

    st.subheader("Community summary")
    sizes = sorted([len(c) for c in communities], reverse=True)
    st.write(f"Number of communities: **{len(communities)}**")
    st.write("Sizes:", sizes)

    idx = st.selectbox("Inspect a community", list(range(len(communities))))
    nodes = sorted(list(communities[idx]))
    st.write(f"Community {idx} size: **{len(nodes)}**")
    st.write(nodes)
