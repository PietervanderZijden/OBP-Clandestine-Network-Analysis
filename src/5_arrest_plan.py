import os
from collections import defaultdict

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network
from scipy.io import mmread

from src.arrest_algorithms import (
    load_graph,
    community_first_assignment,
    compute_regret,
    improve_with_greedy_moves,
    improve_with_balanced_swaps,
)

FILE_PATH = "data/clandestine_network_example.mtx"



# helpers for dashboard


@st.cache_resource
def load_graph_cached(file_path: str) -> nx.Graph:
    G = load_graph(file_path)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def assignment_to_df(assignment: dict[int, int]) -> pd.DataFrame:
    rows = [{"node": int(u), "dept": "A" if d == 0 else "B"} for u, d in assignment.items()]
    df = pd.DataFrame(rows).sort_values("node").reset_index(drop=True)
    return df


def colour_palette():
    return [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]


def build_pyvis_html_by_dept(
    G: nx.Graph,
    assignment: dict[int, int] | None,
    height_px: int = 650
):
    net = Network(
        height=f"{height_px}px",
        width="100%",
        bgcolor="#000000",
        font_color="#ffffff",
        directed=False,
    )
    net.barnes_hut()
    net.from_nx(G)

    colours = {0: "#4E79A7", 1: "#E15759"}  # A blue, B red-ish

    for node in net.nodes:
        nid = int(node["id"])
        node["label"] = str(nid)
        node["font"] = {"color": "#ffffff", "size": 18}
        node["size"] = 40
        node["borderWidth"] = 0
        node["shadow"] = False

        if assignment is None:
            node["color"] = "#DDDDDD"
            node["title"] = f"Node {nid}"
        else:
            d = int(assignment[nid])
            node["color"] = colours[d]
            node["title"] = f"Node {nid}<br>Dept {'A' if d==0 else 'B'}"

    for e in net.edges:
        e["color"] = "#AAAAAA"
        e["width"] = 1.5

    html = net.generate_html()
    html = html.replace("<body>", "<body style='margin:0; padding:0; background:#000;'>")
    return html



#  pipeline runner
def run_pipeline(graph, same_comm_multiplier, max_iters, candidate_k):
    # communities + comm_id
    communities = nx.community.louvain_communities(graph, seed=123, weight="weight")
    comm_id = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            comm_id[int(n)] = int(cid)

    cap = graph.number_of_nodes() // 2
    init = community_first_assignment(graph, communities, capacity=cap)
    assignment0 = init.assignment.copy()

    R0 = compute_regret(graph, assignment0, comm_id, same_comm_multiplier)

    assignment1 = improve_with_greedy_moves(
        graph, assignment0.copy(), comm_id, cap, same_comm_multiplier, max_iters=max_iters
    )
    R1 = compute_regret(graph, assignment1, comm_id, same_comm_multiplier)

    assignment2 = improve_with_balanced_swaps(
        graph, assignment1.copy(), comm_id, same_comm_multiplier,
        max_iters=max_iters, candidate_k=candidate_k
    )
    R2 = compute_regret(graph, assignment2, comm_id, same_comm_multiplier)

    return {
        "communities": communities,
        "comm_id": comm_id,
        "cap": cap,
        "init_sizes": (init.sizeA, init.sizeB),
        "R0": R0,
        "R1": R1,
        "R2": R2,
        "assignment0": assignment0,
        "assignment1": assignment1,
        "assignment2": assignment2,
    }



# APP
st.title("Part 5 — Department assignment dashboard")

graph = load_graph_cached(FILE_PATH)
N = graph.number_of_nodes()

st.sidebar.header("Controls")
height_px = st.sidebar.slider("Graph height", 400, 1000, 650, 50)
same_comm_multiplier = st.sidebar.slider("Same-community penalty", 1.0, 5.0, 2.0, 0.5)
max_iters = st.sidebar.slider("Max iterations (moves/swaps)", 10, 500, 100, 10)
candidate_k = st.sidebar.slider("Swap candidate_k", 4, 31, 12, 1)

run_clicked = st.sidebar.button("Run pipeline", type="primary")

if "result" not in st.session_state:
    st.session_state.result = None

if run_clicked:
    st.session_state.result = run_pipeline(
        graph, same_comm_multiplier=same_comm_multiplier, max_iters=max_iters, candidate_k=candidate_k
    )

res = st.session_state.result

st.write(f"Nodes: **{N}**  |  Edges: **{graph.number_of_edges()}**")

if res is None:
    st.info("Click **Run pipeline** to compute an assignment.")
    html = build_pyvis_html_by_dept(graph, assignment=None, height_px=height_px)
    st.components.v1.html(html, height=height_px, scrolling=False)
else:
    st.subheader("Regret summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial regret (R0)", f"{res['R0']:.2f}")
    col2.metric("After moves (R1)", f"{res['R1']:.2f}")
    col3.metric("After swaps (R2)", f"{res['R2']:.2f}")

    st.subheader("Graph (colored by Dept)")
    html = build_pyvis_html_by_dept(graph, assignment=res["assignment2"], height_px=height_px)
    st.components.v1.html(html, height=height_px, scrolling=False)

    st.subheader("Assignment table")
    df = assignment_to_df(res["assignment2"])
    st.dataframe(df, use_container_width=True)

    st.subheader("Department sizes")
    sizeA = (df["dept"] == "A").sum()
    sizeB = (df["dept"] == "B").sum()
    st.write(f"Dept A: **{sizeA}**  |  Dept B: **{sizeB}**  |  Capacity target: **{res['cap']}**")

    st.subheader("Communities (Louvain) summary")
    sizes = sorted([len(c) for c in res["communities"]], reverse=True)
    st.write(f"Number of communities: **{len(res['communities'])}**")
    st.write("Sizes:", sizes)
