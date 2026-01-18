import os
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

from src.arrest_algorithms import (
    load_graph,
    community_first_assignment,
    compute_regret,
    improve_with_greedy_moves,
    improve_with_balanced_swaps,
    run_part5_pipeline
)

FILE_PATH = "data/clandestine_network_example.mtx"

GRAPH_HEIGHT_PX = 850


# helpers for dashboard


@st.cache_resource
def load_graph_cached(file_path: str) -> nx.Graph:
    G = load_graph(file_path)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

@st.cache_data
def compute_fixed_layout(_graph: nx.Graph):
    pos = nx.spring_layout(_graph, seed=42, k=1.2)
    return {u: (pos[u][0] * 1000, pos[u][1] * 1000) for u in _graph.nodes()}



def assignment_to_df(assignment: dict[int, int]) -> pd.DataFrame:
    rows = [{"node": int(u), "dept": "A" if d == 0 else "B"} for u, d in assignment.items()]
    df = pd.DataFrame(rows).sort_values("node").reset_index(drop=True)
    return df


def colour_palette():
    return [
        "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
        "#EDC949", "#AF7AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    ]


def build_agraph_by_dept(
    G: nx.Graph,
    assignment: dict[int, int],
    layout_map: dict[int, tuple[float, float]],
):
    #safety check
    nodes_int = {int(n) for n in G.nodes()}
    assign_int = {int(n) for n in assignment.keys()}
    assert nodes_int == assign_int, "Assignment missing nodes or has extra nodes."

    nodes = []
    edges = []

    COLOR_A = "#4E79A7"   # blue
    COLOR_B = "#E15759"   # red
    COLOR_EDGE = "#444444"
    COLOR_CROSS = "#FF4444"

    for u in G.nodes():
        x, y = layout_map[u]
        dept = assignment.get(u, None)

        nodes.append(
            Node(
                id=str(u),
                label=str(u),
                x=x,
                y=y,
                size=20,
                color=COLOR_A if dept == 0 else COLOR_B,
                font={"color": "white", "size": 16},
            )
        )

    for u, v in G.edges():
        cross = assignment[u] != assignment[v]
        edges.append(
            Edge(
                source=str(u),
                target=str(v),
                color=COLOR_CROSS if cross else COLOR_EDGE,
                width=3 if cross else 1,
                opacity=1.0 if cross else 0.25,
                type="CURVED_CW" if cross else "STRAIGHT",
            )
        )

    config = Config(
        width="100%",
        height=GRAPH_HEIGHT_PX,
        directed=False,
        physics=False,
        staticGraph=True,
        nodeHighlightBehavior=True,
    )

    return agraph(nodes=nodes, edges=edges, config=config)


def communities_split_summary(communities, assignment: dict[int, int]) -> pd.DataFrame:
    rows = []
    for cid, comm in enumerate(communities):
        depts = {assignment[int(u)] for u in comm}
        rows.append({
            "community": cid,
            "size": len(comm),
            "in_A": sum(1 for u in comm if assignment[int(u)] == 0),
            "in_B": sum(1 for u in comm if assignment[int(u)] == 1),
            "split_across_depts": len(depts) > 1,
        })
    df = pd.DataFrame(rows).sort_values(["split_across_depts", "size"], ascending=[False, False]).reset_index(drop=True)
    return df






# APP
st.title("Part 5 — Department assignment dashboard")
import inspect
# st.write("Pipeline signature:", inspect.signature(run_part5_pipeline))

with st.expander("How to use this page", expanded=True):
    st.markdown(
        "1) Choose **Community detection** (how the network is grouped into factions).\n\n"
        "2) Adjust the settings that appear (some options only show for certain methods).\n\n"
        "3) Click **Run pipeline**.\n\n"
        "4) Read the **Regret summary and the graph** — lower Regret is better and red edges are.\n"
    )


graph = load_graph_cached(FILE_PATH)
N = graph.number_of_nodes()
# st.write("Has edge (50,16)?", graph.has_edge(50, 16)) just testing

st.sidebar.header("Controls")
st.sidebar.markdown("### Assignment settings")
# st.sidebar.caption(
#     "Choose how factions are detected and how strictly they are kept together "
#     "when assigning members to departments."
# )

community_method = st.sidebar.selectbox(
    "Community detection",
    ["Louvain", "Leiden", "Infomap", "Spectral", "Girvan-Newman"],
    key="community_method",
    help=(
        "How the network is initially grouped into factions. "
        "Louvain/Leiden find tightly connected groups. "
        "Infomap follows information flow. "
        "Spectral and Girvan–Newman create balanced structural splits."
    ),
)



# method specific params
resolution = 1.0
k = 2
assign_labels = "kmeans"
two_level = True

if community_method in ("Louvain", "Leiden"):
    resolution = st.sidebar.slider(
        "Community detail level",
        0.2, 1.0, 0.4, 0.1,
        key="resolution",
        help=(
            "Controls how detailed the detected factions are. "
            "Lower values produce fewer, larger factions. "
            "Higher values produce more, smaller factions."
        ),
    )

elif community_method in ("Spectral", "Girvan-Newman"):
    k = st.sidebar.slider(
        "Number of factions",
        2, 10, 2, 1,
        key="k_factions",
        help=(
            "Forces the algorithm to split the network into exactly k groups. "
            "Use small values for coarse splits, larger values for finer structure."
        ),
    )


elif community_method == "Infomap":
    two_level = st.sidebar.checkbox(
        "Two-level structure",
        value=True,
        key="infomap_two_level",
        help=(
            "If enabled, Infomap produces a simpler, high-level grouping. "
            "Disable to allow more detailed nested factions."
        ),
    )

same_comm_multiplier = st.sidebar.slider(
    "Same-community penalty",
    1.0, 5.0, 2.0, 0.5,
    key="same_comm_multiplier",
    help=(
        "How strongly connections inside the same faction are penalized "
        "when members are placed in different departments. "
        "Higher values keep factions together more strongly."
    ),
)


max_iters = st.sidebar.slider(
    "Optimization effort",
    10, 500, 100, 10,
    key="max_iters",
    help=(
        "How hard the system tries to improve the assignment. "
        "Higher values may find better solutions but take longer."
    ),
)


candidate_k = st.sidebar.slider(
    "Swap search width",
    4, 31, 12, 1,
    key="candidate_k",
    help=(
        "How many high-risk members are considered when swapping departments. "
        "Higher values explore more options but increase computation time."
    ),
)



run_clicked = st.sidebar.button("Run pipeline", type="primary", key="p5_run")



# height_px = st.sidebar.slider("Graph height", 400, 1000, 650, 50)
# same_comm_multiplier = st.sidebar.slider("Same-community penalty", 1.0, 5.0, 2.0, 0.5)
# max_iters = st.sidebar.slider("Max iterations (moves/swaps)", 10, 500, 100, 10)
# candidate_k = st.sidebar.slider("Swap candidate_k", 4, 31, 12, 1)
#
# run_clicked = st.sidebar.button("Run pipeline", type="primary")

if "result" not in st.session_state:
    st.session_state.result = None

# if run_clicked:
#     st.session_state.result = run_pipeline(
#         graph, same_comm_multiplier=same_comm_multiplier, max_iters=max_iters, candidate_k=candidate_k
#     )

# if run_clicked:
#     st.session_state.result = run_part5_pipeline(
#         FILE_PATH,
#         same_comm_multiplier=same_comm_multiplier,
#         seed=123,
#         max_move_iters=max_iters,
#         max_swap_iters=max_iters,
#         candidate_k=candidate_k,
#     )

if run_clicked:
    st.session_state.result = run_part5_pipeline(
        FILE_PATH,
        same_comm_multiplier=same_comm_multiplier,
        seed=123,
        max_move_iters=max_iters,
        max_swap_iters=max_iters,
        candidate_k=candidate_k,
        community_method=community_method,
        resolution=resolution,
        k=k,
        assign_labels=assign_labels,
        two_level=two_level,
    )



res = st.session_state.result

st.write(f"Nodes: **{N}**  |  Edges: **{graph.number_of_edges()}**")
layout_map = compute_fixed_layout(graph)

if res is None:
    st.info("Click **Run pipeline** to compute an assignment.")
    dummy_assignment = {int(u): 0 for u in graph.nodes()}  # all A just to render
    build_agraph_by_dept(
        graph,
        assignment=dummy_assignment,
        layout_map=layout_map,
    )
else:
    st.subheader("Regret summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial regret (R0)", f"{res['R0']:.2f}")
    col2.metric("After moves (R1)", f"{res['R1']:.2f}")
    col3.metric("After swaps (R2)", f"{res['R2']:.2f}")

    st.subheader("Graph (colored by Dept)")
    # html = build_pyvis_html_by_dept(graph, assignment=res["assignment2"], height_px=GRAPH_HEIGHT_PX )
    # st.components.v1.html(html, height=GRAPH_HEIGHT_PX , scrolling=False)

    build_agraph_by_dept(
        graph,
        assignment=res["assignment2"],
        layout_map=layout_map,
    )

    df = assignment_to_df(res["assignment2"])

    st.subheader("Department sizes")
    sizeA = (df["dept"] == "A").sum()
    sizeB = (df["dept"] == "B").sum()
    st.write(f"Dept A: **{sizeA}**  |  Dept B: **{sizeB}**  |  Capacity target: **{res['cap']}**")

    st.subheader(f"Communities ({community_method}) summary")
    sizes = sorted([len(c) for c in res["communities"]], reverse=True)
    st.write(f"Number of communities: **{len(res['communities'])}**")
    st.write("Sizes:", sizes)

    st.subheader("Assignment quality checks")

    cross_edges = sum(1 for u, v in graph.edges() if res["assignment2"][int(u)] != res["assignment2"][int(v)])
    total_edges = graph.number_of_edges()
    pct_cross = 0.0 if total_edges == 0 else 100.0 * cross_edges / total_edges

    st.write(f"- Cross-department links: **{cross_edges}/{total_edges}** (**{pct_cross:.1f}%**)")
    st.write(f"- Total regret (penalized cross links): **{res['R2']:.2f}**")

    df_split = communities_split_summary(res["communities"], res["assignment2"])
    st.write(
        f"- Communities split across departments: **{df_split['split_across_depts'].sum()} / {len(res['communities'])}**")

    st.subheader("Assignment table")
    st.dataframe(df, use_container_width=True)

