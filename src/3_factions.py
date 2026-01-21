import random
from collections import defaultdict

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import streamlit as st
from infomap import Infomap
from networkx.algorithms.community import girvan_newman, modularity
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from streamlit_agraph import Config, Edge, Node, agraph

# --- NEW IMPORT ---
from src.data_manager import get_active_network
from ui_components import COLOR_STEEL, COLOR_WIRE

HEIGH_PX = 650


def theme_style():
    base = st.get_option("theme.base")
    if base == "light":
        return {
            "bg": "#ffffff",
            "font": "#111827",
            "edge": "#111827",
            "edge_opacity": 0.18,
            "neutral_node": "#6b7280",
        }
    return {
        "bg": "#000000",
        "font": "#ffffff",
        "edge": "#ffffff",
        "edge_opacity": 0.35,
        "neutral_node": "#9ca3af",
    }


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


def align_membership_to_reference(
    membership_ref: dict[int, int],
    membership_new: dict[int, int],
) -> dict[int, int]:
    """
    Remap community ids in membership_new so they best match membership_ref.
    Works even if #communities differ.
    """
    nodes = sorted(set(membership_ref) & set(membership_new))

    ref_labels = [membership_ref[n] for n in nodes]
    new_labels = [membership_new[n] for n in nodes]

    ref_ids = sorted(set(ref_labels))
    new_ids = sorted(set(new_labels))

    ref_to_i = {c: i for i, c in enumerate(ref_ids)}
    new_to_j = {c: j for j, c in enumerate(new_ids)}

    M = np.zeros((len(ref_ids), len(new_ids)), dtype=int)
    for r, c in zip(ref_labels, new_labels):
        M[ref_to_i[r], new_to_j[c]] += 1

    row_ind, col_ind = linear_sum_assignment(-M)

    new_to_ref = {}
    for i, j in zip(row_ind, col_ind):
        new_to_ref[new_ids[j]] = ref_ids[i]

    next_id = (max(ref_ids) + 1) if ref_ids else 0
    for c in new_ids:
        if c not in new_to_ref:
            new_to_ref[c] = next_id
            next_id += 1

    aligned = {n: new_to_ref[membership_new[n]] for n in membership_new}
    return aligned


def run_spectral(G: nx.Graph, k: int, assign_labels: str = "kmeans"):
    nodes = list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format="csr")
    A.indices = A.indices.astype("int32", copy=False)
    A.indptr = A.indptr.astype("int32", copy=False)
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels=assign_labels,
        random_state=42,
    )

    labels = sc.fit_predict(A)

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
    removed_norm = {tuple(sorted((int(u), int(v)))) for u, v in remove}
    return Gp, removed_norm


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


def algo_run_signature(algo: str, param_value: float | int | None = None) -> str:
    if algo in ("Louvain", "Leiden"):
        return (
            f"{algo}(res={float(param_value):.2f})"
            if param_value is not None
            else f"{algo}(res=n/a)"
        )
    if algo in ("Spectral", "Girvan-Newman"):
        return (
            f"{algo}(k={int(param_value)})"
            if param_value is not None
            else f"{algo}(k=n/a)"
        )
    return f"{algo}"


def pct_nodes_changed_between_memberships(
    membership_a: dict[int, int],
    membership_b: dict[int, int],
    nodes_list,
) -> float:
    """
    Returns % of nodes whose community id differs between two memberships.
    Note: this assumes community ids are comparable across the two runs.
    If label switching is possible, align membership_b to membership_a first.
    """
    common = [
        int(n) for n in nodes_list if int(n) in membership_a and int(n) in membership_b
    ]
    if not common:
        return 0.0

    changed = sum(1 for n in common if int(membership_a[n]) != int(membership_b[n]))
    return 100.0 * changed / len(common)


def compute_run_summary(G: nx.Graph, communities):
    Q = float(compute_modularity(G, communities))
    sizes = sorted((len(c) for c in communities), reverse=True)
    return {
        "modularity_Q": Q,
        "n_communities": int(len(communities)),
        "largest_community": int(sizes[0]) if sizes else 0,
        "sizes": sizes,
    }


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

    Gp, removed_edges = perturb_graph(G, frac=frac, seed=seed)

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
        "removed_edges": removed_edges,
    }


def compute_changed_nodes(
    membership_before: dict[int, int],
    membership_after: dict[int, int],
):
    common = set(membership_before).intersection(membership_after)
    return {
        n: (membership_before[n], membership_after[n])
        for n in common
        if membership_before[n] != membership_after[n]
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
    ]


def build_agraph_factions(
    G: nx.Graph,
    membership: dict[int, int] | None,
    layout_map: dict[int, tuple[float, float]],
    height_px: int = HEIGH_PX,
    changed_nodes: dict[int, tuple[int, int]] | None = None,
    removed_edges: set[tuple[int, int]] | None = None,
    show_removed: bool = False,
):
    changed_nodes = changed_nodes or {}
    colours = colour_palette()
    removed_edges = removed_edges or set()

    style = theme_style()
    nodes = []
    edges = []

    for u in G.nodes():
        x, y = layout_map[int(u)]
        nid = int(u)

        if membership is None:
            node_color = COLOR_STEEL
        else:
            cid = int(membership.get(nid, 0))
            node_color = colours[cid % len(colours)]
        title_txt = f"Node {nid}"
        if membership is not None:
            cid = int(membership.get(nid, 0))
            title_txt += f"\nCommunity: {cid}"

        if nid in changed_nodes:
            old_c, new_c = changed_nodes[nid]
            title_txt += (
                f"\nChanged cluster after disturbance\nPrevious: {old_c}\nNew: {new_c}"
            )
        node_kwargs = dict(
            id=str(nid),
            label=str(nid),
            title=title_txt,
            x=float(x),
            y=float(y),
            size=20,
            color=node_color,
            font={"color": "white", "size": 16, "vadjust": -38},
        )

        if nid in changed_nodes:
            node_kwargs.update(
                {
                    "shape": "diamond",
                    "size": 26,
                    "borderWidth": 4,
                }
            )

        nodes.append(Node(**node_kwargs))

    for u, v in G.edges():
        e = tuple(sorted((int(u), int(v))))
        is_removed = e in removed_edges

        if is_removed and not show_removed:
            continue

        edges.append(
            Edge(
                source=str(int(u)),
                target=str(int(v)),
                color=COLOR_WIRE,
                width=1,
                opacity=0.08 if is_removed else style["edge_opacity"],
                type="STRAIGHT",
                dashes=True if is_removed else False,
            )
        )

    config = Config(
        width="100%",
        height=height_px,
        directed=False,
        physics=False,
        staticGraph=True,
        nodeHighlightBehavior=True,
        backgroundColor=style["bg"],
        visjs_config={"interaction": {"hover": True}},
    )

    return agraph(nodes=nodes, edges=edges, config=config)


def render_changed_legend():
    st.markdown(
        """
        <div style="
            display:inline-flex;
            align-items:center;
            gap:10px;
            padding:10px 12px;
            border-radius:8px;
            border:1px solid rgba(128,128,128,0.35);
            background: rgba(0,0,0,0.35);
            color: inherit;
            font-size: 14px;
            margin-top: 8px;
        ">
            <div style="
                width:12px;
                height:12px;
                transform: rotate(45deg);
                border: 2px solid #ff4d4d;
                background: #ffffff;
                display:inline-block;
            "></div>
            <div>Node changed cluster after disturbance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if "show_tutorial" not in st.session_state:
    st.session_state.show_tutorial = True
if "communities" not in st.session_state:
    st.session_state.communities = None
    st.session_state.membership = None
    st.session_state.algo_used = None
if "scroll_to_disturbance" not in st.session_state:
    st.session_state.scroll_to_disturbance = False
if "disturbance_results" not in st.session_state:
    st.session_state.disturbance_results = None
if "compare_store" not in st.session_state:
    st.session_state.compare_store = {}

st.title("Faction Selection")

G, metadata = get_active_network()
G.remove_edges_from(nx.selfloop_edges(G))
tab_explore, tab_compare = st.tabs(["Explore", "Compare"])
with tab_explore:
    if st.session_state.show_tutorial:
        with st.expander("📘 Quick guide", expanded=True):
            st.markdown(
                """
                **Step 1 — Choose an algorithm (left panel)** Pick a method to detect factions in the network.

                **Step 2 — Adjust settings (optional)** Parameter _Resolution_ controls the level of detail and number of the communities. 
                For **Louvain** and **Leiden**, lower values produce fewer, larger communities while larger values create more, smaller communities. 

                **Step 3 — Click Run** The network will update with color-coded factions.

                **Step 4 — Compare** Click on "Add to Comparison" to compare different algorithms. The results of the comparison are displayed in the Compare tab.

                **Step 5 — Robustness Check** Click on the "Run Disturbance test" button to remove 5% of the edges of the graph and recompute the evaluation metrics

                _Hint: click on the ? to learn more about the methods and results_
                """
            )

    @st.cache_data
    def compute_layout(_graph, source_name):
        pos = nx.spring_layout(_graph, seed=42)
        return {int(u): (pos[u][0] * 1000, pos[u][1] * 1000) for u in _graph.nodes()}

    layout_map = compute_layout(G, metadata["name"])

    st.sidebar.header("Controls")
    target_name = metadata.get("name", "Unknown")

    st.sidebar.markdown(f"""
    <div style="font-family:'Share Tech Mono'; font-size:12px; color:#8b949e; margin-bottom:10px;">
        dataset: <span style="color:#58a6ff">{target_name}</span>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Roles settings")

    st.sidebar.markdown(
        """
    <style>
    .tooltip {
      position: relative;
      display: inline-block;
      cursor: help;
      font-weight: 700;
    }

    .tooltip .tooltiptext {
      visibility: hidden;
      width: 200px;
      background-color: #555;
      color: #fff;
      text-align: left;
      border-radius: 6px;
      padding: 8px;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 0;
      opacity: 0;
      transition: opacity 0.2s;
    }

    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>

    <div class="tooltip">
    Algorithm
    <span class="tooltiptext">
    Choose the community detection method used to identify factions.
    </span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    algo = st.sidebar.selectbox(
        "",
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
    param_value = None
    if algo in ("Louvain", "Leiden"):
        param_value = st.sidebar.slider(
            "Resolution",
            0.2,
            1.0,
            0.4,
            0.1,
            help="Higher values produce smaller communities. Lower values produce larger communities.",
        )
    elif algo in ("Spectral", "Girvan-Newman"):
        param_value = st.sidebar.slider(
            "Resolution", 2, 10, 2, 1, help="Number of Communities"
        )
        assign_labels = "kmeans"
    else:
        param_value = None

    run_clicked = st.sidebar.button("Run", type="primary")
    run_disturbance = False
    show_perturbed_graph = False
    if st.session_state.get("communities") is not None:
        st.sidebar.markdown(
            "<hr style='margin:8px 0; border-color:#333;'>", unsafe_allow_html=True
        )
        run_disturbance = st.sidebar.button(
            "Run robustness test",
            help="Removes 5% of links and re-evaluates faction stability.",
        )
        if run_disturbance:
            st.session_state.scroll_to_disturbance = True
        show_perturbed_graph = st.sidebar.checkbox(
            "Show perturbed graph",
            value=False,
            help="Display the network after 5% of links are removed.",
        )
        show_removed_edges = st.sidebar.checkbox(
            "Show removed edges",
            value=False,
            help="Overlay the removed links as dashed edges.",
        )

    if run_clicked:
        if algo == "Louvain":
            communities = run_louvain(G, resolution=param_value)
        elif algo == "Leiden":
            communities = run_leiden(G, resolution=param_value)
        elif algo == "Infomap":
            communities = run_infomap(G)
        elif algo == "Spectral":
            communities = run_spectral(G, k=param_value, assign_labels=assign_labels)
        elif algo == "Girvan-Newman":
            communities = run_girvan_newman(G, k=param_value)
        st.session_state.communities = communities
        st.session_state.membership = communities_to_membership(communities)
        st.session_state.algo_used = algo
        st.session_state.param_value = param_value
        st.session_state.param_value
        st.session_state.disturbance_results = None
        st.rerun()

    if st.session_state.communities is None:
        st.info("Pick an algorithm and click **Run**.")
        build_agraph_factions(
            G, membership=None, layout_map=layout_map, height_px=HEIGH_PX
        )
    else:
        communities = st.session_state.communities
        membership = st.session_state.membership
        Q_before = float(modularity(G, communities))
        st.success(
            "Factions detected. Click on **Add to comparison** to compare the output or on **Run robustness test** to test the robustness of the Algorithm."
        )
        st.subheader("Results")
        c1, c2, c3 = st.columns(3)
        c1.metric("Modularity (Q)", f"{Q_before:.3f}")
        c2.metric("Detected factions", f"{len(communities)}")
        c3.metric("Largest faction", f"{max(len(c) for c in communities)}")

        st.subheader(f"{st.session_state.algo_used}")
        build_agraph_factions(
            G, membership=membership, layout_map=layout_map, height_px=HEIGH_PX
        )
        st.markdown('<div id="disturbance-results"></div>', unsafe_allow_html=True)

        sig = algo_run_signature(
            st.session_state.algo_used, param_value=st.session_state.param_value
        )

        b1, b2, b3 = st.columns([1.2, 1.2, 2])
        with b1:
            add_to_compare = st.button("Add to comparison", use_container_width=True)
        with b2:
            go_compare = st.button("Compare selected", use_container_width=True)
        with b3:
            st.caption(f"Selected: **{len(st.session_state.compare_store)}**")

        if add_to_compare:
            pv = st.session_state.get("param_value", None)

            st.session_state.compare_store[sig] = {
                "signature": sig,
                "algo": st.session_state.algo_used,
                "params": {
                    "param_value": pv,  # the one slider value you call "Resolution"
                    "param_name": (
                        "resolution"
                        if st.session_state.algo_used in ("Louvain", "Leiden")
                        else "k"
                        if st.session_state.algo_used in ("Spectral", "Girvan-Newman")
                        else None
                    ),
                },
                "communities": communities,
                "membership": st.session_state.membership,
                "summary": compute_run_summary(G, communities),
            }

            st.toast(f"Added: {sig}", icon="✅")

        if go_compare:
            st.toast("Open the **Compare** tab to view selected runs →", icon="➡️")

        st.subheader("Robustness check")
        st.caption(
            "_Hint_ : Click on **Show perturbed graph** to see which nodes changed clusters"
        )
        if st.session_state.get("scroll_to_disturbance", False):
            st.components.v1.html(
                """
                <script>
                  const el = window.parent.document.getElementById("disturbance-results");
                  if (el) el.scrollIntoView({behavior: "smooth"});
                </script>
                """,
                height=0,
            )
            st.session_state.scroll_to_disturbance = False

        st.caption(
            "This test removes 5% of links to simulate missing information and checks whether the detected factions remain stable."
        )

        if run_disturbance:
            pv = st.session_state.get("param_value", None)

            st.session_state.disturbance_results = validate_partition_robustness(
                G,
                algo_name=st.session_state.algo_used,
                communities=communities,
                frac=0.05,
                seed=42,
                resolution=float(pv)
                if st.session_state.algo_used in ("Louvain", "Leiden")
                else None,
                k=int(pv)
                if st.session_state.algo_used in ("Spectral", "Girvan-Newman")
                else None,
            )

            st.session_state.scroll_to_disturbance = True
            st.rerun()
        if st.session_state.disturbance_results is None:
            st.info(
                "Click **Run robustness test (remove 5%)** in the sidebar to evaluate stability."
            )
        else:
            results = st.session_state.disturbance_results
            membership_before = st.session_state.membership
            membership_after_raw = communities_to_membership(
                results["communities_perturbed"]
            )
            membership_after = align_membership_to_reference(
                membership_before, membership_after_raw
            )

            changed_nodes = compute_changed_nodes(membership_before, membership_after)
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
                membership_p = communities_to_membership(
                    results["communities_perturbed"]
                )
                build_agraph_factions(
                    G,  # <- original graph so removed edges exist
                    membership=membership_after,
                    layout_map=layout_map,
                    height_px=HEIGH_PX,
                    changed_nodes=changed_nodes,
                    removed_edges=results.get("removed_edges", set()),
                    show_removed=show_removed_edges,
                )

                if len(changed_nodes) > 0:
                    render_changed_legend()
                else:
                    st.success("No node changed cluster")

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
# with tab_compare:
#     with st.expander("How to read these results", expanded=True):
#         st.markdown("""
#         **Modularity (Q)** Measures how strongly the network is divided into factions.  A high modularity score means the network has dense connections within the communities and sparse connections between them.

#         **NMI (Normalized Mutual Information)** Measures how similar two faction results are.
#         • 1.0 = identical
#         • 0.0 = completely different

#         **ARI (Adjusted Rand Index)** Measures agreement between two results while correcting for chance.
#         • 1.0 = identical
#         • ~0 = random similarity

#         High NMI & ARI means two algorithms see the network in the same way.
#         """)

#     st.subheader("Algorithm comparison")

#     if len(st.session_state.compare_store) == 0:
#         st.info(
#             "No algorithms selected yet. Run an algorithm in **Explore** and click **Add to comparison**."
#         )
#     else:
#         c1, c2 = st.columns([1, 3])
#         with c1:
#             if st.button("🧹 Clear selection", use_container_width=True):
#                 st.session_state.compare_store = {}
#                 st.rerun()
#         with c2:
#             st.caption(
#                 "Compare modularity and agreement (NMI/ARI) across selected partitions."
#             )

#         rows = []
#         for key, item in st.session_state.compare_store.items():
#             s = item["summary"]
#             rows.append(
#                 {
#                     "Run": item["signature"],
#                     "Algorithm": item["algo"],
#                     "Modularity Q": round(s["modularity_Q"], 3),
#                     "# Factions": s["n_communities"],
#                     "Largest faction": s["largest_community"],
#                 }
#             )

#         rows = sorted(rows, key=lambda r: r["Modularity Q"], reverse=True)

#         st.markdown("### Summary table")
#         st.dataframe(rows, use_container_width=True, hide_index=True)

#         st.markdown("### Agreement between partitions (NMI / ARI)")

#         nodes = list(G.nodes())
#         keys = list(st.session_state.compare_store.keys())
#         memberships = {k: st.session_state.compare_store[k]["membership"] for k in keys}

#         n = len(keys)
#         nmi_mat = [[0.0] * n for _ in range(n)]
#         ari_mat = [[0.0] * n for _ in range(n)]

#         def membership_to_label_list(memb: dict[int, int], nodes_list):
#             return [int(memb.get(int(node), -1)) for node in nodes_list]

#         label_lists = {k: membership_to_label_list(memberships[k], nodes) for k in keys}

#         for i in range(n):
#             for j in range(n):
#                 if i == j:
#                     nmi_mat[i][j] = 1.0
#                     ari_mat[i][j] = 1.0
#                 elif j > i:
#                     nmi = normalized_mutual_info_score(
#                         label_lists[keys[i]], label_lists[keys[j]]
#                     )
#                     ari = adjusted_rand_score(
#                         label_lists[keys[i]], label_lists[keys[j]]
#                     )
#                     nmi_mat[i][j] = nmi_mat[j][i] = float(nmi)
#                     ari_mat[i][j] = ari_mat[j][i] = float(ari)
#         avg_nmi = {keys[i]: sum(nmi_mat[i]) / len(nmi_mat[i]) for i in range(n)}

#         max_val = max(avg_nmi.values())

#         best_agreement = [k for k, v in avg_nmi.items() if abs(v - max_val) < 1e-6]

#         if len(best_agreement) == 1:
#             st.success(f"Most consistent (Agreement): **{best_agreement[0]}**")
#         else:
#             st.success(
#                 "Most consistent (Agreement): "
#                 + ", ".join(f"**{b}**" for b in best_agreement)
#                 + " (tie)"
#             )

#         st.markdown(
#             """
#         **NMI matrix**
#         <span title="Measures how much information two partitions share. 1 = identical, 0 = unrelated.">ℹ️</span>
#         """,
#             unsafe_allow_html=True,
#         )

#         st.dataframe(
#             [
#                 {"": keys[i], **{keys[j]: round(nmi_mat[i][j], 3) for j in range(n)}}
#                 for i in range(n)
#             ],
#             use_container_width=True,
#             hide_index=True,
#         )

#         st.markdown(
#             """
#             **ARI matrix**
#             <span title="Measures exact agreement between node pairs, corrected for chance. 1 = identical.">ℹ️</span>
#             """,
#             unsafe_allow_html=True,
#         )

#         st.dataframe(
#             [
#                 {"": keys[i], **{keys[j]: round(ari_mat[i][j], 3) for j in range(n)}}
#                 for i in range(n)
#             ],
#             use_container_width=True,
#             hide_index=True,
#         )

#         st.markdown("### Manage selection")
#         remove_key = st.selectbox("Remove a run", ["(none)"] + keys)
#         if remove_key != "(none)":
#             if st.button("Remove selected run"):
#                 st.session_state.compare_store.pop(remove_key, None)
#                 st.rerun()
with tab_compare:
    st.subheader("Compare results")

    st.caption(
        "This page helps choose a faction result that is both **meaningful** (high modularity) "
        "and **reliable** (consistent with other runs)."
    )

    with st.expander("How to read these metrics", expanded=False):
        st.markdown(
            """
**Modularity (Q)**  
How cleanly the network splits into factions.  
Higher = stronger separation (dense links inside factions, fewer links between factions).

**NMI (Normalized Mutual Information)**  
How similar two results are (1 = identical, 0 = unrelated).

**ARI (Adjusted Rand Index)**  
Agreement of node-pair grouping, corrected for chance (1 = identical, ~0 = random).
            """
        )

    if len(st.session_state.compare_store) == 0:
        st.info(
            "No runs selected yet. Go to **Explore**, run an algorithm, then click **Add to comparison**."
        )
    else:
        # --- Data prep ---
        nodes = list(G.nodes())
        keys = list(st.session_state.compare_store.keys())
        memberships = {k: st.session_state.compare_store[k]["membership"] for k in keys}

        def membership_to_label_list(memb: dict[int, int], nodes_list):
            return [int(memb.get(int(node), -1)) for node in nodes_list]

        label_lists = {k: membership_to_label_list(memberships[k], nodes) for k in keys}

        n = len(keys)
        nmi_mat = [[0.0] * n for _ in range(n)]
        ari_mat = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    nmi_mat[i][j] = 1.0
                    ari_mat[i][j] = 1.0
                elif j > i:
                    nmi = normalized_mutual_info_score(
                        label_lists[keys[i]], label_lists[keys[j]]
                    )
                    ari = adjusted_rand_score(
                        label_lists[keys[i]], label_lists[keys[j]]
                    )
                    nmi_mat[i][j] = nmi_mat[j][i] = float(nmi)
                    ari_mat[i][j] = ari_mat[j][i] = float(ari)

        avg_nmi = {keys[i]: sum(nmi_mat[i]) / n for i in range(n)}
        avg_ari = {keys[i]: sum(ari_mat[i]) / n for i in range(n)}

        # Build summary rows (add consensus)
        rows = []
        for k in keys:
            item = st.session_state.compare_store[k]
            s = item["summary"]
            rows.append(
                {
                    "Run": item["signature"],
                    "Algorithm": item["algo"],
                    "Q_raw": float(s["modularity_Q"]),
                    "Modularity Q": round(s["modularity_Q"], 3),
                    "# Factions": int(s["n_communities"]),
                    "Largest faction": int(s["largest_community"]),
                    "Consensus (avg NMI)": round(avg_nmi[k], 3),
                    "ARI_raw": float(avg_ari[k]),
                    "Consensus (avg ARI)": round(avg_ari[k], 3),
                }
            )

        # Simple interpretability label
        def interpretability_label(num_factions: int) -> str:
            if num_factions <= 4:
                return "High"
            if num_factions <= 8:
                return "Medium"
            return "Low"

        for r in rows:
            r["Interpretability"] = interpretability_label(r["# Factions"])

        # Sort: modularity first, then agreement
        rows = sorted(rows, key=lambda r: (r["Q_raw"], r["ARI_raw"]), reverse=True)

        # --- Top actions ---
        top_left, top_right = st.columns([1, 2])
        with top_left:
            if st.button("🧹 Clear all selected runs", use_container_width=True):
                st.session_state.compare_store = {}
                st.rerun()
        with top_right:
            st.caption(f"Selected runs: **{len(rows)}**")

        # --- Recommendation card ---
        best = rows[0]["Run"]
        best_row = rows[0]
        EPS_Q = 0.01
        EPS_ARI = 0.01
        best_row = rows[0]

        tied = [
            r
            for r in rows
            if abs(r["Q_raw"] - best_row["Q_raw"]) <= EPS_Q
            and abs(r["ARI_raw"] - best_row["ARI_raw"]) <= EPS_ARI
        ]

        if len(tied) > 1:
            st.info(
                "Top recommendation is a tie: "
                + ", ".join(f"**{r['Run']}**" for r in tied)
                + ". Use interpretability (# factions) to choose."
            )
        else:
            st.success(
                f"Recommended: **{best}**  \n"
                f"Why: high separation (**Q={best_row['Modularity Q']}**) and strong consistency "
                f"(**avg ARI={best_row['Consensus (avg ARI)']}**). "
                f"Interpretability: **{best_row['Interpretability']}**."
            )

        # --- Summary table ---
        st.markdown("### Overview")
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )

        # --- Pairwise comparison (much more intuitive than matrices) ---
        st.markdown("### Do two runs agree?")
        if len(keys) == 1:
            st.info("Add at least **two** runs to compare agreement.")
        else:
            cA, cB = st.columns(2)
            with cA:
                run_a = st.selectbox("Run A", keys, index=0)
            with cB:
                run_b = st.selectbox("Run B", keys, index=1)

            i, j = keys.index(run_a), keys.index(run_b)
            nmi = float(nmi_mat[i][j])
            ari = float(ari_mat[i][j])
            memb_a = st.session_state.compare_store[run_a]["membership"]
            memb_b_raw = st.session_state.compare_store[run_b]["membership"]

            memb_b = align_membership_to_reference(memb_a, memb_b_raw)

            pct_changed = pct_nodes_changed_between_memberships(memb_a, memb_b, nodes)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("NMI", f"{nmi:.3f}")
            m2.metric("ARI", f"{ari:.3f}")
            common = [int(x) for x in nodes if int(x) in memb_a and int(x) in memb_b]
            n_changed = int(round(pct_changed / 100 * len(common)))
            m3.metric("% Nodes that changed faction", f"{pct_changed:.1f}%")
            m3.caption(f"≈ {n_changed} nodes out of {len(common)}")

            # Optional: a human label based on pct
            if pct_changed <= 5:
                label = "Very stable"
            elif pct_changed <= 15:
                label = "Mostly stable"
            elif pct_changed <= 30:
                label = "Mixed"
            else:
                label = "Unstable"

            m4.metric("Quick read", label)

        # --- Manage selection (clean) ---
        st.markdown("### Manage selection")
        rm_left, rm_right = st.columns([2, 1])
        with rm_left:
            remove_key = st.selectbox("Remove one run", ["(none)"] + keys)
        with rm_right:
            if remove_key != "(none)":
                if st.button("Remove", use_container_width=True):
                    st.session_state.compare_store.pop(remove_key, None)
                    st.rerun()

        # --- Advanced: matrices ---
        with st.expander("Advanced: show full NMI / ARI matrices", expanded=False):
            st.markdown("**NMI matrix** (1 = identical, 0 = unrelated)")
            st.dataframe(
                [
                    {
                        "": keys[i],
                        **{keys[j]: round(nmi_mat[i][j], 3) for j in range(n)},
                    }
                    for i in range(n)
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("**ARI matrix** (1 = identical, ~0 = random)")
            st.dataframe(
                [
                    {
                        "": keys[i],
                        **{keys[j]: round(ari_mat[i][j], 3) for j in range(n)},
                    }
                    for i in range(n)
                ],
                use_container_width=True,
                hide_index=True,
            )
