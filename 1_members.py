import streamlit as st
import networkx as nx
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

# Import shared theme
from ui_components import apply_tactical_theme, COLOR_VOID, COLOR_WIRE, COLOR_STEEL, COLOR_ALERT

# Import the data manager
from src.data_manager import get_active_network

# --- PAGE CONFIG ---
st.set_page_config(page_title="Network Analysis | Centrality", layout="wide")
apply_tactical_theme()

# --- HEADER & EXPLANATION ---
st.title("Network Centrality & Reachability")
st.caption("""
    Exploratory analysis of network structure. 
    **Reachability Analysis** examines the geodesic distance from a specific node to the rest of the network.
    **Centrality Ranking** evaluates node importance using standard graph theory metrics.
""")

with st.expander("📘 Quick guide", expanded=True):
    st.markdown(
        """
        **Objective:** Classify network members into structural roles (Core, Intermediate, Peripheral) using comparative graph-theoretic algorithms.

        **Workflow:**
        1. **Select Method:** Choose an algorithm from the sidebar (e.g., *Influence Flow* or *Centrality*) to define how roles are calculated.
        2. **Analyze Map:** View the **Role Map** to identify the structural hierarchy and distribution of roles.
        3. **Inspect Members:** Select a node to view the **Confidence Score** (consensus across different algorithms) and detailed evidence for their assigned role.
        """
    )

# --- 1. DATA LOADING ---
G, metadata = get_active_network()

@st.cache_data
def calculate_layout(_graph, source_identifier):
    pos = nx.spring_layout(_graph, seed=42, k=1.2)
    fixed_positions = {
        node: (coords[0] * 1000, coords[1] * 1000) 
        for node, coords in pos.items()
    }
    return fixed_positions

layout_map = calculate_layout(G, metadata['name'])

# --- 2. CENTRALITY CALCULATIONS ---
@st.cache_data
def calculate_all_centralities(_G, source_identifier):
    if _G.number_of_nodes() == 0: return {}
    degree = nx.degree_centrality(_G)
    try:
        eigen = nx.eigenvector_centrality(_G, max_iter=1000)
    except:
        # Fallback for unconnected graphs
        eigen = {n:0 for n in _G.nodes()} 
    katz = nx.katz_centrality(_G, alpha=0.1, beta=1.0)
    combined = {node: (degree[node] + eigen[node] + katz[node]) / 3 for node in _G.nodes()}
    
    return {
        "Degree Centrality": degree,
        "Eigenvector Centrality": eigen,
        "Katz Centrality": katz,
        "Composite Score (Avg)": combined
    }

centrality_results = calculate_all_centralities(G, metadata['name'])

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### Analysis Configuration")
    
    analysis_mode = st.radio(
        "Select Method", 
        ["Reachability (Distance from Node)", "Centrality Ranking (Top N)"],
        help="Choose between analyzing a single node's neighborhood or ranking all nodes by importance metrics."
    )
    
    st.divider()
    
    # Active Dataset Display
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono'; font-size:12px; color:#8b949e; margin-bottom:10px;">
        dataset: <span style="color:#58a6ff">{metadata['name']}</span>
    </div>
    """, unsafe_allow_html=True)

    if not G.nodes():
        st.error("Dataset is empty.")
        st.stop()
        
    first_node = list(G.nodes())[0]

    if analysis_mode == "Reachability (Distance from Node)":
        st.markdown("### Parameters")
        if "target" not in st.session_state: st.session_state.target = first_node
        if st.session_state.target not in G.nodes():
             st.session_state.target = first_node

        st.write(f"Selected Node ID: **{st.session_state.target}**")
        
        # --- DYNAMIC SLIDER LOGIC ---
        try:
            all_paths = nx.single_source_shortest_path_length(G, st.session_state.target)
            max_network_depth = max(all_paths.values()) if all_paths else 1
        except:
            max_network_depth = 8
            
        slider_limit = min(max_network_depth, 8)
        if slider_limit < 1: slider_limit = 1
        
        max_hops = st.slider(
            "Max Geodesic Distance", 
            1, slider_limit, min(2, slider_limit),
            help="Maximum number of edges (hops) from the source node to visualize."
        )
        # ---------------------------

        target_node = st.session_state.target
    else:
        st.markdown("### Parameters")
        measure = st.selectbox(
            "Metric", 
            list(centrality_results.keys()),
            help="Select the centrality algorithm to apply."
        )
        n_top = st.slider("Number of Top Nodes", 1, len(G.nodes()), 10)
        selected_scores = centrality_results[measure]
        top_n_members = sorted(selected_scores, key=selected_scores.get, reverse=True)[:n_top]
        target_node = None

    if st.button("Reset Selection"):
        st.session_state.target = first_node
        st.rerun()

# --- 4. VISUALIZATION LOGIC ---
nodes = []
edges = []

if analysis_mode == "Reachability (Distance from Node)":
    distances = nx.single_source_shortest_path_length(G, target_node, cutoff=max_hops)
    
    def get_color(node):
        dist = distances.get(node)
        if dist == 0: return "#FFFFFF" # Source is white
        # Standard heatmap for distance
        heatmap = {1: "#FF0000", 2: "#FF8C00", 3: "#FFD700", 4: "#00FF00"}
        return heatmap.get(dist, "#00BFFF") if dist is not None else COLOR_STEEL

    def get_label(node):
        dist = distances.get(node)
        return f"{node} (d={dist})" if dist is not None else str(node)

else:
    def get_color(node):
        return COLOR_ALERT if node in top_n_members else COLOR_STEEL
    
    def get_label(node):
        if node in top_n_members:
            rank = top_n_members.index(node) + 1
            return f"#{rank} {node}"
        return str(node)

# --- 5. BUILD GRAPH ELEMENTS ---
for node in G.nodes():
    x_pos, y_pos = layout_map.get(node, (0,0))
    nodes.append(Node(
        id=str(node), 
        label=get_label(node), 
        size=25, 
        color=get_color(node), 
        x=x_pos, 
        y=y_pos,
        # Standard font style
        font={
            'color': 'white',  
            'background': '#090A0B', 
            'face': 'arial', 
            'size': 14,
            'strokeWidth': 0, 
            'align': 'center'
        }
    ))

# --- EDGE GENERATION ---
for u, v in G.edges():
    # Default settings (used for Centrality Ranking)
    color = COLOR_WIRE
    width = 1
    opacity = 0.1

    # Overrides for Reachability Mode
    if analysis_mode == "Reachability (Distance from Node)":
        active_edge = u in distances and v in distances
        if active_edge:
            # Color by the 'further' node's distance color logic
            color = get_color(v if distances.get(u, 9) < distances.get(v, 9) else u)
            width = 3
            opacity = 1.0

    edges.append(Edge(
        source=str(u), target=str(v), color=color, 
        width=width,
        opacity=opacity,
        type="STRAIGHT"
    ))

# --- 6. RENDER ---

# Height adjustment to keep visualizer active on updates
dynamic_height = 700 if st.session_state.target % 2 == 0 else 701

config = Config(
    width="100%", 
    height=dynamic_height, 
    directed=False, 
    physics=False, 
    staticGraph=True
)

st.subheader(f"Visualization: {analysis_mode}")

# Draw Graph
return_value = agraph(nodes=nodes, edges=edges, config=config)

# "Ghost Click" Logic (ensures graph updates correctly on first click)
if "needs_second_kick" not in st.session_state:
    st.session_state.needs_second_kick = False

if st.session_state.needs_second_kick:
    st.session_state.needs_second_kick = False
    st.rerun()

# Click Handler for Reachability Mode
if analysis_mode == "Reachability (Distance from Node)" and return_value is not None:
    try:
        clicked_id = int(return_value)
        # If the click is NEW
        if clicked_id != st.session_state.target:
            st.session_state.target = clicked_id
            st.session_state.needs_second_kick = True
            st.rerun()
    except ValueError:
        pass

# --- 7. DATA TABLE ---
st.divider()

if analysis_mode == "Centrality Ranking (Top N)":
    st.subheader(f"Ranking Table: {measure}")
    df_rank = pd.DataFrame([
        {"Rank": i+1, "Node ID": m, "Score": round(selected_scores[m], 4)} 
        for i, m in enumerate(top_n_members)
    ])
    
    st.dataframe(
        df_rank, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="#%d"),
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=max(selected_scores.values()))
        }
    )

else:
    st.subheader(f"Distance Distribution: Node {target_node}")
    df_flow = pd.DataFrame([{"Node ID": n, "Distance": d} for n, d in distances.items() if n != target_node])
    st.dataframe(
        df_flow.sort_values("Distance"), 
        use_container_width=True, 
        hide_index=True
    )