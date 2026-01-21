import streamlit as st
import networkx as nx
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

# Import shared theme
from ui_components import apply_tactical_theme, COLOR_VOID, COLOR_WIRE, COLOR_STEEL, COLOR_ALERT

# Import the data manager
from src.data_manager import get_active_network

# --- PAGE CONFIG ---
st.set_page_config(page_title="DSS | Intelligence & Centrality", layout="wide")
apply_tactical_theme()

# --- 1. DATA LOADING (Fixed Caching) ---
# A. Get the Active Graph (Fast, no cache needed)
G, metadata = get_active_network()

# B. Calculate Layout (Cached)
# FIX: Added underscore to '_graph' so Streamlit ignores it during hashing
@st.cache_data
def calculate_layout(_graph, source_identifier):
    """
    Calculates layout. Refreshes only when source_identifier changes.
    The underscore in '_graph' tells Streamlit not to hash the graph object.
    """
    pos = nx.spring_layout(_graph, seed=42, k=1.2)
    fixed_positions = {
        node: (coords[0] * 1000, coords[1] * 1000) 
        for node, coords in pos.items()
    }
    return fixed_positions

# Execute Layout
layout_map = calculate_layout(G, metadata['name'])


# --- 2. CENTRALITY CALCULATIONS ---
# FIX: Ensure '_G' has the underscore here too
@st.cache_data
def calculate_all_centralities(_G, source_identifier):
    if _G.number_of_nodes() == 0: return {}
    
    degree = nx.degree_centrality(_G)
    try:
        eigen = nx.eigenvector_centrality(_G, max_iter=1000)
    except:
        eigen = {n:0 for n in _G.nodes()} 
        
    katz = nx.katz_centrality(_G, alpha=0.1, beta=1.0)
    
    combined = {node: (degree[node] + eigen[node] + katz[node]) / 3 for node in _G.nodes()}
    
    return {
        "Degree": degree,
        "Eigenvector": eigen,
        "Katz": katz,
        "Hybrid (Combined)": combined
    }

centrality_results = calculate_all_centralities(G, metadata['name'])

# --- 3. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### CONTROL_MODE")
    analysis_mode = st.radio("SELECT_MODE", ["Tactical Flow (Click)", "Importance Analysis (Top N)"])
    
    st.divider()
    
    # Active Target Display
    st.markdown(f"""
    <div style="font-family:'Share Tech Mono'; font-size:12px; color:#8b949e; margin-bottom:10px;">
        TARGET: <span style="color:#58a6ff">{metadata['name']}</span>
    </div>
    """, unsafe_allow_html=True)

    if not G.nodes():
        st.error("NO DATA")
        st.stop()
        
    first_node = list(G.nodes())[0]

    if analysis_mode == "Tactical Flow (Click)":
        st.markdown("### FLOW_SETTINGS")
        if "target" not in st.session_state: st.session_state.target = first_node
        
        # Validate target exists in new dataset
        if st.session_state.target not in G.nodes():
             st.session_state.target = first_node

        st.write(f"FOCUS: **NODE {st.session_state.target}**")
        max_hops = st.slider("ANALYSIS RADIUS (HOPS)", 1, 8, 2)
        target_node = st.session_state.target
    else:
        st.markdown("### CENTRALITY_SETTINGS")
        measure = st.selectbox("CENTRALITY_MEASURE", list(centrality_results.keys()))
        n_top = st.slider("SELECT TOP N MEMBERS", 1, len(G.nodes()), 10)
        selected_scores = centrality_results[measure]
        top_n_members = sorted(selected_scores, key=selected_scores.get, reverse=True)[:n_top]
        target_node = None

    if st.button("RESET SYSTEM"):
        st.session_state.target = first_node
        st.rerun()

# --- 4. VISUALIZATION LOGIC ---
nodes = []
edges = []

if analysis_mode == "Tactical Flow (Click)":
    distances = nx.single_source_shortest_path_length(G, target_node, cutoff=max_hops)
    
    def get_color(node):
        dist = distances.get(node)
        if dist == 0: return "#FFFFFF" 
        heatmap = {1: "#FF0000", 2: "#FF8C00", 3: "#FFD700", 4: "#00FF00"}
        return heatmap.get(dist, "#00BFFF") if dist is not None else COLOR_STEEL

    def get_label(node):
        dist = distances.get(node)
        return f"{node} [L{dist}]" if dist is not None else str(node)

else:
    def get_color(node):
        return COLOR_ALERT if node in top_n_members else COLOR_STEEL
    
    def get_label(node):
        if node in top_n_members:
            rank = top_n_members.index(node) + 1
            return f"#{rank} (ID:{node})"
        return str(node)

# --- 5. BUILD GRAPH ELEMENTS ---
for node in G.nodes():
    x_pos, y_pos = layout_map.get(node, (0,0))
    nodes.append(Node(
        id=str(node), label=get_label(node), size=25, 
        color=get_color(node), x=x_pos, y=y_pos,
        font={'color': 'white', 'face': 'monospace', 'size': 14}
    ))

for u, v in G.edges():
    active_edge = False
    if analysis_mode == "Tactical Flow (Click)":
        active_edge = u in distances and v in distances
        color = get_color(v if distances.get(u, 9) < distances.get(v, 9) else u) if active_edge else COLOR_WIRE
    else:
        active_edge = u in top_n_members and v in top_n_members
        color = COLOR_ALERT if active_edge else COLOR_WIRE

    edges.append(Edge(
        source=str(u), target=str(v), color=color, 
        width=3 if active_edge else 1,
        opacity=1.0 if active_edge else 0.1,
        type="STRAIGHT" # Clean lines
    ))

# --- 6. RENDER ---
config = Config(width=1100, height=700, directed=False, physics=False, staticGraph=True)

st.markdown(f"### SYSTEM_VIEW: {analysis_mode.upper()}")
return_value = agraph(nodes=nodes, edges=edges, config=config)

if analysis_mode == "Tactical Flow (Click)" and return_value is not None:
    try:
        clicked_id = int(return_value)
        if clicked_id != st.session_state.target:
            st.session_state.target = clicked_id
            st.rerun()
    except:
        pass

# --- 7. DATA TABLE ---
st.divider()

if analysis_mode == "Importance Analysis (Top N)":
    st.markdown(f"### RANKING: {measure.upper()}")
    df_rank = pd.DataFrame([
        {"RANK": i+1, "MEMBER_ID": m, "SCORE": round(selected_scores[m], 4)} 
        for i, m in enumerate(top_n_members)
    ])
    
    st.dataframe(
        df_rank, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            "RANK": st.column_config.NumberColumn("RANK", format="#%d"),
            "SCORE": st.column_config.ProgressColumn("SCORE", min_value=0, max_value=max(selected_scores.values()))
        }
    )

else:
    st.markdown(f"### LAYER_RECON: SOURCE {target_node}")
    df_flow = pd.DataFrame([{"NODE": n, "LAYER": d} for n, d in distances.items() if n != target_node])
    st.dataframe(
        df_flow.sort_values("LAYER"), 
        use_container_width=True, 
        hide_index=True
    )