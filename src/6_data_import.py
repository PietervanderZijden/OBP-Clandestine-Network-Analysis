import streamlit as st
from ui_components import apply_tactical_theme
# Import the logic functions we just wrote
from src.data_manager import load_repository_data, parse_mtx
import networkx as nx


def df_to_graph(edge_df, n_nodes=None):
    """
    Builds an undirected NetworkX graph from an edge list DataFrame.
    Expects either:
      - columns: ['source','target'] (+ optional 'weight')
      - or the first 2 columns are the endpoints (+ optional 'weight')
    Assumes node IDs are already ints and consistent (0-based OR 1-based, but consistent).
    """
    # pick endpoint columns
    if "source" in edge_df.columns and "target" in edge_df.columns:
        s_col, t_col = "source", "target"
    else:
        s_col, t_col = edge_df.columns[0], edge_df.columns[1]

    weighted = "weight" in edge_df.columns

    G = nx.Graph()

    # optionally pre-add nodes (helps if isolated nodes exist)
    if n_nodes is not None:
        # If your nodes are 0..n-1, keep this.
        # If your nodes are 1..n, change to range(1, n_nodes+1)
        G.add_nodes_from(range(n_nodes))

    if weighted:
        for s, t, w in edge_df[[s_col, t_col, "weight"]].itertuples(index=False):
            if int(s) != int(t):
                G.add_edge(int(s), int(t), weight=float(w))
    else:
        for s, t in edge_df[[s_col, t_col]].itertuples(index=False):
            if int(s) != int(t):
                G.add_edge(int(s), int(t), weight=1.0)

    return G

# --- Page Config ---
st.set_page_config(page_title="DSS | Data Ingest", layout="wide")
apply_tactical_theme()

# --- 1. RUN LOGIC ---
# This one line handles scanning the folder and updating the registry
load_repository_data("data")

# --- Header Section ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("DATA IMPORTING TOOL")
    
with col_status:
    count = len(st.session_state['data_registry'])
    st.markdown(f"""
<div style="border: 1px solid var(--color-wire); padding: 15px; background-color: var(--box-bg);">
    <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color: var(--color-accent);">
        SYSTEM: ONLINE
    </p>
    <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color: var(--color-text);">
        DATASETS LOADED: {count}
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Main Interface ---
col_upload, col_select = st.columns([1, 1])

# ==========================================
# LEFT: UPLOADER (Cleaned up)
# ==========================================
with col_upload:
    with st.container(border=True):
        st.subheader("Data Upload")
        st.caption("System accepts verified Matrix Market (.mtx) formats only.")
        
        uploaded_files = st.file_uploader(
            "Select Intel Files", 
            type=['mtx'], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            new_count = 0
            for file in uploaded_files:
                # Use logic from network_manager
                if file.name not in st.session_state['data_registry']:
                    data_pack = parse_mtx(file, file.name, "UPLOAD")
                    
                    if data_pack:
                        st.session_state['data_registry'][file.name] = data_pack
                        new_count += 1
                    else:
                        st.error(f"Failed to parse {file.name}")
            
            if new_count > 0:
                st.rerun()

# ==========================================
# RIGHT: SELECTOR (Cleaned up)
# ==========================================
with col_select:
    with st.container(border=True):
        st.subheader("Active Operation Data")
        
        if not st.session_state['data_registry']:
            st.warning("NO DATA AVAILABLE")
        else:
            options = list(st.session_state['data_registry'].keys())
            
            # Smart Indexing
            index = 0
            if 'data_source' in st.session_state and st.session_state['data_source'] in options:
                index = options.index(st.session_state['data_source'])

            selected_name = st.selectbox("Select Dataset", options, index=index)

            if selected_name:
                # Activate the selection
                data_pack = st.session_state['data_registry'][selected_name]
                st.session_state['network_data'] = data_pack['df']
                st.session_state['network_shape'] = data_pack['shape']
                st.session_state['data_source'] = data_pack['name']
                n_nodes = data_pack["shape"][0]
                st.session_state["network_graph"] = df_to_graph(data_pack["df"], n_nodes=n_nodes)

                st.divider()
                
                # Source Badge
                src_type = data_pack.get('type', 'UNKNOWN')
                st.markdown(f"<p style='font-family: \"Share Tech Mono\", monospace; color:#8b949e; font-size:12px;'>data origin: <span style='color:var(--color-accent);'>[{src_type}]</span></p>", unsafe_allow_html=True)
                st.success(f"DATA ACTIVE: {selected_name}")
                
                # Stats
                m1, m2 = st.columns(2)
                m1.metric("Nodes", data_pack['shape'][0])
                m2.metric("Connections", len(data_pack['df']))
                
                if st.button("Purge Registry", type="primary", use_container_width=True):
                    st.session_state['data_registry'] = {}
                    if 'network_data' in st.session_state:
                        del st.session_state['network_data']
                    st.rerun()
    
    # Help Expander
    with st.expander("DATA STRUCTURE SPECIFICATION"):
        st.markdown("**REQUIRED FORMAT: Matrix Market (.mtx)**")
        st.download_button("DOWNLOAD TEMPLATE", data="%%MatrixMarket matrix coordinate pattern symmetric\n5 5 4\n1 2\n", file_name="template.mtx")