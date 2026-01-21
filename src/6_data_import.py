import streamlit as st
import pandas as pd
import scipy.io
import os
from ui_components import apply_tactical_theme

# --- Page Config ---
st.set_page_config(page_title="DSS | Data Ingest", layout="wide")
apply_tactical_theme()

# --- CONSTANTS ---
DATA_FOLDER = "data"

# --- 1. INITIALIZE & AUTO-LOAD ---
if 'data_registry' not in st.session_state:
    st.session_state['data_registry'] = {}

def load_repo_datasets():
    """Scans the 'data/' folder and loads .mtx files automatically."""
    if not os.path.exists(DATA_FOLDER):
        return

    # Get all .mtx files in the folder
    repo_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.mtx')]
    
    for filename in repo_files:
        if filename not in st.session_state['data_registry']:
            file_path = os.path.join(DATA_FOLDER, filename)
            try:
                # Parse directly from disk
                sparse_matrix = scipy.io.mmread(file_path)
                coo = sparse_matrix.tocoo()
                df_edges = pd.DataFrame({
                    'Source': coo.row,
                    'Target': coo.col,
                    'Weight': coo.data if coo.data.size > 0 else 1 
                })
                
                # Add to registry with REPO tag
                st.session_state['data_registry'][filename] = {
                    "df": df_edges,
                    "shape": sparse_matrix.shape,
                    "name": filename,
                    "type": "REPO"
                }
            except Exception as e:
                print(f"Error auto-loading {filename}: {e}")

# Run the auto-loader
load_repo_datasets()

# --- Header Section ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("DATA IMPORTING TOOL")
    st.caption("MULTI-SOURCE INTELLIGENCE IMPORT & VALIDATION LAYER")

with col_status:
    count = len(st.session_state['data_registry'])
    st.markdown(f"""
        <div style="border: 1px solid #30363D; padding: 15px; background-color: rgba(9, 10, 11, 0.8);">
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#58a6ff;">SYSTEM: ONLINE</p>
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#8b949e;">DATASETS_LOADED: {count}</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Main Interface ---
col_upload, col_select = st.columns([1, 1])

# ==========================================
# LEFT COLUMN: BATCH UPLOADER
# ==========================================
with col_upload:
    with st.container(border=True):
        st.subheader("Target Upload")
        st.markdown(
            "<span style='color:#8B949E; font-size:14px;'>System accepts verified Matrix Market (.mtx) coordinate formats only.</span>", 
            unsafe_allow_html=True
        )
        
        uploaded_files = st.file_uploader(
            "Select Intel Files", 
            type=['mtx'], 
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if uploaded_files:
            new_files_count = 0
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state['data_registry']:
                    try:
                        sparse_matrix = scipy.io.mmread(uploaded_file)
                        coo = sparse_matrix.tocoo()
                        df_edges = pd.DataFrame({
                            'Source': coo.row,
                            'Target': coo.col,
                            'Weight': coo.data if coo.data.size > 0 else 1 
                        })

                        st.session_state['data_registry'][uploaded_file.name] = {
                            "df": df_edges,
                            "shape": sparse_matrix.shape,
                            "name": uploaded_file.name,
                            "type": "UPLOAD"
                        }
                        new_files_count += 1

                    except Exception as e:
                        st.error(f"ERROR LOADING {uploaded_file.name}")
                        st.warning(f"Trace: {e}")
            
            if new_files_count > 0:
                st.rerun()

# ==========================================
# RIGHT COLUMN: SELECTOR & ACTIVE STATUS
# ==========================================
with col_select:
    with st.container(border=True):
        st.subheader("Active Operation Target")
        
        if not st.session_state['data_registry']:
            st.warning("NO DATA AVAILABLE")
            st.caption("Add .mtx files to /data folder or upload manually.")
        else:
            options = list(st.session_state['data_registry'].keys())
            
            index = 0
            if 'data_source' in st.session_state and st.session_state['data_source'] in options:
                index = options.index(st.session_state['data_source'])

            selected_name = st.selectbox(
                "Select Dataset for Analysis", 
                options,
                index=index
            )

            if selected_name:
                data_pack = st.session_state['data_registry'][selected_name]
                
                st.session_state['network_data'] = data_pack['df']
                st.session_state['network_shape'] = data_pack['shape']
                st.session_state['data_source'] = data_pack['name']

                st.divider()
                
                # Visual Indicator of Source (Repo vs Upload)
                src_type = data_pack.get('type', 'UNKNOWN')
                st.markdown(f"<p style='font-family: \"Share Tech Mono\", monospace; color:#8b949e; font-size:12px;'>SOURCE_ORIGIN: <span style='color:#58a6ff;'>[{src_type}]</span></p>", unsafe_allow_html=True)
                st.success(f"TARGET ACTIVE: {selected_name}")
                
                m1, m2 = st.columns(2)
                m1.metric("Nodes", data_pack['shape'][0])
                m2.metric("Connections", len(data_pack['df']))
                
                if st.button("Purge Registry", type="primary", use_container_width=True):
                    st.session_state['data_registry'] = {}
                    if 'network_data' in st.session_state:
                        del st.session_state['network_data']
                    st.rerun()

    with st.expander("DATA STRUCTURE SPECIFICATION"):
        st.markdown("""
        **REQUIRED FORMAT: Matrix Market (.mtx)**
        `%%MatrixMarket matrix coordinate pattern symmetric`
        
        **Structure:**
        1. Headers
        2. Dimensions (`Rows Cols Entries`)
        3. Payload (`Source Target`)
        """)
        
        sample_mtx = """%%MatrixMarket matrix coordinate pattern symmetric
% Sample Clandestine Network
5 5 4
1 2
2 3
3 4
4 1
"""
        st.download_button(
            label="DOWNLOAD TEMPLATE",
            data=sample_mtx,
            file_name="reference_network.mtx",
            mime="text/plain"
        )