import streamlit as st
import pandas as pd
import scipy.io
from ui_components import apply_tactical_theme

# --- Page Config ---
st.set_page_config(page_title="DSS | Data Ingest", layout="wide")
apply_tactical_theme()

# --- Header Section ---
col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("DATA INGEST TERMINAL")
    st.caption("EXTERNAL INTELLIGENCE IMPORT & VALIDATION LAYER")

with col_status:
    # Custom HTML to match your Tactical Theme
    st.markdown("""
        <div style="border: 1px solid #30363D; padding: 15px; background-color: rgba(9, 10, 11, 0.8);">
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#58a6ff;">MODULE: INGEST</p>
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#8b949e;">PROTOCOL: MTX_STRICT</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()

# --- Main Interface ---
col_upload, col_info = st.columns([1, 1])

# LEFT COLUMN: UPLOADER
with col_upload:
    with st.container(border=True):
        st.subheader("Target Upload")
        st.markdown(
            "<span style='color:#8B949E; font-size:14px;'>System accepts verified Matrix Market (.mtx) coordinate formats only.</span>", 
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader("Select Intel File", type=['mtx'], label_visibility="collapsed")

        if uploaded_file is not None:
            try:
                # 1. VALIDATION via Scipy
                # This explicitly checks structure, headers, and symmetry
                sparse_matrix = scipy.io.mmread(uploaded_file)
                
                # 2. PARSING
                coo = sparse_matrix.tocoo()
                df_edges = pd.DataFrame({
                    'Source': coo.row,
                    'Target': coo.col,
                    # If weights exist use them, else default to 1
                    'Weight': coo.data if coo.data.size > 0 else 1 
                })

                # 3. GLOBAL STATE STORAGE
                st.session_state['network_data'] = df_edges
                st.session_state['network_shape'] = sparse_matrix.shape
                st.session_state['data_source'] = uploaded_file.name

                # Success UI
                st.success("INTEGRATION SUCCESSFUL: Structure Validated")
                
                # Metrics Row
                m1, m2 = st.columns(2)
                m1.metric("Nodes Identified", sparse_matrix.shape[0])
                m2.metric("Connections Mapped", len(df_edges))

            except Exception as e:
                st.error("CRITICAL ERROR: CORRUPT DATA STRUCTURE")
                st.warning(f"Trace: {e}")

# RIGHT COLUMN: SPECS & HELP
with col_info:
    with st.expander("ℹ️ DATA STRUCTURE SPECIFICATION", expanded=True):
        st.markdown("""
        **REQUIRED FORMAT: Matrix Market (.mtx)**
        
        File must adhere to the `coordinate pattern symmetric` standard.
        
        **Header Signature:**
        `%%MatrixMarket matrix coordinate pattern symmetric`
        
        **Structure:**
        1. **Headers:** Standard MatrixMarket banners.
        2. **Dimensions:** Line defining `Rows Cols Entries`.
        3. **Payload:** Space-separated node pairs.
        
        **Example Payload:**
        ```text
        %%MatrixMarket matrix coordinate pattern symmetric
        % comments allowed
        62 62 159
        11 1
        15 1
        ```
        """)
        
    # Sample Download
    sample_mtx = """%%MatrixMarket matrix coordinate pattern symmetric
% Sample Clandestine Network
5 5 4
1 2
2 3
3 4
4 1
"""
    st.download_button(
        label="⬇️ DOWNLOAD REFERENCE TEMPLATE",
        data=sample_mtx,
        file_name="reference_network.mtx",
        mime="text/plain",
        help="Download a clean .mtx template for data formatting."
    )