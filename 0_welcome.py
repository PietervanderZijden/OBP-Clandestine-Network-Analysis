import streamlit as st
from ui_components import apply_tactical_theme
from src.data_manager import get_active_network

st.set_page_config(page_title="Network Analysis DSS", layout="wide")
apply_tactical_theme()

# Try to get active dataset name for the header
try:
    _, metadata = get_active_network()
    dataset_name = metadata.get('name', 'Unknown')
except:
    dataset_name = "Not Selected"

# --- HEADER SECTION ---
col_title, col_info = st.columns([3, 1])

with col_title:
    st.title("Network Analysis Decision Support System")
    st.caption("Structural Analysis & Optimization Framework for Clandestine Networks")

with col_info:
    # A cleaner, more academic status box
    st.markdown(f"""
    <div style="
        border: 1px solid #30363d; 
        padding: 15px; 
        border-radius: 6px; 
        background-color: #0d1117;
        text-align: right;">
        <div style="font-size: 12px; color: #8b949e; margin-bottom: 4px;">Active Dataset</div>
        <div style="font-size: 16px; font-weight: 600; color: #58a6ff;">{dataset_name}</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --- INTRODUCTION ---
st.markdown("### Overview")
st.markdown("""
This platform is a Decision Support System (DSS) designed for the mathematical analysis of social networks. 
It combines graph theory, linear algebra, and combinatorial optimization to provide quantitative insights into network structure, resilience, and partitioning.
""")

# --- MODULES OVERVIEW ---
st.markdown("### System Capabilities")

st.markdown("""
#### Data Management
**Import and switch datasets.** The system allows you to upload custom Matrix Market (.mtx) files and toggle between different network snapshots for comparative analysis.

#### Centrality & Reachability
**Identify key actors.** Analyze node importance using Degree, Eigenvector, and Katz centrality measures. Visualize geodesic paths to understand information flow range.

#### Functional Roles
**Classify network positions.** Group nodes by structural equivalence (e.g., Leaders, Bridges, Peripheral members) rather than just their connectivity count.

#### Resilience Analysis (Kemeny)
**Measure structural robustness.** Evaluate how the network's communication efficiency (Mean First Passage Time) degrades when specific links are removed.

#### Partitioning Strategy
**Optimize network division.** Algorithms to split the network into disjoint groups (e.g., for arrest planning or departmental assignment) while minimizing the information leakage (regret) between groups.
""")

# --- FOOTER GUIDANCE ---
st.markdown("---")
st.info(
    "Getting Started: "
    "Navigate to the **Import Data** page to select or upload a dataset. "
    "Then, use the sidebar to access the analytical modules."
)
