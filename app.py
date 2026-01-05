import streamlit as st
import networkx as nx
from scipy.io import mmread
import os
from streamlit_option_menu import option_menu

# Page settings
st.set_page_config(page_title="Clandestine Network DSS", layout="wide")

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        "Main Menu", 
        ["Situation Room", "Centrality", "Factions", "Connectivity", "Arrest Plan"],
        icons=['eye', 'person-badge', 'people', 'diagram-3', 'shield-lock'], 
        menu_icon="cast", default_index=0
    )

# Load Data function
@st.cache_data
def load_data():
    # Make sure your file is named 'network.mtx' in the 'data' folder
    path = "data/network.mtx"
    if os.path.exists(path):
        matrix = mmread(path)
        return nx.Graph(matrix)
    return None

G = load_data()

if selected == "Situation Room":
    st.title("Clandestine Network Overview")
    if G:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Members", G.number_of_nodes())
        col2.metric("Verified Links", G.number_of_edges())
        col3.metric("Network Density", f"{nx.density(G):.4f}")
        
        st.write("### Network Preview")
        st.info("The clandestine network consists of 62 members organized in a highly resilient structure.")
    else:
        st.error("Data file not found. Please ensure 'network.mtx' is in the '/data' folder.")

# Placeholders for other members' work
elif selected == "Centrality":
    st.title("Member Importance")
    st.write("This section will be implemented by Team Member 2.")

# ... repeat for other sections