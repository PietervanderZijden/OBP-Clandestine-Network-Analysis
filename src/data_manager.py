import streamlit as st
import networkx as nx
import pandas as pd
import os
import scipy.io

# Define the default file location
DEFAULT_FILE = "data/clandestine_network_example.mtx"

def parse_mtx(file_input, name, origin_type):
    """
    Universal parser: Reads .mtx from a path or file-uploader object
    and returns the standardized data dictionary.
    """
    try:
        # scipy.io.mmread handles both strings (paths) and open file objects
        sparse_matrix = scipy.io.mmread(file_input)
        coo = sparse_matrix.tocoo()
        df_edges = pd.DataFrame({
            'Source': coo.row,
            'Target': coo.col,
            'Weight': coo.data if coo.data.size > 0 else 1 
        })
        
        return {
            "df": df_edges,
            "shape": sparse_matrix.shape,
            "name": name,
            "type": origin_type
        }
    except Exception as e:
        print(f"Error parsing {name}: {e}")
        return None

def load_repository_data(folder="data"):
    """
    Scans the /data folder and automatically adds new files to the session registry.
    """
    # Ensure registry exists
    if 'data_registry' not in st.session_state:
        st.session_state['data_registry'] = {}
        
    if not os.path.exists(folder):
        return

    # Scan and load
    for f in os.listdir(folder):
        if f.endswith('.mtx') and f not in st.session_state['data_registry']:
            full_path = os.path.join(folder, f)
            # Use the reusable parser
            data_pack = parse_mtx(full_path, f, "REPO")
            if data_pack:
                st.session_state['data_registry'][f] = data_pack

def get_active_network():
    """
    Returns the active network graph.
    Priority 1: User selection. Priority 2: Default REPO file.
    """
    # CASE 1: User Selected
    if 'network_data' in st.session_state:
        df_edges = st.session_state['network_data']
        source_name = st.session_state.get('data_source', 'Unknown')
        
        if 'Weight' in df_edges.columns:
            G = nx.from_pandas_edgelist(df_edges, 'Source', 'Target', edge_attr='Weight')
        else:
            G = nx.from_pandas_edgelist(df_edges, 'Source', 'Target')
            
        return G, {"name": source_name, "type": "USER_SELECTION"}

    # CASE 2: Default Fallback
    if os.path.exists(DEFAULT_FILE):
        try:
            data_pack = parse_mtx(DEFAULT_FILE, "Default Protocol", "REPO")
            G = nx.from_pandas_edgelist(data_pack['df'], 'Source', 'Target')
            return G, {"name": "Default Protocol", "type": "REPOSITORY_DEFAULT"}
        except:
            st.error("Critical Error: Default file corrupted.")
            st.stop()
    
    st.error("SYSTEM HALT: No data found.")
    st.stop()