import streamlit as st
import pandas as pd
import numpy as np

# Gebruik kolommen voor een mooie lay-out
col1, col2 = st.columns([6, 2])

with col1:
    st.title("Decision Support System")
    st.subheader("Clandestine Network Analysis & Optimization")

with col2:
    # Een visuele status indicator
    st.info("Status: **Active**\n\nUser: **Admin**")

st.divider()

st.markdown("""
### Welcome, Analyst
This system is designed to provide tactical insights into the clandestine network of 62 members. 
By utilizing state-of-the-art **Network Science** and **Mathematical Optimization**, this DSS (Decision Support System) 
helps identify key figures and vulnerabilities within the organization.

#### 📊 Current Objectives:
1. **Identify Key Players**: Use Centrality measures (Degree, Betweenness, Katz) to find the most influential members.
2. **Community Detection**: Map out distinct cells or "factions" within the network.
3. **Connectivity Analysis**: Measure the network's resilience using the **Kemeny Constant**.
4. **Disruption Planning**: Optimize the arrest strategy to maximize network disruption within police resource constraints.
""")
