# OBP Clandestine Network Analysis

Decision Support System for Social Network Analysis of a 62-person clandestine organization.

## Installation (recommended)

This project uses a Conda environment to ensure reproducibility across systems
(native libraries such as igraph, leidenalg, infomap are required).

### Setup
```bash
conda env create -f environment.yml
conda activate obp-dss
streamlit run app.py
