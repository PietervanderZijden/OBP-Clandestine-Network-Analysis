# 🕵️ DSS // CLANDESTINE NETWORK ANALYSIS
**Tactical Decision Support System** for the social network analysis of a 62-person clandestine organization.

## 🚀 GETTING STARTED

This project has been migrated from Conda to **Rye** to ensure faster builds and better stability for native network libraries (igraph, leidenalg, infomap).

### 1. Installation & Setup

To ensure your environment matches the production server exactly, we recommend using Rye:

- **Install Rye (macOS/Linux):** 

    curl -sSf https://rye.astral.sh/get | bash

- **Install Rye (Windows):** 
    Download the installer from [Rye Official Site](https://rye.astral.sh).

Once installed, run the following in the project folder:

rye sync


This automatically downloads Python 3.12, creates a virtual environment, and installs all required dependencies.

### 2. Running the Application

To launch the dashboard locally:

rye run streamlit run app.py


#### 🛠 ALTERNATIVE SETUP (PIP ONLY)

If you prefer to use a standard Python environment instead of Rye:

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py


#### ☁️ CLOUD DEPLOYMENT (STREAMLIT CLOUD)

This repository is configured to deploy via `requirements.txt`.

**Important for Developers:** If you add new libraries locally using Rye, you must update the requirements file before pushing to GitHub to avoid "ModuleNotFound" errors in the cloud:

rye pip freeze > requirements.txt


### 🔍 SYSTEM ARCHITECTURE & FEATURES

- **Community Detection:** Multi-algorithmic suite including Louvain, Leiden, Infomap, Spectral Clustering, and Girvan-Newman.
- **Role Identification:** Proprietary scoring for member embeddedness and network flow analysis.
- **Security Layer:** Military-grade Argon2id hashing for secure access control.
- **Visualization:** Custom tactical UI with interactive graph renderings via streamlit-agraph and Plotly.
- **Robustness Testing:** Built-in disturbance simulation (5% edge removal) to verify faction stability.

