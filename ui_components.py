import streamlit as st

# ==========================================
# 1. PYTHON COLOR PALETTES (Restored)
# ==========================================
# These are needed for Matplotlib/NetworkX charts that run in Python
# and cannot read CSS variables.

# --- DARK MODE (Tactical / Default) ---
COLOR_VOID  = "#090A0B"
COLOR_GRID  = "#1B1F23"
COLOR_WIRE  = "#30363D"
COLOR_STEEL = "#444C56"
COLOR_TEXT  = "#8B949E"
COLOR_HEADER = "#FFFFFF"
COLOR_ACCENT = "#58a6ff"
COLOR_ALERT = "#F85149"  # Red

# --- LIGHT MODE (Operator / Print) ---
# You can use these if you ever want to force a light-mode chart
LIGHT_VOID  = "#FFFFFF"
LIGHT_GRID  = "#E1E4E8"
LIGHT_WIRE  = "#D0D7DE"
LIGHT_STEEL = "#6E7781"
LIGHT_TEXT  = "#57606A"
LIGHT_HEADER = "#24292F"
LIGHT_ACCENT = "#0969da"
LIGHT_ALERT = "#cf222e"

# ==========================================
# 2. THEME INJECTOR
# ==========================================
def apply_tactical_theme():
    """
    Injects CSS that supports both Dark and Light modes based on
    the user's system preferences.
    """
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

        /* CSS VARIABLES: Define colors for both modes */
        :root {{
            /* DEFAULT: DARK MODE */
            --bg-void: {COLOR_VOID};
            --bg-grid: {COLOR_GRID};
            --color-wire: {COLOR_WIRE};
            --color-text: {COLOR_TEXT};
            --color-header: {COLOR_HEADER};
            --color-accent: {COLOR_ACCENT};
            --box-bg: rgba(9, 10, 11, 0.8);
        }}

        /* BROWSER CHECK: If user prefers Light Mode */
        @media (prefers-color-scheme: light) {{
            :root {{
                /* LIGHT MODE OVERRIDES */
                --bg-void: {LIGHT_VOID};
                --bg-grid: {LIGHT_GRID};
                --color-wire: {LIGHT_WIRE};
                --color-text: {LIGHT_TEXT};
                --color-header: {LIGHT_HEADER};
                --color-accent: {LIGHT_ACCENT};
                --box-bg: rgba(255, 255, 255, 0.9);
            }}
        }}

        /* APP STYLING USING VARIABLES */
        .stApp {{
            background-color: var(--bg-void);
            background-image: 
                linear-gradient(var(--bg-grid) 1px, transparent 1px),
                linear-gradient(90deg, var(--bg-grid) 1px, transparent 1px);
            background-size: 35px 35px;
            color: var(--color-text);
        }}
        
        [data-testid="stSidebar"] {{
            background-color: var(--bg-void) !important;
            border-right: 1px solid var(--color-wire);
        }}

        [data-testid="stSidebarNav"] span {{
            color: var(--color-text) !important;
            font-family: 'Share Tech Mono', monospace !important;
            text-transform: uppercase !important;
            font-size: 14px !important;
            letter-spacing: 1px !important;
        }}

        h1, h2, h3, h4, h5, h6 {{
            color: var(--color-header) !important;
            font-family: 'Share Tech Mono', monospace;
        }}
        
        .stCaption, p, li, label, .stMarkdown {{
            color: var(--color-text) !important;
        }}

        [data-testid="stExpander"], [data-testid="stForm"] {{
            border-color: var(--color-wire) !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: var(--color-header) !important;
        }}

        iframe {{ border: none !important; }}
        
        /* Fix for Plotly charts background */
        .js-plotly-plot .plotly .main-svg {{
            background: rgba(0,0,0,0) !important;
        }}
        </style>
        """, unsafe_allow_html=True)