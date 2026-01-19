import streamlit as st
from ui_components import apply_tactical_theme

st.set_page_config(page_title="DSS | Welcome", layout="wide")
apply_tactical_theme()

col_title, col_status = st.columns([3, 1])

with col_title:
    st.title("DECISION SUPPORT SYSTEM")
    st.caption("CLANDESTINE NETWORK ANALYSIS & OPTIMIZATION")

with col_status:
    st.markdown(f"""
        <div style="border: 1px solid #30363D; padding: 15px; background-color: rgba(17, 20, 24, 0.8);">
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#58a6ff;">SYSTEM_STATUS: ACTIVE</p>
            <p style="margin:0; font-family: 'Share Tech Mono', monospace; font-size:12px; color:#8b949e;">OPERATOR: ADMIN</p>
        </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("### Welcome, Analyst")

st.markdown(
    "You are working with a decision support system for analyzing clandestine communication networks "
    "using **network science** and **mathematical optimization**."
)

st.markdown(
    "Scope: static snapshot of observed communications (62 members). "
    "Results reflect structural patterns rather than intent or hierarchy."
)


st.markdown("### What this system does")

st.markdown(
    "- **Identifies structurally important members** using multiple network perspectives.\n"
    "- **Classifies members into functional roles** based on their position in the network.\n"
    "- **Detects groups and structural weak points** within the network.\n"
    "- **Assesses network resilience** by simulating the removal of key members.\n"
    "- **Supports arrest planning** by translating analytical insights into actionable recommendations."
)


st.info(
    "Use the system comparatively: explore patterns, contrast perspectives, and treat disagreement as analytical signal."
)





