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
st.markdown("""### WELCOME, ANALYST
You are working with a decision support system for analyzing clandestine communication networks "
"using network science and optimization-based methods."
""")
st.caption(
    "Scope: static snapshot of observed communications (N=62). "
    "Results reflect structural patterns, not intent or hierarchy."
)

c1, c2, c3 = st.columns(3)
c1.metric("Members", 62)
c2.metric("Observed links", 159)
c3.metric("Analysis modes", 5)

st.info(
    "Use this system comparatively: explore patterns, contrast perspectives, "
    "and treat disagreement as analytical signal."
)


