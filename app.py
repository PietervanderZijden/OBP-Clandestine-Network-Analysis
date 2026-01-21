import os

import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Import your shared theme logic
from ui_components import apply_tactical_theme

# --- Page Configuration ---
st.set_page_config(page_title="DSS Command Center", layout="wide")

# Apply the theme globally to app.py
apply_tactical_theme()

# --- Authentication Logic ---
USERS = {
    "admin": "$argon2id$v=19$m=19456,t=2,p=1$YjNlMTcyYTIwYzkxNzVhNjYxNjAyYjE2ODZhZWI2Y2M$ouBS5CiPrpo2U7dxzEdKiU5tSB4crnLnGZzSdKbuI7s",  # admin
}

if "logged_in" not in st.session_state:
    # Auto-login for local development if .venv exists
    if os.path.exists(".venv"):
        st.session_state.logged_in = True
        st.session_state.username = "admin"
    else:
        st.session_state.logged_in = False


def login_screen():
    # Styled Header for Login
    st.title("DSS // CLANDESTINE NETWORK ANALYSIS")
    st.caption("TACTICAL COMMAND OS // UNAUTHORIZED ACCESS PROHIBITED")

    # Use a container to frame the login box
    with st.container(border=True):
        username = st.text_input("CREDENTIAL_ID")
        password = st.text_input("ACCESS_KEY", type="password")

        def verify_password(u: str, p: str) -> bool:
            if u not in USERS:
                return False
            try:
                PasswordHasher().verify(USERS[u], p)
                return True
            except VerifyMismatchError:
                return False

        if st.button("AUTHORIZE ACCESS"):
            if verify_password(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("ACCESS DENIED: INVALID CREDENTIALS")


def logout_screen():
    st.session_state.logged_in = False
    st.rerun()


# --- Navigation ---
if st.session_state.logged_in:
    pg = st.navigation(
        [
            st.Page("0_welcome.py", title="Welcome", default=True),
            st.Page("src/6_data_import.py", title="Import Data"),
            st.Page("1_members.py", title="Members"),
            st.Page("2_roles.py", title="Roles"),
            st.Page("src/3_factions.py", title="Factions"),
            st.Page("4_kemeny.py", title="Kemeny Constant"),
            st.Page("src/5_arrest_plan.py", title="Arrest Plan"),
            st.Page(logout_screen, title="Log out"),
        ]
    )
else:
    pg = st.navigation([st.Page(login_screen, title="Log in", icon=":material/login:")])

pg.run()
