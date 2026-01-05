import streamlit as st
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
import os

# https://www.authgear.com/tools/password-hash-generator
USERS = {
    "admin": "$argon2id$v=19$m=19456,t=2,p=1$YjNlMTcyYTIwYzkxNzVhNjYxNjAyYjE2ODZhZWI2Y2M$ouBS5CiPrpo2U7dxzEdKiU5tSB4crnLnGZzSdKbuI7s", # password: admin
}

if "logged_in" not in st.session_state:
    if os.path.exists(".venv"): 
        st.session_state.logged_in = True
        st.session_state.username = "admin"
    else:
        st.session_state.logged_in = False


def login_screen():
    st.title("DSS Clandestine Network Analysis")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    def verify_password(username: str, password: str) -> bool:
        if username not in USERS:
            return False

        try:
            PasswordHasher().verify(USERS[username], password)
            return True

        except VerifyMismatchError:
            return False

    if st.button("Log in"):
        if verify_password(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout_screen():
    st.session_state.logged_in = False
    st.rerun()

if st.session_state.logged_in:
    pg = st.navigation([
        st.Page("0_welcome.py", title="Welcome", icon=":material/dashboard:", default=True),
        st.Page("1_members.py", title="Members", icon=":material/people:"),
        st.Page("2_roles.py", title="Roles", icon=":material/security:"),
        st.Page(logout_screen, title="Log out", icon=":material/logout:")
    ])
else:
    pg = st.navigation([
        st.Page(login_screen, title="Log in", icon=":material/login:")
    ])

pg.run()
