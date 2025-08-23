# Home.py
import streamlit as st
from utils.db import get_user, create_user, touch_user
from utils.auth import DANCE_OPTIONS, COLOR_PALETTE, gen_salt, make_hash

from pathlib import Path
LOGO = Path(__file__).resolve().parents[1] / "assets" / "pepperpot_wordmark.png"
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), use_column_width=True)

st.set_page_config(page_title="Pepperpot", layout="wide")
st.title("Pepperpot")



def auth_ui():
    st.subheader("Sign in or create an account")
    tab_login, tab_signup = st.tabs(["Log in", "Create account"])

    # ---- Sign up (no email) ----
    with tab_signup:
        su_username = st.text_input("Username (letters, numbers, underscores)", key="su_user")
        su_dance = st.selectbox("Favorite dance (used for login)", DANCE_OPTIONS, key="su_dance")
        color_names = list(COLOR_PALETTE.keys())
        su_color_name = st.selectbox("Favorite color (used for login)", color_names, key="su_color")
        su_color_hex = COLOR_PALETTE[su_color_name]

        if st.button("Create account"):
            if not su_username or not su_username.replace("_", "").isalnum():
                st.error("Username must be letters/numbers/underscores.")
            elif get_user(su_username) is not None:
                st.error("That username is taken.")
            else:
                salt = gen_salt()
                phash = make_hash(su_username, su_dance, su_color_hex, salt)
                ok = create_user(
                    username=su_username,
                    pass_hash=phash,
                    salt=salt,
                    fav_dance=su_dance,
                    fav_color_name=su_color_name,
                    fav_color_hex=su_color_hex,
                )
                if ok:
                    st.session_state["user"] = {"id": su_username, "name": su_username}
                    st.success("Account created and signed in!")
                    st.rerun()
                else:
                    st.error("Could not create account. Try another username.")

    # ---- Log in (dance & color start BLANK) ----
    with tab_login:
        lg_username = st.text_input("Username", key="lg_user")

        dance_options = ["—"] + DANCE_OPTIONS
        color_names = list(COLOR_PALETTE.keys())
        color_options = ["—"] + color_names

        lg_dance = st.selectbox("Favorite dance", dance_options, index=0, key="lg_dance_label")
        lg_color = st.selectbox("Favorite color", color_options, index=0, key="lg_color_label")

        if st.button("Log in"):
            if not lg_username:
                st.error("Enter your username.")
            elif lg_dance == "—" or lg_color == "—":
                st.error("Select your favorite dance and color.")
            else:
                rec = get_user(lg_username)
                if rec is None:
                    st.error("Unknown username.")
                else:
                    _, pass_hash_db, salt, *_ = rec
                    attempt = make_hash(lg_username, lg_dance, COLOR_PALETTE[lg_color], salt)
                    if attempt == pass_hash_db:
                        st.session_state["user"] = {"id": lg_username, "name": lg_username}
                        touch_user(lg_username)
                        st.success("Signed in!")
                        st.rerun()
                    else:
                        st.error("Dance + color do not match. Try again.")

user = st.session_state.get("user")
if not user:
    auth_ui()
    st.stop()

st.success(f"Signed in as **{user['id']}**")

st.divider()
st.subheader("Pages")
st.page_link("pages/My Account.py", label="👤 My Account")
st.page_link("pages/CDA Proficiency Point Lookup.py", label="📊 CDA Proficiency Point Lookup")

st.divider()
if st.button("Sign out"):
    st.session_state.pop("user", None)
    st.session_state.pop("theme_overrides", None)
    st.success("Signed out.")
    st.rerun()
