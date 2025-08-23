# pages/My Account.py
import streamlit as st
from utils.db import get_prefs, set_prefs, get_user, update_user_login
from utils.auth import DANCE_OPTIONS, COLOR_PALETTE, gen_salt, make_hash

st.set_page_config(page_title="Pepperpot • My Account", layout="wide")



# ---- auth check ----
user = st.session_state.get("user")
if not user:
    st.warning("Please sign in on the Home page first.")
    st.page_link("Home.py", label="⬅️ Go to Home")
    st.stop()
uid = user["id"]



# ---- defaults & load (NO email) ----
DEFAULTS = {
    "display_name": "",
    "theme_c1": "#FFD1DC",  # ≥7
    "theme_c2": "#FFD8A8",  # negative→X
    "theme_c3": "#D3F9D8",  # 1–6
    "theme_c4": "#D0EBFF",  # extra/unused
}
db_saved = get_prefs(uid)
prefs = {k: db_saved.get(k, v) for k, v in DEFAULTS.items()}

# baselines
if "account_saved_prefs" not in st.session_state:
    st.session_state["account_saved_prefs"] = {k: db_saved.get(k, v) for k, v in DEFAULTS.items()}
if "account_have_saved_once" not in st.session_state:
    st.session_state["account_have_saved_once"] = bool(db_saved)

# login baseline
user_row = get_user(uid)  # (username, pass_hash, salt, fav_dance, fav_color_name, fav_color_hex)
if user_row is None:
    st.error("Account record not found. Please sign out and sign in again.")
    st.stop()
_, _, _, curr_dance, curr_color_name, curr_color_hex = user_row
if "login_saved_combo" not in st.session_state:
    st.session_state["login_saved_combo"] = (curr_dance, curr_color_name)

def is_account_dirty(current: dict, baseline: dict) -> bool:
    for k in DEFAULTS.keys():
        if str(current.get(k, "")) != str(baseline.get(k, "")):
            return True
    return False

# ---- Global CSS (banner + button + pickers) ----
st.markdown(
    """
    <style>
      .unsaved-banner {
        position: fixed; top: 0; left: 0; right: 0;
        background: #ffe5e5; color: #c92a2a; border-bottom: 1px solid #e03131;
        padding: 10px 16px; text-align: center; z-index: 1000; font-weight: 600;
        transform: translateY(-100%); animation: slideDown 0.5s ease-out forwards;
      }
      @keyframes slideDown { from {transform:translateY(-100%);} to {transform:translateY(0);} }
      div[data-testid="stAppViewContainer"] .main .block-container { padding-top: 68px; }

      div[data-testid="stColorPicker"] { padding:0 !important; margin:0 !important; }
      div[data-testid="stColorPicker"] label { display:none !important; }
      div[data-testid="stColorPicker"] input[type="color"],
      div[data-testid="stColorPicker"] canvas,
      div[data-testid="stColorPicker"] button {
        width:72px !important; height:72px !important;
        border-radius:8px !important; border:1px solid #bbb !important;
        padding:0 !important; margin:0 !important;
      }

      .stButton > button:disabled {
        background:#e9ecef !important; color:#495057 !important;
        border:1px solid #ced4da !important; border-radius:8px !important;
        cursor:not-allowed !important;
      }
      .stButton > button:not(:disabled) {
        background:#D3F9D8 !important; color:#0b4228 !important;
        border:1px solid #69db7c !important; border-radius:8px !important;
        cursor:pointer !important;
      }
      .stButton > button:not(:disabled):hover { filter: brightness(0.97); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("My Account")

# ---- Profile (NO email) ----
prefs["display_name"] = st.text_input("My Name", value=prefs["display_name"])

# ---- Theme (2×2 window) ----
st.subheader("Theme")
r1 = st.columns(2)
with r1[0]:
    prefs["theme_c1"] = st.color_picker(" ", prefs["theme_c1"], key="c1", label_visibility="collapsed")
with r1[1]:
    prefs["theme_c2"] = st.color_picker(" ", prefs["theme_c2"], key="c2", label_visibility="collapsed")
r2 = st.columns(2)
with r2[0]:
    prefs["theme_c3"] = st.color_picker(" ", prefs["theme_c3"], key="c3", label_visibility="collapsed")
with r2[1]:
    prefs["theme_c4"] = st.color_picker(" ", prefs["theme_c4"], key="c4", label_visibility="collapsed")

# ---- Login (favorite dance + color) ----
st.subheader("Login")
lc1, lc2 = st.columns(2)
with lc1:
    new_dance = st.selectbox(
        "Favorite dance",
        DANCE_OPTIONS,
        index=DANCE_OPTIONS.index(curr_dance) if curr_dance in DANCE_OPTIONS else 0,
        key="login_dance",
    )
with lc2:
    color_names = list(COLOR_PALETTE.keys())
    start_idx = color_names.index(curr_color_name) if curr_color_name in color_names else 0
    new_color_name = st.selectbox(
        "Favorite color",
        color_names,
        index=start_idx,
        key="login_color",
    )
    new_color_hex = COLOR_PALETTE[new_color_name]
    st.markdown(
        f"<div style='width:72px;height:24px;border:1px solid #bbb;border-radius:6px;background:{new_color_hex};'></div>",
        unsafe_allow_html=True,
    )

# ---- Dirty checks ----
account_dirty = (not st.session_state.get("account_have_saved_once", False)) or \
                is_account_dirty(prefs, st.session_state.get("account_saved_prefs", DEFAULTS))
base_dance, base_color_name = st.session_state["login_saved_combo"]
login_dirty = (new_dance != base_dance) or (new_color_name != base_color_name)
anything_dirty = account_dirty or login_dirty

# ---- Top banner (only when dirty) ----
if anything_dirty:
    st.markdown('<div class="unsaved-banner">Unsaved changes</div>', unsafe_allow_html=True)

# ---- Save everything ----
clicked = st.button("Save changes", disabled=not anything_dirty)
if clicked:
    # 1) save profile + theme
    set_prefs(uid, prefs)
    st.session_state["account_saved_prefs"] = prefs.copy()
    st.session_state["account_have_saved_once"] = True

    # 2) save login combo if changed
    if login_dirty:
        salt = gen_salt()
        phash = make_hash(uid, new_dance, new_color_hex, salt)
        update_user_login(
            username=uid,
            pass_hash=phash,
            salt=salt,
            fav_dance=new_dance,
            fav_color_name=new_color_name,
            fav_color_hex=new_color_hex,
        )
        st.session_state["login_saved_combo"] = (new_dance, new_color_name)

    # 3) mirror theme/profile for other pages
    st.session_state["theme_overrides"] = {
        "theme_c1": prefs["theme_c1"],
        "theme_c2": prefs["theme_c2"],
        "theme_c3": prefs["theme_c3"],
        "theme_c4": prefs["theme_c4"],
        "display_name": prefs["display_name"],
    }
    st.rerun()

st.page_link("pages/CDA Proficiency Point Lookup.py", label="Open CDA Proficiency Point Lookup")
