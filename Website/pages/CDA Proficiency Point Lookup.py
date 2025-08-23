# pages/CDA Proficiency Point Lookup.py
import io, os, re
from datetime import date
from typing import Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from utils.db import get_prefs
except Exception:
    def get_prefs(uid: str): return {}


st.set_page_config(page_title="Pepperpot • Proficiency Point Lookup", layout="wide")

# ===== Theme + Profile (from session/DB; NO email) =====
BORDER_COLOR = "#bbb"

def _theme_and_profile():
    base = {
        "theme_c1": "#FFD1DC",  # ≥7
        "theme_c2": "#FFD8A8",  # negative→X
        "theme_c3": "#D3F9D8",  # 1–6
        "theme_c4": "#D0EBFF",  # unused here
        "display_name": "",
    }
    overrides = st.session_state.get("theme_overrides") or {}
    user = st.session_state.get("user")
    if user and not overrides:
        dbp = get_prefs(user["id"])
        for k in base:
            if k in dbp:
                base[k] = dbp[k]
    base.update(overrides)
    return base

T = _theme_and_profile()
PINK_BG, ORANGE_BG, GREEN_BG = T["theme_c1"], T["theme_c2"], T["theme_c3"]
ZERO_BG = "transparent"
DEFAULT_PERSON = T.get("display_name", "").strip() if st.session_state.get("user") else ""

# ===== Config =====
DEFAULT_CSV   = "PPMid-March2025.csv"
FONT_BODY_PX  = 18
FONT_STYLE_PX = 22
THRESHOLD     = 7

LEVELS = ["Newcomer", "Bronze", "Silver", "Gold", "Novice", "Prechamp", "Champ"]
STYLES: Dict[str, List[Tuple[str, str]]] = {
    "Standard": [("W","Waltz"),("T","Tango"),("V","Viennese Waltz"),("F","Foxtrot"),("Q","Quickstep")],
    "Smooth":   [("W","Waltz"),("T","Tango"),("F","Foxtrot"),("V","Viennese Waltz")],
    "Latin":    [("C","Cha cha"),("S","Samba"),("R","Rumba"),("P","Paso Doble"),("J","Jive")],
    "Rhythm":   [("C","Cha cha"),("R","Rumba"),("S","Swing"),("B","Bolero"),("M","Mambo")],
}
STYLE_LEVEL_ONLY = {"Novice","Prechamp","Champ"}
LEVEL_SYNONYMS = {"prechamp":["prechamp","pre-champ","pre champ","prechampionship"], "champ":["champ","championship"]}

# ===== Helpers =====
def normalize(s: str) -> str: return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def load_points_csv(source) -> pd.DataFrame:
    df_raw = pd.read_csv(source, dtype=str)
    first = df_raw.iloc[0].fillna("").astype(str).str.strip().tolist()
    looks_like_header = any(x.lower()=="dancer" for x in first)
    if looks_like_header:
        df = df_raw.iloc[1:].copy()
        df.columns = [str(x).strip() if str(x).strip() else f"col_{i}" for i,x in enumerate(first)]
    else:
        df = df_raw.copy(); df.columns = [str(c).strip() for c in df.columns]
    return df.applymap(lambda x: x.strip() if isinstance(x,str) else x)

def find_name_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if re.search(r"\b(dancer|name|student|participant)\b", c, re.I)]
    if cands: return cands[0]
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip()
        if len(s) and (s.str.match(r"^[A-Za-z][A-Za-z'\-\.]+(?:\s+[A-Za-z][A-Za-z'\-\.]+)+$")).mean() > 0.2:
            return c
    return df.columns[0]

def build_column_index(df: pd.DataFrame) -> Dict[str, str]:
    return {normalize(c): c for c in df.columns}

def possible_level_keys(level: str) -> List[str]:
    key = normalize(level); out = [key]
    if key in LEVEL_SYNONYMS: out += [normalize(x) for x in LEVEL_SYNONYMS[key]]
    return list(dict.fromkeys(out))

def find_header_for_combo(col_index: Dict[str,str], style: str, level: str, dance_full: str) -> Optional[str]:
    style_n, dance_n = normalize(style), normalize(dance_full)
    for lvl_n in possible_level_keys(level):
        t1 = f"{style_n} {lvl_n} {dance_n}"; t2 = f"{lvl_n} {style_n} {dance_n}"
        if t1 in col_index: return col_index[t1]
        if t2 in col_index: return col_index[t2]
    return None

def find_header_for_style_level(col_index: Dict[str,str], style: str, level: str) -> Optional[str]:
    style_n = normalize(style)
    for lvl_n in possible_level_keys(level):
        t1 = f"{style_n} {lvl_n}"; t2 = f"{lvl_n} {style_n}"
        if t1 in col_index: return col_index[t1]
        if t2 in col_index: return col_index[t2]
    return None

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

def compute_values(df: pd.DataFrame, name_col: str, person: str):
    mask = df[name_col].astype(str).str.strip().str.casefold() == person.strip().casefold()
    person_df = df.loc[mask].copy()
    col_index = build_column_index(df)
    cols = [(style, letter) for style, dances in STYLES.items() for letter, _ in dances]
    table_df = pd.DataFrame(0.0, index=LEVELS, columns=pd.MultiIndex.from_tuples(cols, names=["Style","Dance"]))
    style_level_vals: Dict[Tuple[str,str], float] = {}
    for level in LEVELS:
        for style in STYLES.keys():
            hdr_style = find_header_for_style_level(col_index, style, level)
            if hdr_style is not None:
                val = float(to_num(person_df[hdr_style]).sum(skipna=True) or 0.0)
            else:
                s = 0.0
                for _, dance_full in STYLES[style]:
                    hdr = find_header_for_combo(col_index, style, level, dance_full)
                    if hdr is not None:
                        s += float(to_num(person_df[hdr]).sum(skipna=True) or 0.0)
                val = s
            style_level_vals[(level, style)] = val
        row_vals = []
        for style, dances in STYLES.items():
            for _, dance_full in dances:
                hdr = find_header_for_combo(col_index, style, level, dance_full)
                if hdr is not None:
                    v = float(to_num(person_df[hdr]).sum(skipna=True) or 0.0)
                else:
                    v = style_level_vals[(level, style)] if level in STYLE_LEVEL_ONLY else 0.0
                row_vals.append(v)
        table_df.loc[level] = row_vals
    return table_df, style_level_vals

# ===== Cell styling =====
def cell_style(value: float):
    if value < 0: return f"background:{ORANGE_BG}; color:#000;", "X"
    if value == 0: return f"background:{ZERO_BG}; color:#000;", "0"
    if value >= THRESHOLD: return f"background:{PINK_BG}; color:#000;", f"{int(value):d}"
    return f"background:{GREEN_BG}; color:#000;", f"{int(value):d}"

# ===== HTML table =====
def render_html_table(table_df: pd.DataFrame, style_level_vals: Dict[Tuple[str,str], float]) -> str:
    def header_html() -> str:
        html = ['<thead>','<tr>','<th class="corner filled"></th>']
        for style, dances in STYLES.items():
            html.append(f'<th class="stylehead" colspan="{len(dances)}">{style}</th>')
        html.append('</tr><tr><th class="leveltag">Level</th>')
        for style, dances in STYLES.items():
            for letter,_ in dances: html.append(f'<th class="dancehead">{letter}</th>')
        html.append('</tr></thead>'); return ''.join(html)

    def body_html() -> str:
        rows = ['<tbody>']
        for lvl in LEVELS:
            rows.append('<tr>')
            rows.append(f'<th class="levelhead">{lvl}</th>')
            for style, dances in STYLES.items():
                if lvl in STYLE_LEVEL_ONLY:
                    val = float(style_level_vals.get((lvl, style), 0.0))
                    style_str, text = cell_style(val)
                    rows.append(f'<td class="span" colspan="{len(dances)}" style="{style_str}">{text}</td>')
                else:
                    for letter,_ in dances:
                        v = float(table_df.loc[lvl, (style, letter)])
                        style_str, text = cell_style(v)
                        rows.append(f'<td style="{style_str}">{text}</td>')
            rows.append('</tr>')
        rows.append('</tbody>'); return ''.join(rows)

    css = f"""
    <style>
      .proficiency-wrap {{ border: 1px solid {BORDER_COLOR}; border-radius: 8px; overflow: hidden; }}
      table.proficiency {{ border-collapse: collapse; width: 100%; border: 1px solid {BORDER_COLOR}; }}
      table.proficiency th, table.proficiency td {{ border: 1px solid {BORDER_COLOR}; padding: 8px; text-align: center; font-size: {FONT_BODY_PX}px; }}
      table.proficiency th.stylehead {{ background: #f7f7fb; font-size: {FONT_STYLE_PX}px; font-weight: 700; text-align: center; }}
      table.proficiency th.dancehead {{ background: #fafafa; text-align: center; }}
      table.proficiency th.leveltag  {{ background: #fff; font-weight: 700; text-align: center; }}
      table.proficiency th.levelhead {{ background: #fff; text-align: center; }}
      table.proficiency th.corner.filled {{ background: #f7f7fb; border: 1px solid {BORDER_COLOR}; }}
      table.proficiency td.span {{ border-left: 1px solid {BORDER_COLOR}; border-right: 1px solid {BORDER_COLOR}; text-align: center; }}
    </style>
    """
    return f'<div class="proficiency-wrap">{css}<table class="proficiency">{header_html()}{body_html()}</table></div>'

# ===== Sidebar: data source =====
if "show_data_source" not in st.session_state:
    st.session_state.show_data_source = False
def _toggle_data_source(): st.session_state.show_data_source = not st.session_state.show_data_source

with st.sidebar:
    st.markdown("### Data")
    st.button("🔧 Data source", on_click=_toggle_data_source)
    up = None
    use_default = os.path.exists(DEFAULT_CSV)
    if st.session_state.show_data_source:
        up = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        use_default = st.checkbox(f"Use local file: {DEFAULT_CSV}", value=use_default, key="use_default")

# ===== Load data =====
st.title("CDA Proficiency Point Lookup")
df, errs = None, []
if up is not None:
    try: df = load_points_csv(up)
    except Exception as e: errs.append(f"Upload load error: {e}")
elif use_default:
    try:
        with open(DEFAULT_CSV, "rb") as f: df = load_points_csv(io.BytesIO(f.read()))
    except Exception as e: errs.append(f"Local file load error: {e}")

if df is None:
    st.info("Click **🔧 Data source** in the sidebar to upload a CSV or enable the local file.")
    if errs:
        with st.expander("Load errors"):
            for e in errs: st.error(e)
    st.stop()

# ===== Content =====
name_col = find_name_column(df)

st.subheader("Find a person")
prefill = DEFAULT_PERSON if st.session_state.get("user") else ""
query = st.text_input("Enter name", value=prefill, placeholder="e.g., Jane Doe").strip()

names = df[name_col].dropna().astype(str).str.strip()
opts = sorted(names[names.str.contains(re.escape(query), case=False, na=False)].unique().tolist()) if query else sorted(names.unique().tolist()[:200])

user_logged_in = bool(st.session_state.get("user"))
if not user_logged_in:
    opts = [""] + opts
elif DEFAULT_PERSON and DEFAULT_PERSON in opts:
    opts.remove(DEFAULT_PERSON); opts.insert(0, DEFAULT_PERSON)

if not opts or (len(opts) == 1 and opts[0] == ""):
    st.warning("No matching names found."); st.stop()

person = st.selectbox("Select person", opts, index=0, placeholder="Select a person")
if not person:
    st.info("Select a person to view proficiency points."); st.stop()

table_df, style_level_vals = compute_values(df, name_col, person)
html = render_html_table(table_df, style_level_vals)
components.html(html, height=min(1200, 180 + 48 * max(1, len(table_df))), scrolling=True)

st.caption(f"Points calculated on {date.today():%B %d, %Y}")
