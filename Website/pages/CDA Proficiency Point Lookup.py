# pages/CDA Proficiency Point Lookup.py
import io
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------- App & logo ----------
APP_DIR = Path(__file__).resolve().parents[1]              # -> .../Website
LOGO = APP_DIR / "assets" / "pepperpot_wordmark.png"
st.set_page_config(
    page_title="Pepperpot • Proficiency Point Lookup",
    page_icon=str(LOGO) if LOGO.exists() else None,
    layout="wide",
)
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), use_container_width=True)

# ---------- Theme & profile (no email anywhere) ----------
try:
    from utils.db import get_prefs
except Exception:
    def get_prefs(uid: str): return {}

BORDER_COLOR = "#bbb"

def _theme_and_profile():
    base = {
        "theme_c1": "#FFD1DC",  # (default) ≥7  (pink)
        "theme_c2": "#FFD8A8",  # (default) negative→X (orange)
        "theme_c3": "#D3F9D8",  # (default) 1–6 (green)
        "theme_c4": "#D0EBFF",  # extra/unused here
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
PINK_BG   = T["theme_c1"]  # ≥7  (by default pink; users can change in My Account)
ORANGE_BG = T["theme_c2"]  # negatives → X  (by default orange)
GREEN_BG  = T["theme_c3"]  # 1–6
ZERO_BG   = "transparent"
DEFAULT_PERSON = T.get("display_name", "").strip() if st.session_state.get("user") else ""

# ---------- CSV path (bullet-proof) ----------
CSV_NAME = "CALCULATEDPPs.csv"
CSV_CANDIDATES = [
    APP_DIR / CSV_NAME,                 # Website/CALCULATEDPPs.csv
    APP_DIR / "data" / CSV_NAME,        # Website/data/CALCULATEDPPs.csv
    APP_DIR.parent / CSV_NAME,          # repo-root/CALCULATEDPPs.csv
]
def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None
DEFAULT_CSV_PATH = _first_existing(CSV_CANDIDATES)

# ---------- Config ----------
FONT_BODY_PX  = 18
FONT_STYLE_PX = 22
THRESHOLD     = 7

LEVELS = ["Newcomer", "Bronze", "Silver", "Gold", "Novice", "Prechamp", "Champ"]
STYLES: Dict[str, List[Tuple[str, str]]] = {
    "Standard": [("W", "Waltz"), ("T", "Tango"), ("V", "Viennese Waltz"), ("F", "Foxtrot"), ("Q", "Quickstep")],
    "Smooth":   [("W", "Waltz"), ("T", "Tango"), ("F", "Foxtrot"), ("V", "Viennese Waltz")],
    "Latin":    [("C", "Cha cha"), ("S", "Samba"), ("R", "Rumba"), ("P", "Paso Doble"), ("J", "Jive")],
    "Rhythm":   [("C", "Cha cha"), ("R", "Rumba"), ("S", "Swing"), ("B", "Bolero"), ("M", "Mambo")],
}
STYLE_LEVEL_ONLY = {"Novice", "Prechamp", "Champ"}  # style-wide cells (span across dances)

# ---------- Synonyms (robust header matching) ----------
STYLE_SYNONYMS: Dict[str, List[str]] = {
    # canonical -> variants you might actually see in headers
    "Standard": ["standard", "international standard", "intl standard", "international", "intl", "int'l", "int’l", "std"],
    "Smooth":   ["smooth", "american smooth", "american"],
    "Latin":    ["latin", "international latin", "intl latin", "international", "intl", "int'l", "int’l"],
    "Rhythm":   ["rhythm", "american rhythm", "american"],
}
LEVEL_SYNONYMS = {
    "Newcomer": ["newcomer","new","rookie","beginner"],
    "Bronze": ["bronze","open bronze","bronze open"],
    "Silver": ["silver","open silver","silver open"],
    "Gold":   ["gold","open gold","gold open"],
    "Novice": ["novice"],
    "Prechamp": ["prechamp","pre-champ","pre champ","pre championship","pre-championship","prechampionship"],
    "Champ": ["champ","championship","open champ","champ open"],
}
DANCE_SYNONYMS: Dict[str, List[str]] = {
    "Waltz":          ["waltz","w"],
    "Tango":          ["tango","t"],
    "Foxtrot":        ["foxtrot","f","ft"],
    "Viennese Waltz": ["viennese waltz","viennese","v waltz","vwaltz","v-waltz","v","vw"],
    "Quickstep":      ["quickstep","qs","q"],

    "Cha cha":        ["cha cha","chacha","cha-cha","cha","cha-cha-cha","c"],
    "Rumba":          ["rumba","r"],
    "Samba":          ["samba","s"],
    "Paso Doble":     ["paso doble","paso-doble","pasodoble","paso","p"],
    "Jive":           ["jive","j"],

    "Bolero":         ["bolero","b"],
    "Mambo":          ["mambo","m"],
    "Swing":          ["swing","east coast swing","ecs","sw"],
}

# ---------- CSV helpers ----------
def normalize(s: str) -> str:
    """Whitespace+punctuation-insensitive normalization used for matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()

def load_points_csv(source) -> pd.DataFrame:
    """
    Robust CSV reader for CALCULATEDPPs:

    - Sniffs delimiter (comma/tab/semicolon/pipe)
    - Handles files that start with banner rows (e.g., 'Standard/Smooth/Latin/Rhythm', 'Newcomer/Bronze/…')
    - Promotes the FIRST row that contains 'Dancer' (or 'UID') to the header row
    - Deduplicates/repairs blank column names
    """
    import pandas as pd
    import numpy as np
    import re

    # 1) Read with no header so we can pick which row is the real header
    raw = pd.read_csv(source, dtype=str, sep=None, engine="python", header=None, keep_default_na=False)

    # 2) Find the header row: first row containing 'Dancer' OR (fallback) 'UID'
    header_row = None
    max_scan = min(10, len(raw))
    for i in range(max_scan):
        cells = [str(x).strip().lower() for x in raw.iloc[i].tolist()]
        if any(c == "dancer" for c in cells) or any(c == "uid" for c in cells):
            header_row = i
            break

    # Fallback: if not found, keep pandas' first row as header
    if header_row is None:
        # Try the old path (works for simple CSVs)
        df_fallback = pd.read_csv(source, dtype=str, sep=None, engine="python")
        df_fallback.columns = [str(c).strip() for c in df_fallback.columns]
        return df_fallback.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # 3) Promote that row to header
    header_vals = [str(x).strip() for x in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()

    # 4) Repair blank/duplicate header names
    fixed_cols = []
    seen = {}
    for idx, name in enumerate(header_vals):
        col = name if name else f"col_{idx}"
        col = col.strip()
        # De-dup: add suffix for repeats
        base = col
        k = 1
        while col in seen:
            k += 1
            col = f"{base}__{k}"
        seen[col] = True
        fixed_cols.append(col)

    data.columns = fixed_cols

    # 5) Drop completely empty rows
    data = data.replace("", np.nan)
    data = data.dropna(how="all")
    data = data.fillna("")

    # 6) Strip whitespace from all string cells
    return data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

def find_name_column(df: pd.DataFrame) -> str:
    cands = [c for c in df.columns if re.search(r"\b(dancer|name|student|participant)\b", c, re.I)]
    if cands:
        return cands[0]
    # heuristic: find “looks like a full name”
    for c in df.columns:
        s = df[c].dropna().astype(str).str.strip()
        if len(s) and (s.str.match(r"^[A-Za-z][A-Za-z'\-\.]+(?:\s+[A-Za-z][A-Za-z'\-\.]+)+$")).mean() > 0.2:
            return c
    return df.columns[0]

def build_column_index(df: pd.DataFrame) -> Dict[str, str]:
    """Map normalized header → original header."""
    return {normalize(c): c for c in df.columns}

def _variants_for_style(style: str) -> List[str]:
    base = [style]
    base += STYLE_SYNONYMS.get(style, [])
    return [normalize(x) for x in base]

def _variants_for_level(level: str) -> List[str]:
    base = [level]
    base += LEVEL_SYNONYMS.get(level, [])
    return [normalize(x) for x in base]

def _variants_for_dance(dance_full: str) -> List[str]:
    base = [dance_full]
    base += DANCE_SYNONYMS.get(dance_full, [])
    return [normalize(x) for x in base]

def _permute_join(parts: List[str]) -> List[str]:
    """Return normalized strings for several orderings of header parts."""
    a = parts
    perms = [
        a,                                 # A B C
        [a[1], a[0], a[2]],                # B A C
        [a[0], a[2], a[1]],                # A C B
        [a[2], a[0], a[1]],                # C A B
        [a[1], a[2], a[0]],                # B C A
        [a[2], a[1], a[0]],                # C B A
    ]
    out = []
    for p in perms:
        out.append(normalize(" ".join(p)))
    return list(dict.fromkeys(out))

def possible_level_keys(level: str) -> List[str]:
    # kept for backward compatibility elsewhere
    return _variants_for_level(level)

def find_header_for_combo(col_index: Dict[str, str], style: str, level: str, dance_full: str) -> Optional[str]:
    """Find a column for a specific (style, level, dance) allowing synonyms and any header order."""
    styles = _variants_for_style(style)
    levels = _variants_for_level(level)
    dances = _variants_for_dance(dance_full)
    for sn in styles:
        for ln in levels:
            for dn in dances:
                for key in _permute_join([sn, ln, dn]):
                    if key in col_index:
                        return col_index[key]
    # Also tolerate headers missing the style when dance uniquely implies style (e.g., Quickstep, Bolero)
    UNIQUE_STYLE_BY_DANCE = {
        "quickstep": "standard",
        "samba": "latin", "jive": "latin", "paso doble": "latin",
        "bolero": "rhythm", "mambo": "rhythm", "swing": "rhythm",
    }
    if normalize(dance_full) in UNIQUE_STYLE_BY_DANCE:
        for ln in levels:
            for dn in dances:
                for key in _permute_join([ln, dn, UNIQUE_STYLE_BY_DANCE[normalize(dance_full)]]):
                    if key in col_index:
                        return col_index[key]
                key2 = normalize(" ".join([ln, dn]))
                if key2 in col_index:
                    return col_index[key2]
    return None

def find_header_for_style_level(col_index: Dict[str, str], style: str, level: str) -> Optional[str]:
    """Find a style-wide column (no dance), allowing synonyms and either order."""
    styles = _variants_for_style(style)
    levels = _variants_for_level(level)
    for sn in styles:
        for ln in levels:
            for key in (normalize(f"{sn} {ln}"), normalize(f"{ln} {sn}")):
                if key in col_index:
                    return col_index[key]
    return None

def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")

def compute_values(df: pd.DataFrame, name_col: str, person: str):
    mask = df[name_col].astype(str).str.strip().str.casefold() == person.strip().casefold()
    person_df = df.loc[mask].copy()

    # If no exact match, allow first prefix match
    if person_df.empty:
        contains = df[name_col].astype(str).str.contains(re.escape(person), case=False, na=False)
        if contains.any():
            person_df = df[contains].copy()

    col_index = build_column_index(df)
    cols = [(style, letter) for style, dances in STYLES.items() for letter, _ in dances]
    table_df = pd.DataFrame(0.0, index=LEVELS,
                            columns=pd.MultiIndex.from_tuples(cols, names=["Style", "Dance"]))
    style_level_vals: Dict[Tuple[str, str], float] = {}

    # Precompute per (level, style)
    for level in LEVELS:
        for style in STYLES.keys():
            style_val = 0.0
            hdr_style = find_header_for_style_level(col_index, style, level)
            if hdr_style is not None:
                style_val = float(to_num(person_df[hdr_style]).sum(skipna=True) or 0.0)
            else:
                s = 0.0
                for _, dance_full in STYLES[style]:
                    hdr = find_header_for_combo(col_index, style, level, dance_full)
                    if hdr is not None:
                        s += float(to_num(person_df[hdr]).sum(skipna=True) or 0.0)
                style_val = s
            style_level_vals[(level, style)] = style_val

        # Fill row: style-wide numbers for Novice/Prechamp/Champ; per-dance otherwise
        row_vals = []
        for style, dances in STYLES.items():
            for _, dance_full in dances:
                hdr = find_header_for_combo(col_index, style, level, dance_full)
                if hdr is not None:
                    val = float(to_num(person_df[hdr]).sum(skipna=True) or 0.0)
                else:
                    val = style_level_vals[(level, style)] if level in STYLE_LEVEL_ONLY else 0.0
                row_vals.append(val)
        table_df.loc[level] = row_vals

    return table_df, style_level_vals

# ---------- Cell styling (pastel + X for negatives) ----------
def cell_style(value: float) -> Tuple[str, str]:
    if value < 0:
        return f"background:{ORANGE_BG}; color:#000;", "X"
    if value == 0:
        return f"background:{ZERO_BG}; color:#000;", "0"
    if value >= THRESHOLD:
        return f"background:{PINK_BG}; color:#000;", f"{int(value):d}"
    return f"background:{GREEN_BG}; color:#000;", f"{int(value):d}"

# ---------- HTML table (colspans + full border) ----------
def render_html_table(table_df: pd.DataFrame, style_level_vals: Dict[Tuple[str, str], float]) -> str:
    def header_html() -> str:
        html = ['<thead>']
        html.append('<tr>')
        html.append('<th class="corner filled"></th>')
        for style, dances in STYLES.items():
            html.append(f'<th class="stylehead" colspan="{len(dances)}">{style}</th>')
        html.append('</tr>')
        html.append('<tr>')
        html.append('<th class="leveltag">Level</th>')
        for style, dances in STYLES.items():
            for letter, _ in dances:
                html.append(f'<th class="dancehead">{letter}</th>')
        html.append('</tr></thead>')
        return ''.join(html)

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
                    for letter, _ in dances:
                        v = float(table_df.loc[lvl, (style, letter)])
                        style_str, text = cell_style(v)
                        rows.append(f'<td style="{style_str}">{text}</td>')
            rows.append('</tr>')
            rows.append('\n')
        rows.append('</tbody>')
        return ''.join(rows)

    css = f"""
    <style>
      .proficiency-wrap {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 8px;
        overflow: hidden;
      }}
      table.proficiency {{
        border-collapse: collapse;
        width: 100%;
        border: 1px solid {BORDER_COLOR};
      }}
      table.proficiency th, table.proficiency td {{
        border: 1px solid {BORDER_COLOR};
        padding: 8px;
        text-align: center;
        font-size: {FONT_BODY_PX}px;
      }}
      table.proficiency th.stylehead {{
        background: #f7f7fb;
        font-size: {FONT_STYLE_PX}px;
        font-weight: 700;
        text-align: center;
      }}
      table.proficiency th.dancehead {{ background: #fafafa; text-align: center; }}
      table.proficiency th.leveltag  {{ background: #fff; font-weight: 700; text-align: center; }}
      table.proficiency th.levelhead {{ background: #fff; text-align: center; }}
      table.proficiency th.corner.filled {{
        background: #f7f7fb;
        border: 1px solid {BORDER_COLOR};
      }}
      table.proficiency td.span {{
        border-left: 1px solid {BORDER_COLOR};
        border-right: 1px solid {BORDER_COLOR};
        text-align: center;
      }}
    </style>
    """
    return f'<div class="proficiency-wrap">{css}<table class="proficiency">{header_html()}{body_html()}</table></div>'

# ---------- Sidebar: data source (toggle) ----------
if "show_data_source" not in st.session_state:
    st.session_state.show_data_source = False

def _toggle_data_source():
    st.session_state.show_data_source = not st.session_state.show_data_source

with st.sidebar:
    st.markdown("### Data")
    st.button("🔧 Data source", on_click=_toggle_data_source)
    up = None
    use_default = DEFAULT_CSV_PATH is not None
    label = f"Use local file: {DEFAULT_CSV_PATH.name if DEFAULT_CSV_PATH else '—'}"
    if st.session_state.show_data_source:
        up = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
        use_default = st.checkbox(label, value=use_default, disabled=(DEFAULT_CSV_PATH is None), key="use_default")

# ---------- Load data ----------
st.title("CDA Proficiency Point Lookup")

df, errs = None, []
if up is not None:
    try:
        df = load_points_csv(up)
    except Exception as e:
        errs.append(f"Upload load error: {e}")
elif use_default and DEFAULT_CSV_PATH:
    try:
        with DEFAULT_CSV_PATH.open("rb") as f:
            df = load_points_csv(io.BytesIO(f.read()))
    except Exception as e:
        errs.append(f"Local file load error: {e}")

if df is None:
    st.info("Click **🔧 Data source** in the sidebar to upload a CSV or enable the local file.")
    if errs:
        with st.expander("Load errors"):
            for e in errs:
                st.error(e)
    st.stop()

# ---------- Page content ----------
name_col = find_name_column(df)

st.subheader("Find a person")
prefill = DEFAULT_PERSON if st.session_state.get("user") else ""
query = st.text_input("Enter name", value=prefill, placeholder="e.g., Jane Doe").strip()

names = df[name_col].dropna().astype(str).str.strip()
if query:
    opts = sorted(names[names.str.contains(re.escape(query), case=False, na=False)].unique().tolist())
else:
    opts = sorted(names.unique().tolist()[:200])

# If not logged in, start blank
user_logged_in = bool(st.session_state.get("user"))
if not user_logged_in:
    opts = [""] + opts
elif DEFAULT_PERSON and DEFAULT_PERSON in opts:
    opts.remove(DEFAULT_PERSON)
    opts.insert(0, DEFAULT_PERSON)

if not opts or (len(opts) == 1 and opts[0] == ""):
    st.warning("No matching names found.")
    st.stop()

person = st.selectbox("Select person", opts, index=0, placeholder="Select a person")
if not person:
    st.info("Select a person to view proficiency points.")
    st.stop()

table_df, style_level_vals = compute_values(df, name_col, person)
html = render_html_table(table_df, style_level_vals)
components.html(html, height=min(1200, 180 + 48 * max(1, len(table_df))), scrolling=True)

st.caption(f"Points calculated on {date.today():%B %d, %Y}")
