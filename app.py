"""
VIVC Grape Variety Pedigree Explorer — Python / Streamlit port
Replicates the R/Shiny app using pandas, networkx, pyvis, and plotly.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VIVC Pedigree Explorer",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — burgundy/wine theme matching the R app ──────────────────────
st.markdown("""
<style>
/* Warm parchment background */
.stApp { background: #f0ece2; font-family: 'Segoe UI', Arial, sans-serif; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #faf7f0;
    border-right: 1px solid #c8b99a;
}
section[data-testid="stSidebar"] > div { padding-top: 1rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 2px solid #7b1c2e;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #4a3728;
    font-size: 13px;
    padding: 6px 14px;
    border-radius: 4px 4px 0 0;
}
.stTabs [aria-selected="true"] {
    color: #7b1c2e !important;
    font-weight: 700;
    border-top: 2px solid #7b1c2e;
    background: #fff;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #ffffff;
    border-left: 4px solid #4a7c30;
    border-radius: 6px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}

/* Headings */
h1, h2 { color: #7b1c2e; }
h3 { color: #4a3728; }

/* Buttons */
.stButton > button {
    background: #4a7c30;
    color: white;
    border: none;
    border-radius: 4px;
}
.stButton > button:hover { background: #3a6225; }

/* Sidebar section headers */
.sidebar-heading {
    font-size: 11px;
    font-weight: 700;
    color: #7b1c2e;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 14px 0 4px 0;
}

hr { border-color: #d4c4a0; }

/* DataFrame / table */
.dataframe { font-size: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Colour palettes ───────────────────────────────────────────────────────────

BERRY_COLOR_HEX = {
    "BLANC":      "#e8d96a",
    "NOIR":       "#2d1b4e",
    "ROUGE":      "#8b1a2e",
    "ROSE":       "#d9728a",
    "GRIS":       "#9b8abf",
    "RED-VIOLET": "#6b2060",
}
BERRY_COLOR_DEFAULT = "#c8bfb0"

BERRY_COLOR_LABELS = {
    "BLANC":      "Blanc (white)",
    "NOIR":       "Noir (black)",
    "ROUGE":      "Rouge (red)",
    "ROSE":       "Rosé",
    "GRIS":       "Gris (grey)",
    "RED-VIOLET": "Red-Violet",
}

MARKER_COLORS = {
    "confirmed":    "#1e9645",
    "probable":     "#d4920c",
    "disputed":     "#c45d1a",
    "refuted":      "#cc2222",
    "undocumented": "#c0bab2",
}
MARKER_WIDTHS = {
    "confirmed":    4.0,
    "probable":     2.4,
    "disputed":     1.2,
    "refuted":      1.0,
    "undocumented": 0.45,
}
MARKER_GLYPHS = {
    "confirmed":    "✓",
    "probable":     "~",
    "disputed":     "?",
    "refuted":      "✗",
    "undocumented": "",
}
CONFIDENCE_ORDER = ["confirmed", "probable", "disputed", "refuted", "undocumented"]

# Generation ramp: forest green → sage → olive → pale straw (15 stops)

def _lerp_hex(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"

_GEN_STOPS = ["#2d5a1b", "#4a7c30", "#6ea64f", "#a8c97a", "#c8d9a0", "#e2eaca", "#f0eedd"]
GEN_RAMP: list[str] = []
for _i in range(15):
    _t = _i / 14.0
    _seg = _t * (len(_GEN_STOPS) - 1)
    _idx = min(int(_seg), len(_GEN_STOPS) - 2)
    GEN_RAMP.append(_lerp_hex(_GEN_STOPS[_idx], _GEN_STOPS[_idx + 1], _seg - _idx))


def berry_color_hex(berry_color: str | None) -> str:
    if berry_color is None or (isinstance(berry_color, float) and np.isnan(berry_color)):
        return BERRY_COLOR_DEFAULT
    key = str(berry_color).strip().upper()
    return BERRY_COLOR_HEX.get(key, BERRY_COLOR_DEFAULT)


def gen_color(node_type: str, generation: int | float) -> str:
    if node_type == "Target":
        return "#7b1c2e"
    if node_type == "Founder/Terminal":
        return "#4a235a"
    gen_idx = max(0, min(int(generation if not pd.isna(generation) else 0) - 1, len(GEN_RAMP) - 1))
    return GEN_RAMP[gen_idx]


# ── Data paths ────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
PASSPORT_PATH    = BASE_DIR / "vivc_passport_table.csv"
SLIM_PATH        = BASE_DIR / "data" / "vivc_passport_slim.csv"
MARKER_PATH      = BASE_DIR / "data" / "marker_support.csv"
VIVC_MARKER_PATH = BASE_DIR / "data" / "vivc_confirmed_marker_support.csv"
ALIAS_PATH       = BASE_DIR / "data" / "marker_name_aliases.csv"
SUPP_PATH        = BASE_DIR / "data" / "vivc_supplementary.csv"
ROOTSTOCK_PATH   = BASE_DIR / "data" / "vivc_rootstock_utilization_table.csv"
SUPP_META_PATH   = BASE_DIR / "data" / "vivc_supplementary_meta.json"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading alias map…")
def load_alias_map() -> dict[str, str]:
    if not ALIAS_PATH.exists():
        return {}
    df = pd.read_csv(ALIAS_PATH, dtype=str).fillna("")
    # pandas 2.1+ deprecated applymap; use map instead
    for col in df.columns:
        df[col] = df[col].map(lambda x: x.strip().upper() if isinstance(x, str) else x)
    alias_col = "alias"
    canon_col = "canonical_vivc_name"
    if alias_col not in df.columns or canon_col not in df.columns:
        return {}
    return dict(zip(df[alias_col], df[canon_col]))


@st.cache_data(show_spinner="Loading passport data…")
def load_passport() -> pd.DataFrame:
    """
    Load and normalise the VIVC passport table.
    Replicates read_vivc_passport() from the R app exactly.
    """
    # Try full passport first, then slim
    for path in (PASSPORT_PATH, SLIM_PATH):
        if path.exists():
            try:
                raw = pd.read_csv(path, dtype=str, low_memory=False)
                break
            except Exception:
                continue
    else:
        st.error("Could not find vivc_passport_table.csv or data/vivc_passport_slim.csv")
        return pd.DataFrame()

    # --- Header detection (scraping-artifact header embedded in first data row) ---
    if "prime_name" not in raw.columns and len(raw) >= 3:
        first_row = raw.iloc[0].fillna("").astype(str)
        non_empty = (first_row != "").sum()
        if non_empty >= 4:
            raw.columns = [
                c.strip().lower().replace(" ", "_").replace("-", "_")
                for c in first_row.tolist()
            ]
            raw = raw.iloc[2:].reset_index(drop=True)

    # Clean column names
    raw.columns = [
        c.strip().lower()
         .replace(" ", "_")
         .replace("-", "_")
         .replace("(", "")
         .replace(")", "")
         .replace("/", "_")
        for c in raw.columns
    ]

    # --- Column detection (flexible) ---
    def detect(patterns: list[str]) -> str | None:
        for pat in patterns:
            for col in raw.columns:
                if pat.lower() in col.lower() or col == pat.lower():
                    return col
        return None

    col_prime   = detect(["prime_name"])
    col_vivc    = detect(["variety_number_vivc", "vivc_no", "vivc"])
    col_color   = detect(["colour_of_berry_skin", "color_of_berry_skin", "berry_color", "berry_colour"])
    col_util    = detect(["utilization"])
    col_origin  = detect(["country_or_region_of_origin", "country_or_region", "origin"])
    col_species = detect(["species"])
    col_p1      = detect(["prime_name_of_parent_1", "parent_1", "parent1"])
    col_p2      = detect(["prime_name_of_parent_2", "parent_2", "parent2"])
    col_ped     = detect(["confirmed_pedigree_by_markers", "pedigree_confirmed", "pedigree"])
    col_breeder = detect(["breeder"])
    col_year    = detect(["year_of_crossing", "crossing_year", "year"])

    def col_or_na(col: str | None) -> pd.Series:
        return raw[col].astype(str) if col else pd.Series([""] * len(raw))

    df = pd.DataFrame({
        "prime_name":         col_or_na(col_prime),
        "vivc_no":            col_or_na(col_vivc),
        "berry_color":        col_or_na(col_color),
        "utilization":        col_or_na(col_util),
        "origin":             col_or_na(col_origin),
        "species":            col_or_na(col_species),
        "parent1":            col_or_na(col_p1),
        "parent2":            col_or_na(col_p2),
        "pedigree_confirmed": col_or_na(col_ped),
        "breeder":            col_or_na(col_breeder),
        "year_raw":           col_or_na(col_year),
    })

    # Normalise blanks / NA strings to actual NaN
    na_vals = {"", "NA", "N/A", "nan", "None", "none"}
    df = df.replace(na_vals, np.nan)

    # Strip whitespace on string columns
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Uppercase name columns
    for c in ("prime_name", "parent1", "parent2"):
        df[c] = df[c].str.upper()

    # Remove header artifact rows leaked into parent columns
    artifact = {"PRIME NAME OF PARENT 1", "PRIME NAME OF PARENT 2"}
    df.loc[df["parent1"].isin(artifact), "parent1"] = np.nan
    df.loc[df["parent2"].isin(artifact), "parent2"] = np.nan

    # Extract 4-digit year
    df["year_of_crossing"] = (
        df["year_raw"]
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.drop(columns=["year_raw"])

    # Remove artifact header rows
    df = df[df["prime_name"].notna() & (df["prime_name"] != "") & (df["prime_name"] != "PRIME NAME")]

    # Sanitise berry_color
    df.loc[
        df["berry_color"].notna() & (
            (df["berry_color"].str.len() > 15) |
            (df["berry_color"].str.upper() == "COLOR OF BERRY SKIN")
        ),
        "berry_color"
    ] = np.nan

    # --- Deduplicate by vivc_no (keep lowest), fallback to prime_name ---
    df["_vivc_int"] = pd.to_numeric(df["vivc_no"], errors="coerce")
    df_with_id = df[df["_vivc_int"].notna()].sort_values("_vivc_int")
    df_no_id   = df[df["_vivc_int"].isna()]
    df_with_id = df_with_id.drop_duplicates(subset=["vivc_no"], keep="first")
    df_no_id   = df_no_id.drop_duplicates(subset=["prime_name"], keep="first")
    df = pd.concat([df_with_id, df_no_id], ignore_index=True).drop(columns=["_vivc_int"])

    # --- Apply alias map to parent columns ---
    alias_map = load_alias_map()
    if alias_map:
        df["parent1"] = df["parent1"].map(lambda x: alias_map.get(x, x) if isinstance(x, str) else x)
        df["parent2"] = df["parent2"].map(lambda x: alias_map.get(x, x) if isinstance(x, str) else x)

    # Remove self-parents (data entry errors)
    df.loc[df["parent1"] == df["prime_name"], "parent1"] = np.nan
    df.loc[df["parent2"] == df["prime_name"], "parent2"] = np.nan

    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading marker support data…")
def load_marker_support() -> pd.DataFrame:
    """
    Load and combine marker_support.csv + vivc_confirmed_marker_support.csv.
    Deduplicates by (child_variety, parent_variety), keeping highest confidence.
    """
    dfs = []
    for path in (MARKER_PATH, VIVC_MARKER_PATH):
        if path.exists():
            try:
                d = pd.read_csv(path, dtype=str, low_memory=False)
                dfs.append(d)
            except Exception:
                pass

    if not dfs:
        return pd.DataFrame(columns=[
            "child_variety", "parent_variety", "parent_role",
            "evidence_type", "marker_type", "n_markers",
            "lod_score", "study_reference", "doi",
            "confirmed_year", "confidence_level", "notes"
        ])

    tbl = pd.concat(dfs, ignore_index=True)

    # Normalise
    for c in ("child_variety", "parent_variety"):
        if c in tbl.columns:
            tbl[c] = tbl[c].str.strip().str.upper()

    if "confidence_level" in tbl.columns:
        tbl["confidence_level"] = tbl["confidence_level"].str.strip().str.lower()

    # Apply alias map
    alias_map = load_alias_map()
    if alias_map:
        for c in ("child_variety", "parent_variety"):
            if c in tbl.columns:
                tbl[c] = tbl[c].map(lambda x: alias_map.get(x, x) if isinstance(x, str) else x)

    # Deduplicate — keep highest confidence per (child, parent) pair
    conf_rank = {c: i for i, c in enumerate(CONFIDENCE_ORDER)}
    tbl["_rank"] = tbl["confidence_level"].map(lambda x: conf_rank.get(str(x).lower(), 99))
    tbl = (
        tbl.sort_values("_rank")
           .drop_duplicates(subset=["child_variety", "parent_variety"], keep="first")
           .drop(columns=["_rank"])
           .reset_index(drop=True)
    )
    return tbl


@st.cache_data(show_spinner="Building children map…")
def build_children_map(_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Returns {parent_name: [child_names]} for BFS descendant traversal.
    Leading underscore prevents Streamlit from hashing the DataFrame argument.
    """
    children: dict[str, list[str]] = defaultdict(list)
    for _, row in _df.iterrows():
        child = row["prime_name"]
        for pcol in ("parent1", "parent2"):
            p = row.get(pcol)
            if isinstance(p, str) and p.strip():
                children[p.strip().upper()].append(child)
    return dict(children)


# ── Pedigree algorithms ───────────────────────────────────────────────────────

def find_ancestors(
    variety: str,
    df: pd.DataFrame,
    max_depth: int = 10,
) -> tuple[list[tuple[str, str, str, int]], set[str]]:
    """
    BFS upward through parent1/parent2 columns.
    Returns (edges_list, nodes_set).
    edges_list: list of (parent, child, parent_role, depth) tuples.
    """
    variety = variety.strip().upper()

    # Build O(1) lookup dicts
    p1_lkp: dict[str, str] = {}
    p2_lkp: dict[str, str] = {}
    for _, row in df.iterrows():
        nm = row["prime_name"]
        if isinstance(nm, str):
            p1 = row.get("parent1")
            p2 = row.get("parent2")
            p1_lkp[nm] = p1 if isinstance(p1, str) and p1.strip() else ""
            p2_lkp[nm] = p2 if isinstance(p2, str) and p2.strip() else ""

    edges: list[tuple[str, str, str, int]] = []
    nodes: set[str] = {variety}
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(variety, 0)])

    while queue:
        node, depth = queue.popleft()
        if node in visited or depth >= max_depth:
            continue
        visited.add(node)

        for role, lkp in (("Parent 1", p1_lkp), ("Parent 2", p2_lkp)):
            parent = lkp.get(node, "")
            if parent and parent not in ("?",):
                edges.append((parent, node, role, depth + 1))
                nodes.add(parent)
                if parent not in visited:
                    queue.append((parent, depth + 1))

    # Deduplicate edges
    seen_edges: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str, int]] = []
    for e in edges:
        key = (e[0], e[1])
        if key not in seen_edges:
            seen_edges.add(key)
            deduped.append(e)

    return deduped, nodes


def find_descendants(
    variety: str,
    df: pd.DataFrame,
    max_depth: int = 15,
) -> pd.DataFrame:
    """
    BFS downward to find all descendants.
    Returns a DataFrame with Generation, Variety, Year, Parent1, Parent2, Origin, etc.
    """
    variety = variety.strip().upper()
    df = df.copy()
    df["_name_up"] = df["prime_name"].str.upper()
    df["_p1_up"]   = df["parent1"].fillna("").str.upper()
    df["_p2_up"]   = df["parent2"].fillna("").str.upper()

    if variety not in df["_name_up"].values:
        return pd.DataFrame()

    frontier: set[str] = {variety}
    visited:  set[str] = set()
    result_rows: list[dict] = []

    for depth in range(1, max_depth + 1):
        mask = (
            df["_p1_up"].isin(frontier) | df["_p2_up"].isin(frontier)
        ) & ~df["_name_up"].isin(visited | frontier)

        children = df[mask]
        if children.empty:
            break

        for _, row in children.iterrows():
            result_rows.append({
                "Generation":        depth,
                "Variety":           row["prime_name"],
                "Year":              row.get("year_of_crossing"),
                "Parent 1":          row.get("parent1"),
                "Parent 2":          row.get("parent2"),
                "Origin":            row.get("origin"),
                "Species":           row.get("species"),
                "Breeder":           row.get("breeder"),
                "Pedigree confirmed": row.get("pedigree_confirmed"),
            })

        visited.update(frontier)
        frontier = set(children["_name_up"].tolist())

    return pd.DataFrame(result_rows)


def build_pedigree_graph(
    variety: str,
    df: pd.DataFrame,
    max_depth: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build pedigree nodes and edges DataFrames for a given variety.
    Returns (nodes_df, edges_df).
    nodes_df cols: name, vivc_no, berry_color, origin, species, parent1, parent2,
                   year_of_crossing, pedigree_confirmed, breeder, generation, node_type, x, y
    edges_df cols: from, to, parent_role, depth
    """
    variety = variety.strip().upper()
    raw_edges, all_nodes = find_ancestors(variety, df, max_depth=max_depth)

    if not raw_edges:
        # Single node — no ancestors
        row = df[df["prime_name"] == variety]
        if row.empty:
            node_row = {
                "name": variety, "vivc_no": None, "berry_color": None,
                "origin": None, "species": None, "parent1": None,
                "parent2": None, "year_of_crossing": None,
                "pedigree_confirmed": None, "breeder": None,
                "generation": 0, "node_type": "Target", "x": 0.0, "y": 2000.0,
            }
        else:
            r = row.iloc[0]
            yr = r.get("year_of_crossing")
            node_row = {
                "name": variety,
                "vivc_no":            r.get("vivc_no"),
                "berry_color":        r.get("berry_color"),
                "origin":             r.get("origin"),
                "species":            r.get("species"),
                "parent1":            r.get("parent1"),
                "parent2":            r.get("parent2"),
                "year_of_crossing":   yr,
                "pedigree_confirmed": r.get("pedigree_confirmed"),
                "breeder":            r.get("breeder"),
                "generation": 0,
                "node_type": "Target",
                "x": 0.0,
                "y": float(yr) if pd.notna(yr) else 2000.0,
            }
        return pd.DataFrame([node_row]), pd.DataFrame(columns=["from", "to", "parent_role", "depth"])

    # Build edges DataFrame
    edges_df = pd.DataFrame(raw_edges, columns=["from", "to", "parent_role", "depth"])

    # Build metadata lookup keyed by prime_name
    meta_cols = ["prime_name", "vivc_no", "berry_color", "origin", "species",
                 "parent1", "parent2", "year_of_crossing", "pedigree_confirmed", "breeder"]
    available = [c for c in meta_cols if c in df.columns]
    meta = df[available].drop_duplicates(subset=["prime_name"], keep="first").set_index("prime_name")

    # Compute generation via BFS from target
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row["from"], row["to"])

    generation: dict[str, int] = {variety: 0}
    q: deque[tuple[str, int]] = deque([(variety, 0)])
    visited_gen: set[str] = set()
    while q:
        node, depth = q.popleft()
        if node in visited_gen:
            continue
        visited_gen.add(node)
        for pred in G.predecessors(node):
            if pred not in generation:
                generation[pred] = depth + 1
                q.append((pred, depth + 1))

    # Build nodes DataFrame
    node_rows = []
    for name in all_nodes:
        gen = generation.get(name, 0)
        in_dataset = name in meta.index

        if in_dataset:
            r = meta.loc[name]
            row_data = {
                "name":               name,
                "vivc_no":            r.get("vivc_no")            if "vivc_no"            in r.index else None,
                "berry_color":        r.get("berry_color")        if "berry_color"        in r.index else None,
                "origin":             r.get("origin")             if "origin"             in r.index else None,
                "species":            r.get("species")            if "species"            in r.index else None,
                "parent1":            r.get("parent1")            if "parent1"            in r.index else None,
                "parent2":            r.get("parent2")            if "parent2"            in r.index else None,
                "year_of_crossing":   r.get("year_of_crossing")   if "year_of_crossing"   in r.index else None,
                "pedigree_confirmed": r.get("pedigree_confirmed") if "pedigree_confirmed" in r.index else None,
                "breeder":            r.get("breeder")            if "breeder"            in r.index else None,
                "generation": gen,
            }
        else:
            row_data = {
                "name": name, "vivc_no": None, "berry_color": None,
                "origin": None, "species": None, "parent1": None,
                "parent2": None, "year_of_crossing": None,
                "pedigree_confirmed": None, "breeder": None,
                "generation": gen,
            }

        # Node type
        is_target    = (name == variety)
        has_parents  = name in edges_df["to"].values
        has_children = name in edges_df["from"].values
        if is_target:
            row_data["node_type"] = "Target"
        elif has_parents and has_children:
            row_data["node_type"] = "Intermediate"
        elif not has_children:
            row_data["node_type"] = "Founder/Terminal"
        else:
            row_data["node_type"] = "Other"

        node_rows.append(row_data)

    nodes_df = pd.DataFrame(node_rows)

    # --- Infer years for nodes without a crossing year ---
    # Ancestors inferred as 20 years before earliest known child
    nodes_df["year_of_crossing"] = pd.to_numeric(nodes_df["year_of_crossing"], errors="coerce")
    year_map = nodes_df.set_index("name")["year_of_crossing"].to_dict()

    for _ in range(10):
        changed = False
        for _, erow in edges_df.iterrows():
            parent, child = erow["from"], erow["to"]
            child_yr = year_map.get(child)
            if pd.notna(child_yr) and pd.isna(year_map.get(parent)):
                year_map[parent] = child_yr - 20
                changed = True
        if not changed:
            break

    nodes_df["year_of_crossing"] = nodes_df["name"].map(year_map)

    max_yr = nodes_df["year_of_crossing"].dropna().max()
    if pd.isna(max_yr):
        max_yr = 2000.0

    nodes_df["year_plot"] = nodes_df.apply(
        lambda r: r["year_of_crossing"] if pd.notna(r["year_of_crossing"])
        else max_yr - r["generation"] * 20,
        axis=1,
    )

    # --- Pyramidal x-layout (Reingold-Tilford adapted for pedigree DAGs) ---
    # P1 always left of P2; shared ancestors averaged across branches.
    all_names = nodes_df["name"].tolist()
    gen_map   = nodes_df.set_index("name")["generation"].to_dict()
    max_gen   = max(gen_map.values()) if gen_map else 0
    MIN_SEP   = 1.4

    # Build parent lookup for layout
    parents_lu: dict[str, dict[str, str | None]] = defaultdict(lambda: {"Parent 1": None, "Parent 2": None})
    for _, erow in edges_df.iterrows():
        child  = erow["to"]
        parent = erow["from"]
        role   = erow["parent_role"]
        parents_lu[child][role] = parent

    # Subtree widths
    sw: dict[str, float] = {n: 1.0 for n in all_names}
    for gen_level in range(max_gen, -1, -1):
        for nd in [n for n in all_names if gen_map.get(n, 0) == gen_level]:
            p1 = parents_lu[nd]["Parent 1"]
            p2 = parents_lu[nd]["Parent 2"]
            pars = [p for p in (p1, p2) if p is not None]
            if not pars:
                sw[nd] = 1.0
            elif len(pars) == 1:
                sw[nd] = max(1.0, sw.get(pars[0], 1.0))
            else:
                sw[nd] = sum(sw.get(p, 1.0) for p in pars) + (len(pars) - 1) * MIN_SEP

    # Propose positions top-down
    x_props: dict[str, list[float]] = {n: [] for n in all_names}
    target_names = [n for n in all_names if gen_map.get(n, 0) == 0]
    if target_names:
        x_props[target_names[0]] = [0.0]

    for gen_level in range(0, max_gen):
        for nd in [n for n in all_names if gen_map.get(n, 0) == gen_level]:
            if not x_props[nd]:
                continue
            cx = float(np.mean(x_props[nd]))
            p1 = parents_lu[nd]["Parent 1"]
            p2 = parents_lu[nd]["Parent 2"]

            if p1 is not None and p2 is not None:
                tot = sw.get(p1, 1.0) + sw.get(p2, 1.0) + MIN_SEP
                x_props[p1].append(cx - tot / 2 + sw.get(p1, 1.0) / 2)
                x_props[p2].append(cx + tot / 2 - sw.get(p2, 1.0) / 2)
            elif p1 is not None:
                x_props[p1].append(cx - MIN_SEP * 0.45)
            elif p2 is not None:
                x_props[p2].append(cx + MIN_SEP * 0.45)

    x_pos: dict[str, float] = {}
    for nm in all_names:
        props = x_props[nm]
        x_pos[nm] = float(np.mean(props)) if props else float("nan")

    # Fill unassigned nodes
    assigned_vals = [v for v in x_pos.values() if not np.isnan(v)]
    right_edge = max(assigned_vals) + MIN_SEP if assigned_vals else 0.0
    for nm in all_names:
        if np.isnan(x_pos[nm]):
            x_pos[nm] = right_edge
            right_edge += MIN_SEP

    # Enforce minimum separation within each generation
    for gen_level in range(0, max_gen + 1):
        at_gen = [n for n in all_names if gen_map.get(n, 0) == gen_level]
        if len(at_gen) <= 1:
            continue
        at_gen_sorted = sorted(at_gen, key=lambda n: x_pos[n])
        for i in range(1, len(at_gen_sorted)):
            if x_pos[at_gen_sorted[i]] - x_pos[at_gen_sorted[i - 1]] < MIN_SEP:
                x_pos[at_gen_sorted[i]] = x_pos[at_gen_sorted[i - 1]] + MIN_SEP

    # Centre layout
    mean_x = np.mean(list(x_pos.values()))
    for nm in all_names:
        x_pos[nm] -= mean_x

    nodes_df["x"] = nodes_df["name"].map(x_pos)
    nodes_df["y"] = nodes_df["year_plot"]

    return nodes_df, edges_df


# ── Marker annotation ─────────────────────────────────────────────────────────

def annotate_edges(edges_df: pd.DataFrame, marker_tbl: pd.DataFrame) -> pd.DataFrame:
    """Left-join marker evidence onto edges (parent→child pairs)."""
    if edges_df.empty or marker_tbl.empty:
        edges_df = edges_df.copy()
        for c in ("confidence_level", "evidence_type", "study_reference", "doi",
                  "n_markers", "confirmed_year", "notes"):
            edges_df[c] = np.nan
        edges_df["confidence_level"] = "undocumented"
        return edges_df

    cols_want = ["child_variety", "parent_variety", "confidence_level",
                 "evidence_type", "study_reference", "doi", "n_markers",
                 "confirmed_year", "notes"]
    available = [c for c in cols_want if c in marker_tbl.columns]
    marker_sub = marker_tbl[available].copy()

    merged = edges_df.merge(
        marker_sub,
        left_on=["from", "to"],
        right_on=["parent_variety", "child_variety"],
        how="left",
    )
    merged.drop(columns=["parent_variety", "child_variety"], inplace=True, errors="ignore")
    merged["confidence_level"] = merged["confidence_level"].fillna("undocumented")
    return merged


def summarize_terminal_founders(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> dict:
    """
    Returns a dict with:
      founders      — rows where no incoming edges (no known parents in pedigree)
      origin_comp   — count/% by origin
      species_comp  — count/% by species
    """
    if nodes_df.empty or edges_df.empty:
        empty = pd.DataFrame(columns=["name", "origin", "species", "berry_color",
                                      "vivc_no", "generation", "indegree", "outdegree"])
        return {
            "founders":     empty,
            "origin_comp":  pd.DataFrame(columns=["origin", "n", "percent"]),
            "species_comp": pd.DataFrame(columns=["species", "n", "percent"]),
        }

    indeg  = edges_df.groupby("to").size().reset_index(name="indegree")
    outdeg = edges_df.groupby("from").size().reset_index(name="outdegree")

    founder_df = (
        nodes_df
        .merge(indeg,  left_on="name", right_on="to",   how="left")
        .merge(outdeg, left_on="name", right_on="from", how="left")
    )
    founder_df["indegree"]  = founder_df["indegree"].fillna(0).astype(int)
    founder_df["outdegree"] = founder_df["outdegree"].fillna(0).astype(int)
    founder_df = founder_df[founder_df["indegree"] == 0].copy()
    founder_df["origin"]  = founder_df["origin"].fillna("Unknown")
    founder_df["species"] = founder_df["species"].fillna("Unknown")

    n = len(founder_df)
    origin_comp = (
        founder_df.groupby("origin").size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    origin_comp["percent"] = (origin_comp["n"] / n * 100).round(1) if n > 0 else 0.0

    species_comp = (
        founder_df.groupby("species").size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )
    species_comp["percent"] = (species_comp["n"] / n * 100).round(1) if n > 0 else 0.0

    return {
        "founders":     founder_df,
        "origin_comp":  origin_comp,
        "species_comp": species_comp,
    }


# ── Node tooltip builder ──────────────────────────────────────────────────────

def _safe(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "na", "n/a") else "—"


def build_node_tooltip(row: pd.Series, edges_df: pd.DataFrame) -> str:
    name    = _safe(row.get("name"))
    vivc_no = row.get("vivc_no")
    origin  = _safe(row.get("origin"))
    species = _safe(row.get("species"))
    bcolor  = row.get("berry_color")
    year    = _safe(row.get("year_of_crossing"))
    gen     = _safe(row.get("generation"))
    breeder = _safe(row.get("breeder"))
    ped     = _safe(row.get("pedigree_confirmed"))

    vivc_link = (
        f"<a href='https://www.vivc.de/index.php?r=cultivarname%2Fview&id={vivc_no}' "
        f"target='_blank' style='color:#4a7c30;'>VIVC #{vivc_no} ↗</a>"
        if vivc_no and not (isinstance(vivc_no, float) and np.isnan(vivc_no))
        else "—"
    )

    swatch_color = berry_color_hex(bcolor)
    bc_swatch = (
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
        f"background:{swatch_color};border:1px solid #888;margin-right:4px;"
        f"vertical-align:middle;'></span>{_safe(bcolor)}"
    )

    # Evidence from annotated edges leading into this node
    ev_rows = ""
    if not edges_df.empty and "confidence_level" in edges_df.columns:
        child_edges = edges_df[
            (edges_df["to"] == row.get("name")) &
            (edges_df["confidence_level"] != "undocumented")
        ]
        if not child_edges.empty:
            conf_vals  = child_edges["confidence_level"].dropna().unique().tolist()
            conf_str   = ", ".join(conf_vals)
            conf_color = MARKER_COLORS.get(conf_vals[0], "#888")
            ev_rows += (
                f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Evidence</td>"
                f"<td><span style='background:{conf_color};color:#fff;padding:1px 5px;"
                f"border-radius:3px;font-size:10px;'>{conf_str}</span></td></tr>"
            )
            dois = child_edges["doi"].dropna().unique().tolist() if "doi" in child_edges.columns else []
            dois = [d for d in dois if d and str(d) not in ("nan", "None", "")]
            if dois:
                doi_links = " ".join(
                    f"<a href='https://doi.org/{d}' target='_blank' style='color:#4a7c30;'>"
                    f"{d} ↗</a>"
                    for d in dois[:2]
                )
                ev_rows += (
                    f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>DOI</td>"
                    f"<td style='font-size:11px;'>{doi_links}</td></tr>"
                )
            if "study_reference" in child_edges.columns:
                studies = child_edges["study_reference"].dropna().unique().tolist()
                studies = [s for s in studies if s and str(s) not in ("nan", "None", "")]
                if studies:
                    study_str = "; ".join(studies[:2])
                    ev_rows += (
                        f"<tr><td style='color:#777;padding:2px 10px 2px 0;vertical-align:top;'>Study</td>"
                        f"<td style='font-size:11px;max-width:180px;word-wrap:break-word;'>"
                        f"{study_str}</td></tr>"
                    )

    return (
        f"<div style='font-family:sans-serif;padding:6px 8px;min-width:210px;'>"
        f"<b style='font-size:14px;color:#1a1a1a;'>{name}</b>"
        f"<hr style='margin:5px 0;border:none;border-top:1px solid #e0e0e0;'/>"
        f"<table style='font-size:12px;border-collapse:collapse;width:100%;'>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>VIVC</td><td>{vivc_link}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Origin</td><td>{origin}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Species</td><td>{species}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Berry color</td><td>{bc_swatch}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Year</td><td>{year}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Generation</td><td>{gen}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Breeder</td><td>{breeder}</td></tr>"
        f"<tr><td style='color:#777;padding:2px 10px 2px 0;'>Pedigree confirmed</td><td>{ped}</td></tr>"
        f"{ev_rows}"
        f"</table></div>"
    )


# ── Coordinate conversion for pinned vis.js layout ───────────────────────────

def _dodge_year_band(
    names: list[str],
    x_vals: dict[str, float],
    y_vals: dict[str, float],
    min_gap: float = 110.0,
    axis: str = "x",
) -> dict[str, float]:
    """
    After mapping to pixel space, push nodes that share the same year-bucket
    apart along the spread axis so they don't overlap.
    Returns updated {name: coord} dict for the spread axis.
    """
    spread = dict(x_vals if axis == "x" else y_vals)
    buckets: dict[int, list[str]] = defaultdict(list)
    for nm in names:
        yr_bucket = int(round(y_vals[nm] if axis == "x" else x_vals[nm]))
        buckets[yr_bucket].append(nm)

    for bucket_nodes in buckets.values():
        if len(bucket_nodes) <= 1:
            continue
        bucket_nodes_sorted = sorted(bucket_nodes, key=lambda n: spread[n])
        orig_mid = np.mean([spread[n] for n in bucket_nodes_sorted])
        s = [spread[n] for n in bucket_nodes_sorted]
        for i in range(1, len(s)):
            if s[i] - s[i - 1] < min_gap:
                s[i] = s[i - 1] + min_gap
        # Re-centre
        new_mid = np.mean(s)
        for i, nm in enumerate(bucket_nodes_sorted):
            spread[nm] = s[i] - new_mid + orig_mid

    return spread


def prepare_layout_coords(
    nodes_df: pd.DataFrame,
    mode: str = "UD",
    target_span: float = 900.0,
) -> dict[str, tuple[float, float]]:
    """
    Convert the abstract pyramidal (x, y=year_plot) coordinates into pixel
    coordinates suitable for vis.js fixed positions.

    mode "UD" — vertical timeline: older at top, recent at bottom
    mode "LR" — horizontal timeline: older at left, recent at right
    """
    names   = nodes_df["name"].tolist()
    yr_vals = nodes_df.set_index("name")["y"].to_dict()     # year_plot
    x_tree  = nodes_df.set_index("name")["x"].to_dict()     # pyramidal x

    yr_arr  = np.array([yr_vals[n] for n in names])
    x_arr   = np.array([x_tree[n]  for n in names])

    yr_range = float(np.nanmax(yr_arr) - np.nanmin(yr_arr)) or 100.0
    x_range  = float(np.nanmax(x_arr)  - np.nanmin(x_arr))  or 1.0
    max_yr   = float(np.nanmax(yr_arr))

    yr_scale = target_span / yr_range
    x_scale  = target_span / x_range

    if mode == "LR":
        # Time flows left→right
        x_auto = {n: (yr_vals[n] - max_yr) * yr_scale for n in names}
        y_auto = {n:  x_tree[n]  * x_scale             for n in names}
        # Dodge along y (tree axis) for same-year nodes
        y_auto = _dodge_year_band(names, x_auto, y_auto, min_gap=110.0, axis="y")
    else:
        # Vertical (default): time flows top→bottom
        x_auto = {n:  x_tree[n]  * x_scale             for n in names}
        y_auto = {n: (yr_vals[n] - max_yr) * yr_scale   for n in names}
        # Dodge along x (tree axis) for same-year nodes
        x_auto = _dodge_year_band(names, x_auto, y_auto, min_gap=110.0, axis="x")

    return {n: (x_auto[n], y_auto[n]) for n in names}


# ── Network visualisation (pyvis) ─────────────────────────────────────────────

def build_pyvis_network(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    focal: str,
    color_mode: str = "generation",
    marker_overlay: bool = False,
    layout_mode: str = "UD",
    height: str = "650px",
) -> str:
    """
    Build a pyvis HTML string for the pedigree network.
    Uses vivc_no as internal node ID to avoid duplicate-name crashes.
    Node positions are pinned from the pyramidal layout algorithm.
    Returns raw HTML string.
    """
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#faf8f5",
        font_color="#333333",
    )

    # node_id → name mapping (use vivc_no if available, else name)
    def node_id(row_: pd.Series) -> str:
        vivc = row_.get("vivc_no")
        if vivc and not (isinstance(vivc, float) and np.isnan(vivc)):
            return f"vivc_{vivc}"
        return str(row_["name"])

    name_to_id: dict[str, str] = {}
    for _, row in nodes_df.iterrows():
        name_to_id[row["name"]] = node_id(row)

    # Compute pinned pixel coordinates from layout algorithm
    if not nodes_df.empty and "x" in nodes_df.columns and "y" in nodes_df.columns:
        coords = prepare_layout_coords(nodes_df, mode=layout_mode)
    else:
        coords = {row["name"]: (0.0, 0.0) for _, row in nodes_df.iterrows()}

    # Annotated edges for tooltip/color building
    ann = annotate_edges(edges_df, load_marker_support()) if not edges_df.empty else edges_df.copy()

    # --- Add nodes ---
    for _, row in nodes_df.iterrows():
        nid      = name_to_id[row["name"]]
        nm       = row["name"]
        is_focal = (nm == focal.upper())
        ntype    = row.get("node_type", "Other")
        gen      = row.get("generation", 0)

        # Color
        if color_mode == "berry":
            bg_color = berry_color_hex(row.get("berry_color"))
        else:
            bg_color = gen_color(ntype, gen)

        # Border
        border_color = "#7b1c2e" if is_focal else "#1a1a1a"
        if color_mode == "berry" and ntype == "Founder/Terminal":
            border_color = "#4a235a"

        # Shape
        if ntype == "Target":
            shape = "triangle"
        elif ntype == "Founder/Terminal":
            shape = "square"
        else:
            shape = "dot"

        # Size
        child_count = int(edges_df[edges_df["from"] == nm].shape[0]) if not edges_df.empty else 0
        if is_focal:
            size = 30
        else:
            size = max(12, min(28, 12 + child_count * 4))

        # Font size
        font_size = 20 if is_focal else (16 if gen == 1 else (14 if gen == 2 else 12))

        # Year label
        yr = row.get("year_of_crossing")
        label = nm if pd.isna(yr) else f"{nm}\n({int(yr)})"

        # Fixed position from layout
        px, py = coords.get(nm, (0.0, 0.0))

        tooltip = build_node_tooltip(row, ann)

        net.add_node(
            nid,
            label=label,
            title=tooltip,
            shape=shape,
            color={
                "background": bg_color,
                "border": border_color,
                "highlight": {"background": "#c8a44a", "border": "#9a7a2e"},
            },
            size=size,
            font={"size": font_size, "color": "#111111"},
            shadow={"enabled": True, "size": 4},
            x=px,
            y=py,
            fixed={"x": True, "y": True},
        )

    # --- Add edges ---
    edge_iter = ann.iterrows() if not ann.empty else edges_df.iterrows()
    for _, erow in edge_iter:
        src  = erow.get("from", "")
        dst  = erow.get("to", "")
        role = erow.get("parent_role", "")

        if src not in name_to_id or dst not in name_to_id:
            continue

        src_id = name_to_id[src]
        dst_id = name_to_id[dst]
        is_p2  = (role == "Parent 2")

        if marker_overlay and "confidence_level" in erow.index:
            conf       = str(erow.get("confidence_level") or "undocumented").lower()
            edge_color = MARKER_COLORS.get(conf, MARKER_COLORS["undocumented"])
            base_width = 1.5 if is_p2 else 2.5
            edge_width = base_width * MARKER_WIDTHS.get(conf, 1.0)
            glyph      = MARKER_GLYPHS.get(conf, "")

            doi   = erow.get("doi", "")
            study = erow.get("study_reference", "")
            n_mk  = erow.get("n_markers", "")
            yr_mk = erow.get("confirmed_year", "")
            doi_str   = _safe(doi)
            study_str = _safe(study)

            if doi and str(doi) not in ("", "nan", "None"):
                doi_str = (
                    f"<a href='https://doi.org/{doi}' target='_blank' "
                    f"style='color:#4a7c30;'>{doi} ↗</a>"
                )

            conf_color = MARKER_COLORS.get(conf, "#888")
            edge_title = (
                f"<div style='font-family:sans-serif;padding:6px 8px;min-width:220px;'>"
                f"<b>{role}</b><br/>"
                f"<span style='background:{conf_color};color:#fff;padding:1px 6px;"
                f"border-radius:3px;font-size:11px;'>{conf}</span><br/><br/>"
                f"<table style='font-size:12px;'>"
                f"<tr><td style='color:#777;padding:2px 8px 2px 0;'>Study</td><td>{study_str}</td></tr>"
                f"<tr><td style='color:#777;padding:2px 8px 2px 0;'>DOI</td><td>{doi_str}</td></tr>"
                f"<tr><td style='color:#777;padding:2px 8px 2px 0;'>Markers</td><td>{_safe(n_mk)}</td></tr>"
                f"<tr><td style='color:#777;padding:2px 8px 2px 0;'>Year</td><td>{_safe(yr_mk)}</td></tr>"
                f"</table></div>"
            )
        else:
            edge_color = "#b09a7a" if is_p2 else "#6b4226"
            edge_width = 1.5 if is_p2 else 2.5
            glyph      = ""
            edge_title = role

        net.add_edge(
            src_id, dst_id,
            title=edge_title,
            color=edge_color,
            width=edge_width,
            dashes=is_p2,
            arrows="to",
            label=glyph if glyph else None,
            font={
                "size": 11, "color": "#333333",
                "strokeWidth": 2, "strokeColor": "#ffffff",
                "align": "middle",
            },
        )

    # Physics: disabled since nodes are pinned; keep interactions enabled
    physics_options = """{
  "physics": {
    "enabled": false
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100,
    "navigationButtons": true,
    "keyboard": false,
    "zoomView": true,
    "dragNodes": true
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "diagonalCross",
      "roundness": 0.28
    },
    "font": {"size": 11, "strokeWidth": 2, "strokeColor": "#ffffff"}
  }
}"""
    net.set_options(physics_options)

    return net.generate_html()


# ── Timeline (Plotly) ─────────────────────────────────────────────────────────

def build_timeline_chart(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    focal: str,
    color_mode: str = "generation",
) -> go.Figure:
    """
    Plotly scatter chart: x = pedigree layout x, y = year_plot.
    Edges drawn as lines behind nodes.
    """
    if nodes_df.empty:
        return go.Figure()

    fig = go.Figure()

    # Draw edges first (as lines)
    if not edges_df.empty:
        node_pos = nodes_df.set_index("name")[["x", "y"]].to_dict("index")
        for _, erow in edges_df.iterrows():
            src, dst = erow["from"], erow["to"]
            if src in node_pos and dst in node_pos:
                x0, y0 = node_pos[src]["x"], node_pos[src]["y"]
                x1, y1 = node_pos[dst]["x"], node_pos[dst]["y"]
                is_p2  = erow.get("parent_role") == "Parent 2"
                color  = "#b09a7a" if is_p2 else "#6b4226"
                dash   = "dot" if is_p2 else "solid"
                fig.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode="lines",
                    line={"color": color, "width": 1.5, "dash": dash},
                    showlegend=False,
                    hoverinfo="skip",
                ))

    # Draw nodes
    for _, row in nodes_df.iterrows():
        nm    = row["name"]
        yr    = row.get("y", row.get("year_plot", 2000))
        x_val = row.get("x", 0.0)
        ntype = row.get("node_type", "Other")
        gen   = row.get("generation", 0)
        bcolor = row.get("berry_color")

        if color_mode == "berry":
            color = berry_color_hex(bcolor)
        else:
            color = gen_color(ntype, gen)

        symbol = "triangle-up" if ntype == "Target" else (
            "square" if ntype == "Founder/Terminal" else "circle"
        )
        size = 16 if nm == focal.upper() else 8

        hover = (
            f"<b>{nm}</b><br>"
            f"Year: {_safe(row.get('year_of_crossing'))}<br>"
            f"Origin: {_safe(row.get('origin'))}<br>"
            f"Generation: {gen}"
        )

        fig.add_trace(go.Scatter(
            x=[x_val], y=[yr],
            mode="markers+text",
            marker={"color": color, "size": size, "symbol": symbol,
                    "line": {"color": "#333", "width": 1}},
            text=[nm],
            textposition="top center",
            textfont={"size": 9 if nm != focal.upper() else 12,
                      "color": "#1a1a1a"},
            hovertemplate=hover + "<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(
        title={
            "text": f"Pedigree Timeline — {focal.upper()}",
            "font": {"color": "#7b1c2e", "size": 18},
        },
        xaxis={"visible": False},
        yaxis={
            "title": "Year of crossing",
            "tickfont": {"size": 10, "color": "#888"},
            "gridcolor": "#e8e0d0",
            "gridwidth": 0.5,
            "autorange": "reversed",  # older at top
        },
        plot_bgcolor="#faf7f0",
        paper_bgcolor="#faf7f0",
        height=700,
        margin={"l": 60, "r": 20, "t": 60, "b": 30},
        hovermode="closest",
    )
    return fig


# ── Data summary ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing data summary…")
def build_data_summary(_df: pd.DataFrame) -> dict:
    df = _df.copy()
    n = len(df)

    df["has_parent1"]      = df["parent1"].notna()
    df["has_parent2"]      = df["parent2"].notna()
    df["has_any_parent"]   = df["has_parent1"] | df["has_parent2"]
    df["has_both_parents"] = df["has_parent1"] & df["has_parent2"]
    df["has_year"]         = df["year_of_crossing"].notna()
    df["has_origin"]       = df["origin"].notna()
    df["has_species"]      = df["species"].notna()
    df["has_breeder"]      = df["breeder"].notna()
    df["has_ped_conf"]     = df["pedigree_confirmed"].notna()
    df["is_rootstock"]     = df["utilization"].fillna("").str.lower().str.contains("rootstock")
    df["complete_core"]    = df["has_both_parents"] & df["has_year"] & df["has_origin"] & df["has_species"]

    # Duplicate names (same prime_name, different vivc_no)
    dup_names = set(df["prime_name"][df["prime_name"].duplicated(keep=False)].tolist())
    n_dup  = int(df["prime_name"].isin(dup_names).sum())
    n_self = int(
        (df["parent1"] == df["prime_name"]).sum() +
        (df["parent2"] == df["prime_name"]).sum()
    )

    summary = pd.DataFrame({
        "Metric": [
            "Total accessions", "Rootstock-labeled", "Any lineage defined",
            "Both parents defined", "Year of crossing defined", "Origin defined",
            "Species defined", "Breeder defined", "Pedigree confirmation defined",
            "Complete core entries",
            "— Data quality —",
            "Duplicate prime names (same name, different VIVC #)",
            "Self-parent data errors",
        ],
        "Count": [
            n,
            int(df["is_rootstock"].sum()),
            int(df["has_any_parent"].sum()),
            int(df["has_both_parents"].sum()),
            int(df["has_year"].sum()),
            int(df["has_origin"].sum()),
            int(df["has_species"].sum()),
            int(df["has_breeder"].sum()),
            int(df["has_ped_conf"].sum()),
            int(df["complete_core"].sum()),
            None,
            n_dup,
            n_self,
        ],
    })
    summary["% of total"] = summary["Count"].apply(
        lambda c: f"{round(100 * c / n, 1)}%" if pd.notna(c) else ""
    )

    top_origins = (
        df["origin"].fillna("Unknown")
        .value_counts()
        .head(20)
        .reset_index()
    )
    # pandas value_counts().reset_index() changed column names in pandas 2.x
    if top_origins.columns.tolist() == ["origin", "count"]:
        top_origins.columns = ["Origin", "Count"]
    elif top_origins.columns.tolist() == ["index", "origin"]:
        top_origins.columns = ["Origin", "Count"]
    else:
        top_origins.columns = ["Origin", "Count"]
    top_origins["%"] = (top_origins["Count"] / n * 100).round(1).astype(str) + "%"

    top_species = (
        df["species"].fillna("Unknown")
        .value_counts()
        .head(20)
        .reset_index()
    )
    if top_species.columns.tolist() == ["species", "count"]:
        top_species.columns = ["Species", "Count"]
    else:
        top_species.columns = ["Species", "Count"]
    top_species["%"] = (top_species["Count"] / n * 100).round(1).astype(str) + "%"

    parentage_dist = pd.DataFrame({
        "Parentage state": ["No parents", "One parent only", "Both parents"],
        "Count": [
            int((~df["has_parent1"] & ~df["has_parent2"]).sum()),
            int((df["has_parent1"] ^ df["has_parent2"]).sum()),
            int(df["has_both_parents"].sum()),
        ],
    })
    parentage_dist["% of total"] = (parentage_dist["Count"] / n * 100).round(1).astype(str) + "%"

    yr_valid = df["year_of_crossing"].dropna()
    yr_valid = yr_valid[(yr_valid >= 1700) & (yr_valid <= 2030)]

    return {
        "n":              n,
        "summary":        summary,
        "top_origins":    top_origins,
        "top_species":    top_species,
        "parentage_dist": parentage_dist,
        "years":          yr_valid,
        "df":             df,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, accent: str = "#4a7c30") -> None:
    st.markdown(
        f"""<div style="background:#fff;padding:12px 16px;border-radius:6px;
        border-left:4px solid {accent};box-shadow:0 1px 4px rgba(0,0,0,0.08);
        min-height:80px;box-sizing:border-box;">
        <div style="font-size:10px;color:#888;text-transform:uppercase;
                    letter-spacing:.07em;">{label}</div>
        <div style="font-size:22px;font-weight:700;color:#1a1a1a;
                    margin-top:5px;">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{round(100 * n / total, 1)}%"


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # App header
    st.markdown(
        """<div style="display:flex;align-items:baseline;gap:14px;padding:4px 0 8px 0;">
        <span style="font-size:26px;font-weight:800;color:#7b1c2e;letter-spacing:-.01em;">
        🍇 VIVC Pedigree Explorer</span>
        <span style="font-size:12px;color:#9c8060;letter-spacing:.04em;">
        Vitis International Variety Catalogue</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- Load data ---
    with st.spinner("Loading VIVC passport data…"):
        df = load_passport()

    if df.empty:
        st.error(
            "No passport data found. Place `vivc_passport_table.csv` in the app directory "
            "or `vivc_passport_slim.csv` in `./data/`."
        )
        return

    marker_tbl   = load_marker_support()
    all_names    = sorted(df["prime_name"].dropna().unique().tolist())
    children_map = build_children_map(df)

    # Varieties that have offspring (for grouping in descendant tab)
    names_with_kids = sorted([n for n in all_names if n in children_map and children_map[n]])

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sidebar-heading">Accession</div>', unsafe_allow_html=True)

        default_variety = (
            "MARQUETTE" if "MARQUETTE" in all_names
            else (all_names[0] if all_names else "")
        )
        selected_variety = st.selectbox(
            "Search varieties…",
            options=all_names,
            index=all_names.index(default_variety) if default_variety in all_names else 0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown('<div class="sidebar-heading">Pedigree Depth</div>', unsafe_allow_html=True)
        max_depth       = st.slider("Maximum depth", min_value=1, max_value=20, value=8, step=1)
        gen_limit       = st.slider("Timeline: generation limit", min_value=2, max_value=15, value=5, step=1)
        use_full_depth  = st.checkbox("Networks use full depth", value=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-heading">Network Display</div>', unsafe_allow_html=True)
        color_mode = st.selectbox(
            "Colour nodes by",
            options=["generation", "berry"],
            format_func=lambda x: "Generation depth" if x == "generation" else "Berry skin colour",
        )
        layout_mode = st.selectbox(
            "Timeline direction",
            options=["UD", "LR"],
            format_func=lambda x: "Top-down (oldest at top)" if x == "UD" else "Left-right (oldest at left)",
        )
        st.markdown("---")
        st.markdown('<div class="sidebar-heading">Molecular Evidence</div>', unsafe_allow_html=True)
        show_markers = st.checkbox("Show marker support overlay", value=False)

        if show_markers:
            st.markdown(
                "<div style='font-size:11px;line-height:1.9;margin-top:4px;'>"
                + "".join(
                    f"<span style='display:inline-block;width:12px;height:4px;background:{c};"
                    f"border-radius:2px;margin-right:6px;vertical-align:middle;'></span>{lbl}<br>"
                    for lbl, c in [
                        ("Confirmed",    MARKER_COLORS["confirmed"]),
                        ("Probable",     MARKER_COLORS["probable"]),
                        ("Disputed",     MARKER_COLORS["disputed"]),
                        ("Refuted",      MARKER_COLORS["refuted"]),
                        ("Undocumented", MARKER_COLORS["undocumented"]),
                    ]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown('<div class="sidebar-heading">Founder Composition</div>', unsafe_allow_html=True)
        show_unknown_origins = st.checkbox("Include 'Unknown' origins", value=True)
        show_unknown_species = st.checkbox("Include 'Unknown' species", value=True)

        st.markdown("---")
        st.markdown('<div class="sidebar-heading">Download</div>', unsafe_allow_html=True)
        download_section = st.empty()

    # ── Compute pedigree ──────────────────────────────────────────────────────
    with st.spinner(f"Building pedigree for {selected_variety}…"):
        nodes_df, edges_df = build_pedigree_graph(
            selected_variety, df, max_depth=max_depth
        )

    # Depth-limited version for network tabs
    if use_full_depth:
        net_nodes = nodes_df
        net_edges = edges_df
    else:
        net_nodes = nodes_df[nodes_df["generation"] <= gen_limit].copy()
        keep_names = set(net_nodes["name"])
        net_edges = edges_df[
            edges_df["from"].isin(keep_names) & edges_df["to"].isin(keep_names)
        ].copy()

    ann_edges = annotate_edges(net_edges, marker_tbl)

    # --- Download handlers in sidebar ---
    with download_section:
        if not nodes_df.empty:
            st.download_button(
                "📥 Nodes CSV",
                data=nodes_df.to_csv(index=False).encode(),
                file_name=f"{selected_variety}_nodes.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if not edges_df.empty:
            st.download_button(
                "📥 Edges CSV",
                data=ann_edges.to_csv(index=False).encode(),
                file_name=f"{selected_variety}_edges.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── Summary metrics row ───────────────────────────────────────────────────
    summary_data = build_data_summary(df)
    n_total     = summary_data["n"]
    n_parents   = int(summary_data["df"]["has_any_parent"].sum())
    n_offspring = len(names_with_kids)
    n_confirmed = int(marker_tbl["child_variety"].nunique()) if not marker_tbl.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Total Varieties", f"{n_total:,}", "#7b1c2e")
    with col2:
        metric_card("With Parent Data", f"{n_parents:,} ({pct(n_parents, n_total)})", "#4a7c30")
    with col3:
        metric_card("With Offspring", f"{n_offspring:,} ({pct(n_offspring, n_total)})", "#4a235a")
    with col4:
        metric_card("Marker-Confirmed", f"{n_confirmed:,}", "#7a6a20")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🔍 Pedigree Explorer",
        "🌿 Descendants",
        "📅 Timeline",
        "🌳 Founder Composition",
        "🪴 Rootstock",
        "📋 Raw Data",
        "📊 Data Summary",
        "📚 Resources",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1 — Pedigree Explorer
    # ────────────────────────────────────────────────────────────────────────
    with tab1:
        if nodes_df.empty:
            st.warning(f"No pedigree data found for **{selected_variety}**.")
        else:
            n_nodes = len(net_nodes)
            n_edges = len(net_edges)
            st.markdown(
                f"**{selected_variety}** — "
                f"{n_nodes} nodes · {n_edges} edges",
            )

            with st.spinner("Rendering network…"):
                html_content = build_pyvis_network(
                    net_nodes,
                    net_edges,
                    focal=selected_variety,
                    color_mode=color_mode,
                    marker_overlay=show_markers,
                    layout_mode=layout_mode,
                    height="650px",
                )
            components.html(html_content, height=680, scrolling=False)

            # Legend expanders
            col_a, col_b = st.columns(2)
            with col_a:
                with st.expander("Node Legend"):
                    st.markdown("""
| Symbol | Meaning |
|--------|---------|
| ▲ **Triangle** | Target (searched) variety |
| ■ **Square** | Founder / Terminal (no known parents) |
| ● **Circle** | Intermediate ancestor |

**Generation colors** (mode: generation depth)

<div style='display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;'>
<span style='background:#7b1c2e;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;'>Target</span>
<span style='background:#4a235a;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;'>Founder</span>
<span style='background:#2d5a1b;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;'>Gen 1</span>
<span style='background:#6ea64f;color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;'>Gen 2</span>
<span style='background:#c8d9a0;color:#333;padding:2px 8px;border-radius:3px;font-size:11px;'>Gen 4+</span>
</div>
""", unsafe_allow_html=True)

            with col_b:
                with st.expander("Edge Legend"):
                    if show_markers:
                        st.markdown("""
**Edge color = marker confidence level:**

<div style='font-size:13px;line-height:2;'>
<span style='display:inline-block;width:32px;height:4px;background:#1e9645;border-radius:2px;margin-right:8px;vertical-align:middle;'></span>Confirmed (molecular)<br>
<span style='display:inline-block;width:32px;height:4px;background:#d4920c;border-radius:2px;margin-right:8px;vertical-align:middle;'></span>Probable (single study)<br>
<span style='display:inline-block;width:32px;height:4px;background:#c45d1a;border-radius:2px;margin-right:8px;vertical-align:middle;'></span>Disputed<br>
<span style='display:inline-block;width:32px;height:4px;background:#cc2222;border-radius:2px;margin-right:8px;vertical-align:middle;'></span>Refuted<br>
<span style='display:inline-block;width:32px;height:4px;background:#c0bab2;border-radius:2px;margin-right:8px;vertical-align:middle;'></span>Undocumented<br>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.markdown("""
**Solid line** = Parent 1 (seed parent)

**Dashed line** = Parent 2 (pollen parent)

*Hover over nodes and edges for details.*
""")

            # Pedigree node table
            with st.expander("Pedigree node table", expanded=False):
                display_cols = [c for c in [
                    "name", "generation", "node_type", "vivc_no",
                    "origin", "species", "berry_color", "year_of_crossing",
                    "parent1", "parent2", "breeder", "pedigree_confirmed",
                ] if c in net_nodes.columns]
                st.dataframe(
                    net_nodes[display_cols].sort_values("generation"),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2 — Descendants
    # ────────────────────────────────────────────────────────────────────────
    with tab2:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.markdown(
                "<div style='background:#fff;border-radius:8px;padding:18px;"
                "border-left:4px solid #7b1c2e;margin-bottom:16px;'>"
                "<p style='margin:0 0 12px 0;font-size:13px;color:#555;'>"
                "Find every variety that descends from a selected ancestor."
                "</p></div>",
                unsafe_allow_html=True,
            )

            # Grouped selectbox: has offspring first
            flat_opts = names_with_kids + [n for n in all_names if n not in names_with_kids]

            desc_variety = st.selectbox(
                "Ancestor variety",
                options=flat_opts,
                index=flat_opts.index(selected_variety) if selected_variety in flat_opts else 0,
                key="desc_variety_select",
            )
            desc_depth  = st.slider("Max generations", 1, 20, 10, key="desc_depth")
            search_btn  = st.button("🔍 Find descendants", type="primary")

            n_with_kids = len(names_with_kids)
            st.markdown(
                f"<div style='color:#888;font-size:12px;margin-top:8px;line-height:1.6;'>"
                f"<b style='color:#555;'>{n_with_kids:,} varieties</b> have recorded offspring.<br>"
                f"Try <b>PINOT NOIR</b>, <b>RIESLING WEISS</b>, or <b>HEUNISCH WEISS</b>."
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_right:
            if search_btn or "desc_results" in st.session_state:
                if search_btn:
                    with st.spinner(f"Searching descendants of {desc_variety}…"):
                        results = find_descendants(desc_variety, df, max_depth=desc_depth)
                    st.session_state["desc_results"]      = results
                    st.session_state["desc_variety_used"] = desc_variety
                else:
                    results = st.session_state.get("desc_results", pd.DataFrame())

                if results is None or (hasattr(results, "empty") and results.empty):
                    variety_used = st.session_state.get("desc_variety_used", desc_variety)
                    in_group     = variety_used in names_with_kids
                    st.info(
                        f"No descendants found for **{variety_used}**."
                        + ("" if in_group else " This variety has no offspring recorded in VIVC.")
                    )
                else:
                    n_total_desc = len(results)
                    max_gen_desc = int(results["Generation"].max())
                    st.markdown(
                        f"<div style='background:#f5f0e8;border-radius:6px;padding:10px 12px;"
                        f"font-size:12px;color:#555;margin-bottom:12px;'>"
                        f"<b style='color:#7b1c2e;font-size:14px;'>{n_total_desc:,}</b>"
                        f" descendants across <b>{max_gen_desc}</b> generation(s)."
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        results.sort_values("Generation"),
                        use_container_width=True,
                        hide_index=True,
                        height=500,
                    )
                    st.download_button(
                        "📥 Download results",
                        data=results.to_csv(index=False).encode(),
                        file_name=f"{desc_variety}_descendants.csv",
                        mime="text/csv",
                    )
            else:
                st.markdown(
                    "<div style='color:#aaa;font-size:13px;padding:40px;text-align:center;'>"
                    "Select a variety and click <b>Find descendants</b> to begin."
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3 — Timeline
    # ────────────────────────────────────────────────────────────────────────
    with tab3:
        st.markdown(
            f"**Timeline view** — ancestors of **{selected_variety}** "
            f"positioned by year of crossing."
        )

        if nodes_df.empty:
            st.info("No pedigree data to display.")
        else:
            tl_nodes = nodes_df[nodes_df["generation"] <= gen_limit].copy()
            tl_names = set(tl_nodes["name"])
            tl_edges = edges_df[
                edges_df["from"].isin(tl_names) & edges_df["to"].isin(tl_names)
            ].copy()

            tl_color_mode = st.radio(
                "Color nodes by",
                ["Generation depth", "Berry skin color"],
                horizontal=True,
                label_visibility="collapsed",
            )
            cm = "generation" if tl_color_mode == "Generation depth" else "berry"

            fig = build_timeline_chart(tl_nodes, tl_edges, selected_variety, color_mode=cm)
            st.plotly_chart(fig, use_container_width=True)

            # Berry colour legend
            if cm == "berry":
                legend_items = " ".join(
                    f"<span style='display:inline-flex;align-items:center;gap:5px;margin-right:12px;'>"
                    f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;"
                    f"background:{hex_c};border:1px solid #888;'></span>"
                    f"<span style='font-size:12px;color:#555;'>{lbl}</span></span>"
                    for lbl, hex_c in [
                        ("Blanc (white)", "#e8d96a"),
                        ("Noir (black)",  "#2d1b4e"),
                        ("Rouge (red)",   "#8b1a2e"),
                        ("Rosé",          "#d9728a"),
                        ("Gris (grey)",   "#9b8abf"),
                        ("Red-Violet",    "#6b2060"),
                        ("Unknown",       "#c8bfb0"),
                    ]
                )
                st.markdown(
                    f"<div style='padding:8px 0;'>{legend_items}</div>",
                    unsafe_allow_html=True,
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4 — Founder Composition
    # ────────────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("🌳 Founder Composition")
        st.markdown(
            f"Terminal founders (no known parents) in the pedigree of "
            f"**{selected_variety}** (depth {max_depth})."
        )

        if nodes_df.empty or edges_df.empty:
            st.info("Select a variety with a multi-generation pedigree to see founder composition.")
        else:
            founder_summary = summarize_terminal_founders(nodes_df, edges_df)
            founders_df     = founder_summary["founders"]
            origin_comp     = founder_summary["origin_comp"]
            species_comp    = founder_summary["species_comp"]

            n_founders = len(founders_df)
            st.markdown(
                f"<div style='background:#f5f0e8;border-radius:6px;padding:10px 12px;"
                f"font-size:12px;color:#555;margin-bottom:12px;'>"
                f"<b style='color:#7b1c2e;font-size:14px;'>{n_founders}</b>"
                f" terminal founders identified."
                f"</div>",
                unsafe_allow_html=True,
            )

            # Filter unknowns if requested
            if not show_unknown_origins:
                origin_comp = origin_comp[origin_comp["origin"] != "Unknown"]
            if not show_unknown_species:
                species_comp = species_comp[species_comp["species"] != "Unknown"]

            fc1, fc2 = st.columns(2)

            with fc1:
                st.markdown("**By Geographic Origin**")
                if not origin_comp.empty:
                    fig_orig = px.bar(
                        origin_comp.head(20),
                        x="n",
                        y="origin",
                        orientation="h",
                        color="n",
                        color_continuous_scale=["#f0eedd", "#2d5a1b"],
                        labels={"n": "Founders", "origin": ""},
                        hover_data={"percent": True, "n": True},
                    )
                    fig_orig.update_layout(
                        plot_bgcolor="#faf7f0",
                        paper_bgcolor="#faf7f0",
                        coloraxis_showscale=False,
                        height=max(300, min(600, 20 * len(origin_comp) + 80)),
                        margin={"l": 10, "r": 10, "t": 10, "b": 10},
                        yaxis={"autorange": "reversed"},
                    )
                    st.plotly_chart(fig_orig, use_container_width=True)
                else:
                    st.info("No origin data available for founders.")

            with fc2:
                st.markdown("**By Species**")
                if not species_comp.empty:
                    fig_spec = px.bar(
                        species_comp.head(20),
                        x="n",
                        y="species",
                        orientation="h",
                        color="n",
                        color_continuous_scale=["#f0eedd", "#4a235a"],
                        labels={"n": "Founders", "species": ""},
                        hover_data={"percent": True, "n": True},
                    )
                    fig_spec.update_layout(
                        plot_bgcolor="#faf7f0",
                        paper_bgcolor="#faf7f0",
                        coloraxis_showscale=False,
                        height=max(300, min(600, 20 * len(species_comp) + 80)),
                        margin={"l": 10, "r": 10, "t": 10, "b": 10},
                        yaxis={"autorange": "reversed"},
                    )
                    st.plotly_chart(fig_spec, use_container_width=True)
                else:
                    st.info("No species data available for founders.")

            # Founder table
            with st.expander("Founder details table", expanded=False):
                display_cols = [c for c in [
                    "name", "generation", "vivc_no", "origin", "species",
                    "berry_color", "year_of_crossing", "indegree", "outdegree",
                ] if c in founders_df.columns]
                st.dataframe(
                    founders_df[display_cols].sort_values("generation", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=350,
                )
                st.download_button(
                    "📥 Download founders CSV",
                    data=founders_df.to_csv(index=False).encode(),
                    file_name=f"{selected_variety}_founders.csv",
                    mime="text/csv",
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5 — Rootstock
    # ────────────────────────────────────────────────────────────────────────
    with tab5:
        st.subheader("🪴 Rootstock Varieties")
        if ROOTSTOCK_PATH.exists():
            rootstock_df = pd.read_csv(ROOTSTOCK_PATH, dtype=str, low_memory=False)
            st.markdown(f"**{len(rootstock_df):,}** rootstock entries loaded.")

            search_root = st.text_input("Filter by name…", key="root_search")
            if search_root:
                mask = rootstock_df.apply(
                    lambda col: col.astype(str).str.contains(search_root, case=False, na=False)
                ).any(axis=1)
                rootstock_df = rootstock_df[mask]

            st.dataframe(rootstock_df, use_container_width=True, hide_index=True, height=500)
            st.download_button(
                "📥 Download rootstock data",
                rootstock_df.to_csv(index=False).encode(),
                file_name="vivc_rootstock.csv",
                mime="text/csv",
            )
        else:
            # Fallback: filter passport by utilization containing "rootstock"
            rootstock_sub = df[
                df["utilization"].fillna("").str.lower().str.contains("rootstock")
            ].copy()
            if not rootstock_sub.empty:
                st.markdown(
                    f"Showing **{len(rootstock_sub):,}** rootstock-labeled accessions from the passport table "
                    f"(no dedicated rootstock table found at `{ROOTSTOCK_PATH}`)."
                )
                search_root2 = st.text_input("Filter by name…", key="root_search2")
                if search_root2:
                    mask2 = rootstock_sub["prime_name"].str.contains(search_root2, case=False, na=False)
                    rootstock_sub = rootstock_sub[mask2]
                st.dataframe(rootstock_sub, use_container_width=True, hide_index=True, height=500)
                st.download_button(
                    "📥 Download rootstock data",
                    rootstock_sub.to_csv(index=False).encode(),
                    file_name="vivc_rootstock_filtered.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No dedicated rootstock table found at "
                    f"`{ROOTSTOCK_PATH}`.\n\n"
                    "Place `vivc_rootstock_utilization_table.csv` in the `data/` directory "
                    "to enable this tab."
                )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 6 — Raw Data
    # ────────────────────────────────────────────────────────────────────────
    with tab6:
        st.subheader("📋 Full Passport Table")
        st.markdown(f"**{n_total:,}** varieties · {len(df.columns)} columns")

        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            filter_name = st.text_input("Filter by variety name…", key="raw_name")
        with fcol2:
            filter_origin = st.text_input("Filter by origin/country…", key="raw_origin")
        with fcol3:
            all_colors   = ["All"] + sorted(df["berry_color"].dropna().unique().tolist())
            filter_color = st.selectbox("Filter by berry color…", all_colors, key="raw_color")

        filtered = df.copy()
        if filter_name:
            filtered = filtered[
                filtered["prime_name"].str.contains(filter_name, case=False, na=False)
            ]
        if filter_origin:
            filtered = filtered[
                filtered["origin"].fillna("").str.contains(filter_origin, case=False, na=False)
            ]
        if filter_color != "All":
            filtered = filtered[filtered["berry_color"] == filter_color]

        st.markdown(f"*Showing {len(filtered):,} of {n_total:,} rows*")
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
            height=550,
        )
        st.download_button(
            "📥 Download filtered CSV",
            data=filtered.to_csv(index=False).encode(),
            file_name="vivc_passport_filtered.csv",
            mime="text/csv",
        )

    # ────────────────────────────────────────────────────────────────────────
    # TAB 7 — Data Summary
    # ────────────────────────────────────────────────────────────────────────
    with tab7:
        st.subheader("📊 Data Summary")

        sd  = summary_data
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Total Accessions", f"{sd['n']:,}", "#7b1c2e")
        with m2:
            n_both = int(sd["df"]["has_both_parents"].sum())
            metric_card("Both Parents Known", f"{n_both:,} ({pct(n_both, sd['n'])})", "#4a7c30")
        with m3:
            n_yr = int(sd["df"]["has_year"].sum())
            metric_card("Year of Crossing", f"{n_yr:,} ({pct(n_yr, sd['n'])})", "#4a235a")
        with m4:
            n_orig = int(sd["df"]["has_origin"].sum())
            metric_card("Origin Defined", f"{n_orig:,} ({pct(n_orig, sd['n'])})", "#a0522d")

        st.markdown("<br>", unsafe_allow_html=True)

        # Completeness table
        with st.expander("Completeness table", expanded=True):
            st.dataframe(sd["summary"], use_container_width=True, hide_index=True)

        col_charts1, col_charts2 = st.columns(2)

        with col_charts1:
            st.subheader("Top 20 Origins")
            top_o      = sd["top_origins"].head(20)
            count_col  = "Count" if "Count" in top_o.columns else top_o.columns[1]
            label_col  = "Origin" if "Origin" in top_o.columns else top_o.columns[0]
            fig_orig_s = px.bar(
                top_o.sort_values(count_col),
                x=count_col,
                y=label_col,
                orientation="h",
                color=count_col,
                color_continuous_scale=[
                    "#7b1c2e", "#a0522d", "#7a6a20", "#4a7c30",
                    "#2d5a1b", "#4a235a", "#6b3fa0"
                ],
                labels={count_col: "Varieties", label_col: ""},
            )
            fig_orig_s.update_layout(
                plot_bgcolor="#faf7f0",
                paper_bgcolor="#faf7f0",
                coloraxis_showscale=False,
                height=500,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig_orig_s, use_container_width=True)

        with col_charts2:
            st.subheader("Varieties by Berry Color")
            berry_counts = (
                df["berry_color"]
                .fillna("Unknown")
                .value_counts()
                .reset_index()
            )
            if berry_counts.columns.tolist() == ["berry_color", "count"]:
                berry_counts.columns = ["Berry Color", "Count"]
            else:
                berry_counts.columns = ["Berry Color", "Count"]
            berry_counts["hex"] = berry_counts["Berry Color"].map(
                lambda x: BERRY_COLOR_HEX.get(str(x).upper(), BERRY_COLOR_DEFAULT)
            )
            fig_berry = px.bar(
                berry_counts,
                x="Berry Color",
                y="Count",
                color="Berry Color",
                color_discrete_map={
                    row["Berry Color"]: row["hex"]
                    for _, row in berry_counts.iterrows()
                },
            )
            fig_berry.update_layout(
                plot_bgcolor="#faf7f0",
                paper_bgcolor="#faf7f0",
                showlegend=False,
                height=500,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig_berry, use_container_width=True)

        # Parentage distribution
        st.subheader("Parentage State Distribution")
        fig_par = px.pie(
            sd["parentage_dist"],
            names="Parentage state",
            values="Count",
            color_discrete_sequence=["#c8d9a0", "#4a7c30", "#2d5a1b"],
            hole=0.4,
        )
        fig_par.update_layout(
            plot_bgcolor="#faf7f0",
            paper_bgcolor="#faf7f0",
            height=350,
        )
        st.plotly_chart(fig_par, use_container_width=True)

        # Year histogram
        if not sd["years"].empty:
            st.subheader("Crossing Year Distribution")
            fig_yr = px.histogram(
                sd["years"].rename("Year of crossing"),
                x="Year of crossing",
                nbins=60,
                color_discrete_sequence=["#7b1c2e"],
            )
            fig_yr.update_layout(
                plot_bgcolor="#faf7f0",
                paper_bgcolor="#faf7f0",
                height=300,
                bargap=0.02,
            )
            st.plotly_chart(fig_yr, use_container_width=True)

        # Top species
        st.subheader("Top Species")
        st.dataframe(sd["top_species"], use_container_width=True, hide_index=True)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 8 — Resources
    # ────────────────────────────────────────────────────────────────────────
    with tab8:
        st.subheader("📚 Databases, Literature & Resources")

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(
                "<h4 style='color:#7b1c2e;font-weight:700;'>🌍 Databases & Online Resources</h4>",
                unsafe_allow_html=True,
            )

            def resource_card(title: str, url: str, url_label: str, description: str, accent: str = "#4a7c30") -> None:
                st.markdown(
                    f"<div style='background:#fff;border-radius:6px;padding:16px;"
                    f"border-left:4px solid {accent};margin-bottom:14px;'>"
                    f"<b>{title}</b>"
                    f"<p style='margin:4px 0;'>"
                    f"<a href='{url}' target='_blank' style='color:#4a7c30;'>{url_label}</a></p>"
                    f"<small style='color:#777;'>{description}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            resource_card(
                "VIVC — Vitis International Variety Catalogue",
                "https://www.vivc.de",
                "www.vivc.de",
                "Primary source for this app. Passport data, pedigrees, SSR markers, synonyms, "
                "accession holdings, and OIV ampelographic descriptors for ~27,000 grapevine accessions.",
            )
            resource_card(
                "GRIN — Germplasm Resources Information Network (USDA-ARS)",
                "https://www.ars-grin.gov/",
                "www.ars-grin.gov",
                "USDA national germplasm system. Includes Vitis collections, evaluation data, "
                "and links to NCGR-Davis and other repositories.",
            )
            resource_card(
                "VitisGen2 (Cornell-led USDA SCRI)",
                "https://www.vitisgen2.org",
                "www.vitisgen2.org",
                "Marker-assisted breeding consortium. SNP arrays, QTL maps for disease resistance "
                "(downy/powdery mildew, Botrytis) and berry quality.",
            )
            resource_card(
                "Vitis Genome Browser — INRAE GnpIS (12X v2 / PN40024)",
                "https://urgi.versailles.inrae.fr/Species/Grapevine",
                "INRAE GnpIS Grapevine",
                "Reference genome for Vitis vinifera (PN40024). Essential for interpreting "
                "SNP/SSR positions and linkage group coordinates.",
            )
            resource_card(
                "EnsemblPlants — Vitis vinifera",
                "https://plants.ensembl.org/Vitis_vinifera/",
                "plants.ensembl.org",
                "Gene models, comparative genomics, and variant data anchored to the 12X reference genome.",
                accent="#a0522d",
            )
            resource_card(
                "OIV — Organisation Internationale de la Vigne et du Vin",
                "https://www.oiv.int",
                "www.oiv.int",
                "Standardised ampelographic descriptor list (OIV-UPOV-IBPGR). ~100 morphological traits "
                "used throughout the VIVC catalogue.",
                accent="#a0522d",
            )

        with res_col2:
            st.markdown(
                "<h4 style='color:#7b1c2e;font-weight:700;'>📖 Key Literature</h4>",
                unsafe_allow_html=True,
            )

            def lit_card(title: str, citation: str, doi: str, accent: str = "#4a235a") -> None:
                st.markdown(
                    f"<div style='background:#fff;border-radius:6px;padding:16px;"
                    f"border-left:4px solid {accent};margin-bottom:14px;'>"
                    f"<b>{title}</b>"
                    f"<p style='margin:2px 0;font-size:13px;'>{citation}</p>"
                    f"<a href='https://doi.org/{doi}' target='_blank' "
                    f"style='color:#4a7c30;font-size:12px;'>doi:{doi}</a>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            lit_card(
                "Grapevine genome sequence",
                "Jaillon et al. (2007). The grapevine genome sequence suggests ancestral hexaploidization "
                "in major angiosperm phyla. <i>Nature</i> 449, 463–467.",
                "10.1038/nature06148",
            )
            lit_card(
                "Historical origins & genetic diversity",
                "This et al. (2006). Historical origins and genetic diversity of wine grapes. "
                "<i>Trends in Genetics</i> 22(9), 511–519.",
                "10.1016/j.tig.2006.07.008",
            )
            lit_card(
                "Genetic structure & domestication",
                "Myles et al. (2011). Genetic structure and domestication history of the grape. "
                "<i>PNAS</i> 108(9), 3530–3535.",
                "10.1073/pnas.1009363108",
            )
            lit_card(
                "Large diversity / complex breeding history",
                "Lacombe et al. (2013). A large diversity analysis of wine varieties reveals a complex "
                "breeding history. <i>J. Experimental Botany</i> 64(4), 1003–1017.",
                "10.1093/jxb/ers387",
            )
            lit_card(
                "SSR fingerprinting standard panel",
                "Laucou et al. (2011). High throughput analysis of grape genetic diversity as a tool "
                "for germplasm collection management. <i>Theor. Appl. Genet.</i> 122, 1233–1245.",
                "10.1007/s00122-010-1527-y",
                accent="#7a6a20",
            )
            lit_card(
                "Pedigree reconstruction & parentage verification",
                "Vouillamoz & Grando (2006). Genealogy of wine grape cultivars: 'Pinot' is related "
                "to 'Syrah'. <i>Heredity</i> 97, 102–110.",
                "10.1038/sj.hdy.6800842",
                accent="#7a6a20",
            )

        # Future enrichment opportunities
        st.markdown("---")
        st.markdown(
            "<div style='background:#fff;border-radius:6px;padding:16px;border-left:4px solid #5c3a1e;'>"
            "<h4 style='color:#7b1c2e;margin:0 0 10px 0;'>🧪 Future Data Enrichment Opportunities</h4>"
            "<p style='font-size:13px;color:#444;'>The following fields can be scraped from individual "
            "VIVC variety pages and integrated into this app:</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        enrich_cols = st.columns(3)
        with enrich_cols[0]:
            st.markdown("""
**Genetic / Molecular**
- SSR marker profiles (VVS2, VVMD5, VVMD7…)
- Chloroplast haplotype
- SNP genotypes from resequencing panels
- Parentage probability scores
""")
        with enrich_cols[1]:
            st.markdown("""
**Phenotypic / Ampelographic**
- Full OIV descriptor set (~100 morphological traits)
- Disease resistance ratings (PM, DM, Botrytis)
- Phenological windows (budburst → harvest)
- Berry quality traits (sugar, acid, anthocyanins)
""")
        with enrich_cols[2]:
            st.markdown("""
**Bibliographic / Registry**
- Synonyms (often 10–50+ per variety)
- Literature references cited by VIVC
- Plant Variety Protection / DUS registration
- Accession holdings by country / genebank
""")


if __name__ == "__main__":
    main()
