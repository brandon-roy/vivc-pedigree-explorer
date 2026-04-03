"""
build_master_accession_sheet.py
================================
Produces data/master_accession_sheet.csv — a single reference table that
anchors every variety name encountered across all data sources to its
canonical VIVC prime name and VIVC number.

Columns in output
-----------------
vivc_prime_name         — canonical VIVC prime name (uppercase)
vivc_no                 — VIVC accession number (integer where known)
vivc_origin             — country / region of origin from passport
vivc_species            — species string from passport
vivc_color              — berry colour from passport
vivc_parent1            — passport parent 1
vivc_parent2            — passport parent 2

# Presence flags (True/False) — is this accession referenced by this source?
in_passport             — appears in vivc_passport_table.csv
in_marker_support       — in marker_support.csv (curated landmark pairs)
in_vivc_confirmed       — in vivc_confirmed_marker_support.csv
in_constantini_2026     — in constantini_2026_marker_support.csv
in_myles_2011           — in myles_2011_marker_support.csv
in_lacombe_2013         — in lacombe_2013_marker_support.csv (parentage)
in_oliveira_2020        — in oliveira_2020_marker_support.csv
in_lac_emanuelli_2020   — in lacombe_emanuelli_2020_marker_support.csv
in_laucou_2018          — in laucou_2018_marker_support.csv
in_liang_2019           — in liang_2019_marker_support.csv
in_grin_pedigree        — in grin_pedigree_support.csv (GRIN SNP admixture)
in_grin_alignment       — has a matched GRIN accession (grin_vivc_alignment.csv)
in_ssr_vivc             — has SSR data in ssr_vivc.csv
in_ssr_migliaro         — has SSR data in ssr_migliaro_2019.csv
in_ssr_lacombe          — has SSR data in ssr_lacombe_2013.csv
in_ssr_oliveira         — has SSR data in ssr_oliveira_2020.csv
in_node_profile         — in node_molecular_profile.csv

n_sources               — total number of sources flagged True above

# Synonym columns — what name was used by each source (blank = not present)
name_in_marker_support
name_in_vivc_confirmed
name_in_constantini_2026
name_in_myles_2011
name_in_lacombe_2013
name_in_oliveira_2020
name_in_lac_emanuelli_2020
name_in_laucou_2018
name_in_liang_2019
name_in_grin_pedigree
name_in_grin_alignment      — GRIN plant name used in the alignment
grin_accession_id           — DVIT_*/PI_* identifier from GRIN
grin_match_methods          — how the GRIN→VIVC link was established

# Alias columns
is_alias                    — True if this vivc_prime_name has ≥1 alias in the alias map
aliases                     — semicolon-separated list of known aliases → this name
"""

from __future__ import annotations

import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

_NA = {"", "NA", "N/A", "n/a", "na", "N.A.", "NULL", "null", "None", "none", "nan"}


def _clean(s) -> str:
    """Strip and uppercase; return '' for nullish values."""
    if s is None:
        return ""
    s = str(s).strip()
    return "" if s in _NA or s.lower() in _NA else s.upper()


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Missing: {path.name}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=False)
        df = df.replace(dict.fromkeys(_NA, ""))
        return df
    except Exception as e:
        warnings.warn(f"Could not read {path.name}: {e}")
        return pd.DataFrame()


# ── 1. Load canonical passport ────────────────────────────────────────────────
print("Loading passport…")
passport_raw = _read(BASE / "vivc_passport_table.csv")

# Detect the artifact-header layout used by the scraped passport
if "prime_name" not in passport_raw.columns and len(passport_raw) >= 3:
    first_row = passport_raw.iloc[0].fillna("").astype(str)
    if (first_row != "").sum() >= 4:
        passport_raw.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_")
            for c in first_row.tolist()
        ]
        passport_raw = passport_raw.iloc[2:].reset_index(drop=True)

passport_raw.columns = [
    c.strip().lower().replace(" ", "_").replace("-", "_")
      .replace("(", "").replace(")", "").replace("/", "_")
    for c in passport_raw.columns
]

def _detect(patterns: list[str], cols) -> str | None:
    for pat in patterns:
        for col in cols:
            if pat.lower() in col.lower() or col == pat.lower():
                return col
    return None

cols = passport_raw.columns.tolist()
col_name   = _detect(["prime_name"], cols)
col_vivc   = _detect(["variety_number_vivc", "vivc_no", "vivc"], cols)
col_origin = _detect(["country_or_region_of_origin", "country_or_region", "origin"], cols)
col_species= _detect(["species"], cols)
col_color  = _detect(["colour_of_berry_skin", "color_of_berry_skin", "berry_color", "berry_colour"], cols)
col_p1     = _detect(["prime_name_of_parent_1", "parent_1", "parent1"], cols)
col_p2     = _detect(["prime_name_of_parent_2", "parent_2", "parent2"], cols)

def _col_or(c):
    return passport_raw[c].astype(str) if c else pd.Series([""] * len(passport_raw))

passport = pd.DataFrame({
    "vivc_prime_name": _col_or(col_name).str.strip().str.upper(),
    "vivc_no":         _col_or(col_vivc).str.strip(),
    "vivc_origin":     _col_or(col_origin).str.strip().str.upper(),
    "vivc_species":    _col_or(col_species).str.strip().str.upper(),
    "vivc_color":      _col_or(col_color).str.strip().str.upper(),
    "vivc_parent1":    _col_or(col_p1).str.strip().str.upper(),
    "vivc_parent2":    _col_or(col_p2).str.strip().str.upper(),
})
passport = passport.replace(dict.fromkeys(_NA | {"", "NAN"}, ""))

# Remove artifact rows
artifact = {"PRIME NAME OF PARENT 1", "PRIME NAME OF PARENT 2", "PRIME NAME", ""}
passport = passport[~passport["vivc_prime_name"].isin(artifact)]

# Convert vivc_no to integer string (drop ".0")
passport["vivc_no"] = (
    pd.to_numeric(passport["vivc_no"], errors="coerce")
      .dropna()
      .astype(int)
      .astype(str)
      .reindex(passport.index, fill_value="")
)

# Deduplicate: prefer rows with a vivc_no
passport = (
    passport.sort_values("vivc_no", ascending=False)
            .drop_duplicates(subset=["vivc_prime_name"], keep="first")
            .reset_index(drop=True)
)
print(f"  Passport: {len(passport):,} unique varieties")

# ── 2. Load alias map ─────────────────────────────────────────────────────────
print("Loading alias map…")
alias_df = _read(DATA / "marker_name_aliases.csv")
# alias → canonical_vivc_name
alias_to_canon: dict[str, str] = {}
canon_to_aliases: dict[str, list[str]] = defaultdict(list)

if not alias_df.empty and "alias" in alias_df.columns and "canonical_vivc_name" in alias_df.columns:
    for _, r in alias_df.iterrows():
        alias = _clean(r["alias"])
        canon = _clean(r["canonical_vivc_name"])
        if alias and canon:
            alias_to_canon[alias] = canon
            if alias != canon:
                canon_to_aliases[canon].append(alias)

print(f"  Alias map: {len(alias_to_canon):,} entries → {len(canon_to_aliases):,} canonical names with aliases")


def resolve(name: str) -> str:
    """Map a raw variety name to its canonical VIVC prime name via alias map."""
    up = _clean(name)
    return alias_to_canon.get(up, up)


# ── 3. Helpers to extract variety name sets from marker-support CSVs ──────────

def names_from_marker_csv(path: Path) -> tuple[set[str], dict[str, str]]:
    """
    Returns (canonical_names_set, {canonical_name: original_name_used}).
    Collects both child_variety and parent_variety columns.
    When multiple raw names resolve to the same canonical, keeps the first encountered.
    """
    df = _read(path)
    if df.empty:
        return set(), {}
    canon_set: set[str] = set()
    first_name: dict[str, str] = {}
    for col in ("child_variety", "parent_variety"):
        if col not in df.columns:
            continue
        for raw in df[col].dropna().unique():
            raw_clean = _clean(raw)
            if not raw_clean:
                continue
            canon = resolve(raw_clean)
            if canon and canon not in canon_set:
                canon_set.add(canon)
                first_name[canon] = raw_clean  # store pre-alias name as "synonym"
    return canon_set, first_name


# ── 4. Collect names from every data source ───────────────────────────────────
print("Scanning marker-support sources…")

sources = {
    "marker_support":       DATA / "marker_support.csv",
    "vivc_confirmed":       DATA / "vivc_confirmed_marker_support.csv",
    "constantini_2026":     DATA / "constantini_2026_marker_support.csv",
    "myles_2011":           DATA / "myles_2011_marker_support.csv",
    "lacombe_2013":         DATA / "lacombe_2013_marker_support.csv",
    "oliveira_2020":        DATA / "oliveira_2020_marker_support.csv",
    "lac_emanuelli_2020":   DATA / "lacombe_emanuelli_2020_marker_support.csv",
    "laucou_2018":          DATA / "laucou_2018_marker_support.csv",
    "liang_2019":           DATA / "liang_2019_marker_support.csv",
}

source_sets:  dict[str, set[str]]        = {}
source_names: dict[str, dict[str, str]]  = {}
for key, path in sources.items():
    s, n = names_from_marker_csv(path)
    source_sets[key]  = s
    source_names[key] = n
    print(f"  {key}: {len(s):,} unique canonical names")

# GRIN pedigree support (wide format — child + parent1 + parent2 columns)
print("  grin_pedigree…")
grin_ped = _read(DATA / "grin_pedigree_support.csv")
grin_ped_set: set[str] = set()
grin_ped_names: dict[str, str] = {}
if not grin_ped.empty:
    for col in ("child", "parent1", "parent2"):
        if col not in grin_ped.columns:
            continue
        for raw in grin_ped[col].dropna().unique():
            rc = _clean(raw)
            if rc:
                canon = resolve(rc)
                if canon and canon not in grin_ped_set:
                    grin_ped_set.add(canon)
                    grin_ped_names[canon] = rc
print(f"    grin_pedigree: {len(grin_ped_set):,} unique canonical names")

# GRIN alignment (DVIT/PI → VIVC mapping)
print("  grin_alignment…")
grin_aln = _read(DATA / "grin_vivc_alignment.csv")
grin_aln_set: set[str] = set()
grin_aln_plant_name: dict[str, str]       = {}  # canon → GRIN plant name
grin_aln_accession_id: dict[str, str]     = {}  # canon → DVIT_*/PI_*
grin_aln_match_methods: dict[str, str]    = {}  # canon → match_methods
if not grin_aln.empty:
    for _, row in grin_aln.iterrows():
        vivc_pn = _clean(row.get("vivc_prime_name", ""))
        if not vivc_pn:
            continue
        canon = resolve(vivc_pn)
        if canon:
            grin_aln_set.add(canon)
            grin_aln_plant_name[canon]    = str(row.get("GRIN_PLANT_NAME", "") or "").strip()
            grin_aln_accession_id[canon]  = str(row.get("GRIN_accessionID", "") or "").strip()
            grin_aln_match_methods[canon] = str(row.get("match_methods", "") or "").strip()
print(f"    grin_alignment: {len(grin_aln_set):,} unique canonical names")

# SSR data sources (variety_name column)
print("  SSR sources…")

def _ssr_set(path: Path, name_col: str = "variety_name") -> tuple[set[str], dict[str, str]]:
    df = _read(path)
    s: set[str] = set()
    n: dict[str, str] = {}
    if df.empty or name_col not in df.columns:
        return s, n
    for raw in df[name_col].dropna().unique():
        rc = _clean(raw)
        if rc:
            canon = resolve(rc)
            if canon and canon not in s:
                s.add(canon)
                n[canon] = rc
    return s, n

ssr_vivc_set,     _ = _ssr_set(DATA / "ssr_vivc.csv",          name_col="variety")
ssr_migliaro_set, _ = _ssr_set(DATA / "ssr_migliaro_2019.csv", name_col="variety_name")
ssr_lacombe_set,  _ = _ssr_set(DATA / "ssr_lacombe_2013.csv",  name_col="variety_name")
ssr_oliveira_set, _ = _ssr_set(DATA / "ssr_oliveira_2020.csv", name_col="variety_name")

print(f"    ssr_vivc: {len(ssr_vivc_set):,}, ssr_migliaro: {len(ssr_migliaro_set):,}, "
      f"ssr_lacombe: {len(ssr_lacombe_set):,}, ssr_oliveira: {len(ssr_oliveira_set):,}")

# Node molecular profile
node_prof = _read(DATA / "node_molecular_profile.csv")
node_prof_set: set[str] = set()
if not node_prof.empty and "vivc_prime_name" in node_prof.columns:
    for raw in node_prof["vivc_prime_name"].dropna().unique():
        rc = _clean(raw)
        if rc:
            node_prof_set.add(resolve(rc))
print(f"  node_molecular_profile: {len(node_prof_set):,}")


# ── 4b. Build fast passport lookup for partial-match resolution ───────────────
print("Building passport lookup index for synonym resolution…")
# Set of all passport prime names for O(1) lookup
passport_name_set: set[str] = set(passport["vivc_prime_name"])

# Index: each token of length ≥4 → list of passport prime names containing it
# Used to find "TEMPRANILLO TINTO" when we see "TEMPRANILLO"
from collections import defaultdict as _dd
_token_idx: dict[str, list[str]] = _dd(list)
for pn in passport_name_set:
    for tok in pn.split():
        if len(tok) >= 4:
            _token_idx[tok].append(pn)

def _find_passport_candidate(name: str) -> str:
    """
    For a name NOT in the passport, try to find the most likely VIVC prime name.
    Strategy:
      1. Exact substring: passport names that start with `name`
      2. Token overlap: passport names that share the most tokens ≥4 chars
    Returns the best candidate or '' if none found.
    """
    if not name:
        return ""
    # Strategy 1: starts-with
    starts = [pn for pn in passport_name_set if pn.startswith(name)]
    if len(starts) == 1:
        return starts[0]
    if len(starts) > 1:
        # Prefer exact prefix match that only adds a qualifier word
        shortest = min(starts, key=len)
        return shortest

    # Strategy 2: all tokens ≥4 chars overlap with passport names
    tokens = [t for t in name.split() if len(t) >= 4]
    if not tokens:
        return ""
    candidates: dict[str, int] = _dd(int)
    for tok in tokens:
        for pn in _token_idx.get(tok, []):
            candidates[pn] += 1
    if not candidates:
        return ""
    # Require ≥ 50% of name's tokens to match; pick highest overlap
    min_hits = max(1, len(tokens) // 2)
    qualified = {pn: cnt for pn, cnt in candidates.items() if cnt >= min_hits}
    if not qualified:
        return ""
    return max(qualified, key=qualified.get)


# ── 5. Build universe of all canonical names ──────────────────────────────────
print("Building universe…")
universe: set[str] = set()
universe.update(passport["vivc_prime_name"])
for s in source_sets.values():
    universe.update(s)
universe.update(grin_ped_set)
universe.update(grin_aln_set)
universe.update(ssr_vivc_set)
universe.update(ssr_migliaro_set)
universe.update(ssr_lacombe_set)
universe.update(ssr_oliveira_set)
universe.update(node_prof_set)
universe.discard("")
print(f"  Total unique canonical names: {len(universe):,}")


# ── 6. Assemble master table ──────────────────────────────────────────────────
print("Assembling master sheet…")

passport_idx = passport.set_index("vivc_prime_name")

rows = []
for name in sorted(universe):
    in_passport = name in passport_idx.index
    if in_passport:
        pr = passport_idx.loc[name]
        # Handle duplicate index (multiple passport rows for same name — shouldn't happen post-dedup)
        if isinstance(pr, pd.DataFrame):
            pr = pr.iloc[0]
        vivc_no      = pr.get("vivc_no",      "")
        vivc_origin  = pr.get("vivc_origin",  "")
        vivc_species = pr.get("vivc_species", "")
        vivc_color   = pr.get("vivc_color",   "")
        vivc_parent1 = pr.get("vivc_parent1", "")
        vivc_parent2 = pr.get("vivc_parent2", "")
    else:
        # Try to get vivc_no from GRIN alignment
        vivc_no      = ""
        vivc_origin  = ""
        vivc_species = ""
        vivc_color   = ""
        vivc_parent1 = ""
        vivc_parent2 = ""

    # Presence flags
    flags = {
        "in_passport":           in_passport,
        "in_marker_support":     name in source_sets["marker_support"],
        "in_vivc_confirmed":     name in source_sets["vivc_confirmed"],
        "in_constantini_2026":   name in source_sets["constantini_2026"],
        "in_myles_2011":         name in source_sets["myles_2011"],
        "in_lacombe_2013":       name in source_sets["lacombe_2013"],
        "in_oliveira_2020":      name in source_sets["oliveira_2020"],
        "in_lac_emanuelli_2020": name in source_sets["lac_emanuelli_2020"],
        "in_laucou_2018":        name in source_sets["laucou_2018"],
        "in_liang_2019":         name in source_sets["liang_2019"],
        "in_grin_pedigree":      name in grin_ped_set,
        "in_grin_alignment":     name in grin_aln_set,
        "in_ssr_vivc":           name in ssr_vivc_set,
        "in_ssr_migliaro":       name in ssr_migliaro_set,
        "in_ssr_lacombe":        name in ssr_lacombe_set,
        "in_ssr_oliveira":       name in ssr_oliveira_set,
        "in_node_profile":       name in node_prof_set,
    }
    n_sources = sum(flags.values())

    # Synonym / original-name columns
    def _orig(key, d):
        v = d.get(name, "")
        return v if v != name else ""   # blank when it equals the canonical (not a synonym)

    row = {
        "vivc_prime_name":          name,
        "vivc_no":                  vivc_no,
        "vivc_origin":              vivc_origin,
        "vivc_species":             vivc_species,
        "vivc_color":               vivc_color,
        "vivc_parent1":             vivc_parent1,
        "vivc_parent2":             vivc_parent2,
        **flags,
        "n_sources":                n_sources,
        "name_in_marker_support":       _orig("marker_support",     source_names["marker_support"]),
        "name_in_vivc_confirmed":       _orig("vivc_confirmed",     source_names["vivc_confirmed"]),
        "name_in_constantini_2026":     _orig("constantini_2026",   source_names["constantini_2026"]),
        "name_in_myles_2011":           _orig("myles_2011",         source_names["myles_2011"]),
        "name_in_lacombe_2013":         _orig("lacombe_2013",       source_names["lacombe_2013"]),
        "name_in_oliveira_2020":        _orig("oliveira_2020",      source_names["oliveira_2020"]),
        "name_in_lac_emanuelli_2020":   _orig("lac_emanuelli_2020", source_names["lac_emanuelli_2020"]),
        "name_in_laucou_2018":          _orig("laucou_2018",        source_names["laucou_2018"]),
        "name_in_liang_2019":           _orig("liang_2019",         source_names["liang_2019"]),
        "name_in_grin_pedigree":        _orig("grin_pedigree",      grin_ped_names),
        "name_in_grin_alignment":       grin_aln_plant_name.get(name, ""),
        "grin_accession_id":            grin_aln_accession_id.get(name, ""),
        "grin_match_methods":           grin_aln_match_methods.get(name, ""),
        "is_alias":                     name in canon_to_aliases,
        "aliases":                      "; ".join(sorted(canon_to_aliases.get(name, []))),
        # Resolution metadata
        "in_alias_map":                 name in alias_to_canon.values(),
        "passport_candidate":           "" if in_passport else _find_passport_candidate(name),
    }
    rows.append(row)

master = pd.DataFrame(rows)

# Sort: passport varieties first, then by number of sources desc, then alphabetically
master["_sort_key"] = (~master["in_passport"]).astype(int) * 1000 - master["n_sources"]
master = master.sort_values(["_sort_key", "vivc_prime_name"]).drop(columns=["_sort_key"]).reset_index(drop=True)

# ── 7. Write output ───────────────────────────────────────────────────────────
out_path = DATA / "master_accession_sheet.csv"
master.to_csv(out_path, index=False)
print(f"\nWrote {len(master):,} rows → {out_path}")

# ── 8. Summary stats ──────────────────────────────────────────────────────────
print("\n── Coverage summary ──────────────────────────────────────────────────")
for col in [c for c in master.columns if c.startswith("in_")]:
    n = master[col].sum()
    print(f"  {col:<30} {n:>6,}")

print(f"\n  Varieties in ≥2 sources:  {(master['n_sources'] >= 2).sum():,}")
print(f"  Varieties in ≥5 sources:  {(master['n_sources'] >= 5).sum():,}")
print(f"  Varieties NOT in passport: {(~master['in_passport']).sum():,}")
print(f"  Canonical names with aliases: {master['is_alias'].sum():,}")
print(f"  Total aliases tracked: {(master['aliases'] != '').sum():,}")

# Accessions in molecular sources but missing from passport
orphan_mol = master[
    (~master["in_passport"]) &
    (master[["in_ssr_vivc","in_ssr_migliaro","in_ssr_lacombe","in_ssr_oliveira","in_node_profile"]].any(axis=1))
]
if not orphan_mol.empty:
    print(f"\n  ⚠  Molecular data with no passport match: {len(orphan_mol):,}")
    print(orphan_mol[["vivc_prime_name","n_sources","passport_candidate"]].head(15).to_string(index=False))

# ── 9. Write unresolved synonyms report ──────────────────────────────────────
unresolved = master[~master["in_passport"]].copy()
unresolved = unresolved.sort_values(["n_sources", "vivc_prime_name"], ascending=[False, True])

unresolved_out = unresolved[[
    "vivc_prime_name", "n_sources", "passport_candidate",
    "in_marker_support", "in_vivc_confirmed", "in_constantini_2026",
    "in_myles_2011", "in_lacombe_2013", "in_oliveira_2020",
    "in_lac_emanuelli_2020", "in_laucou_2018", "in_liang_2019",
    "in_grin_pedigree", "in_grin_alignment",
    "in_ssr_vivc", "in_ssr_migliaro", "in_ssr_lacombe", "in_ssr_oliveira",
    "in_node_profile",
    "name_in_lacombe_2013", "name_in_laucou_2018", "name_in_lac_emanuelli_2020",
    "name_in_grin_pedigree", "name_in_grin_alignment",
    "grin_accession_id",
]].reset_index(drop=True)

unresolved_path = DATA / "unresolved_accession_synonyms.csv"
unresolved_out.to_csv(unresolved_path, index=False)
print(f"\nWrote {len(unresolved_out):,} unresolved rows → {unresolved_path}")

# Top candidates where passport_candidate was found
resolvable = unresolved_out[unresolved_out["passport_candidate"] != ""]
print(f"  Auto-candidate found for: {len(resolvable):,} of {len(unresolved_out):,} unresolved")
print("\nSample resolvable synonyms (add these to marker_name_aliases.csv):")
print(resolvable[["vivc_prime_name","passport_candidate","n_sources"]].head(25).to_string(index=False))
