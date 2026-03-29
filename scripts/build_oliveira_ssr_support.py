"""
build_oliveira_ssr_support.py
==============================
Extract SSR profiles for 410 Brazilian grapevine accessions from
de Oliveira et al. 2020 (PLOS ONE, doi: 10.1371/journal.pone.0240665)
Supplementary file pone.0240665.s001.xlsx.

Outputs
-------
1. data/ssr_oliveira_2020.csv   — raw SSR profiles with variety name + VIVC match
2. data/oliveira_2020_marker_support.csv — standard 12-column marker support rows
   for accessions that already have confirmed parent pairs in other sources.

Usage:
    python scripts/build_oliveira_ssr_support.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
XLSX_PATH  = (
    Path("/Users/brandonroy/Desktop/Informatique/Projects/VIVC App")
    / "Supplemental Literature Data"
    / "pone.0240665.s001.xlsx"
)
ALIAS_PATH   = BASE_DIR / "data" / "marker_name_aliases.csv"
SUPP_PATH    = BASE_DIR / "data" / "vivc_supplementary.csv"
MARKER_PATH  = BASE_DIR / "data" / "marker_support.csv"
VIVC_MKR_PATH = BASE_DIR / "data" / "vivc_confirmed_marker_support.csv"
LACOMBE_PATH  = BASE_DIR / "data" / "lacombe_2013_marker_support.csv"

OUT_PROFILES = BASE_DIR / "data" / "ssr_oliveira_2020.csv"
OUT_SUPPORT  = BASE_DIR / "data" / "oliveira_2020_marker_support.csv"

STUDY_REF  = "de Oliveira et al. 2020"
DOI        = "10.1371/journal.pone.0240665"
CONF_YEAR  = 2020
NOTES_BASE = "SSR profile: Brazilian grapevine germplasm collection (IAC)"

SCHEMA_COLS = [
    "child_variety", "parent_variety", "parent_role",
    "evidence_type", "marker_type", "n_markers",
    "lod_score", "study_reference", "doi",
    "confirmed_year", "confidence_level", "notes",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_alias_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "alias" not in df.columns or "canonical_vivc_name" not in df.columns:
        return {}
    return {
        str(a).strip().upper(): str(c).strip().upper()
        for a, c in zip(df["alias"], df["canonical_vivc_name"])
    }


def norm(name: str, alias_map: dict[str, str]) -> str:
    n = str(name).strip().upper()
    return alias_map.get(n, n)


def build_vivc_name_set(supp_path: Path, alias_map: dict[str, str]) -> set[str]:
    """Build a set of all known VIVC names (prime + synonyms) for matching."""
    names: set[str] = set()
    if not supp_path.exists():
        return names
    df = pd.read_csv(supp_path, dtype=str, low_memory=False)
    if "prime_name" in df.columns:
        for n in df["prime_name"].dropna():
            names.add(str(n).strip().upper())
    if "synonyms" in df.columns:
        for syns in df["synonyms"].dropna():
            for s in str(syns).split(";"):
                s = s.strip().upper()
                if s:
                    names.add(alias_map.get(s, s))
    return names


def load_confirmed_pairs(alias_map: dict[str, str]) -> set[str]:
    """
    Return a set of child variety names that have at least one confirmed/probable
    parent pair in any existing marker support source.
    """
    children: set[str] = set()
    for path in (MARKER_PATH, VIVC_MKR_PATH, LACOMBE_PATH):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=False)
            if "child_variety" in df.columns:
                for n in df["child_variety"].dropna():
                    children.add(norm(n, alias_map))
        except Exception:
            pass
    return children


def load_parent_pairs(alias_map: dict[str, str]) -> dict[str, list[tuple[str, str]]]:
    """
    Return {child_name: [(parent1, parent2), ...]} from all confirmed marker
    support sources.  Each pair (p1, p2) is in normalised form.
    """
    pairs: dict[str, list[tuple[str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()

    def _add(child, p1, p2):
        key = (child, p1, p2)
        if key in seen:
            return
        seen.add(key)
        pairs.setdefault(child, []).append((p1, p2))

    for path in (MARKER_PATH, VIVC_MKR_PATH, LACOMBE_PATH):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=False)
            for _, row in df.iterrows():
                child  = norm(str(row.get("child_variety", "")), alias_map)
                parent = norm(str(row.get("parent_variety", "")), alias_map)
                role   = str(row.get("parent_role", "")).strip()
                if not child or not parent:
                    continue
                # We pair rows: group by child then pair P1 + P2
                if "Parent 1" in role:
                    _add(child, parent, "")
                # We build full trios later; for now just store parent entries
        except Exception:
            pass

    # Build proper trio dict from all sources
    pair_dict: dict[str, list[tuple[str, str]]] = {}
    for path in (MARKER_PATH, VIVC_MKR_PATH, LACOMBE_PATH):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype=str, low_memory=False, keep_default_na=False)
            # Group by child_variety and collect parent pairs
            if "child_variety" not in df.columns or "parent_variety" not in df.columns:
                continue
            df["_child"] = df["child_variety"].apply(lambda x: norm(x, alias_map))
            df["_parent"] = df["parent_variety"].apply(lambda x: norm(x, alias_map))
            df["_role"] = df.get("parent_role", pd.Series([""] * len(df))).fillna("")

            for child, grp in df.groupby("_child"):
                p1_rows = grp[grp["_role"].str.contains("1")]
                p2_rows = grp[grp["_role"].str.contains("2")]
                p1_names = p1_rows["_parent"].tolist()
                p2_names = p2_rows["_parent"].tolist()
                # Zip into pairs
                for p1 in p1_names:
                    for p2 in p2_names:
                        trio_key = (child, p1, p2)
                        if trio_key not in seen:
                            seen.add(trio_key)
                            pair_dict.setdefault(child, []).append((p1, p2))
        except Exception:
            pass
    return pair_dict


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    if not XLSX_PATH.exists():
        print(f"ERROR: XLSX not found at {XLSX_PATH}", file=sys.stderr)
        sys.exit(1)

    alias_map = load_alias_map(ALIAS_PATH)
    vivc_names = build_vivc_name_set(SUPP_PATH, alias_map)
    parent_pairs = load_parent_pairs(alias_map)

    print(f"Reading: {XLSX_PATH}")
    df_raw = pd.read_excel(XLSX_PATH, header=1, dtype=str)

    # Identify key columns
    acc_col = "Accession name"
    if acc_col not in df_raw.columns:
        # Try col index 1
        acc_col = df_raw.columns[1]

    # Identify SSR loci columns: everything after column 11 (DAPC K=9 Assignation)
    # Columns are paired _1 / _2 per locus
    meta_cols = [c for c in df_raw.columns if not (c.endswith("_1") or c.endswith("_2"))]
    ssr_cols  = [c for c in df_raw.columns if c.endswith("_1") or c.endswith("_2")]

    # Extract unique locus names
    loci = sorted({c[:-2] for c in ssr_cols})

    print(f"  Accessions read: {len(df_raw)}")
    print(f"  SSR loci: {len(loci)} — {loci}")

    # ── Build profiles CSV ────────────────────────────────────────────────────
    profile_rows: list[dict] = []
    n_matched = 0

    for _, row in df_raw.iterrows():
        raw_name = str(row.get(acc_col, "")).strip()
        if not raw_name or raw_name.lower() in ("nan", ""):
            continue

        norm_name = norm(raw_name, alias_map)

        # Try to match to VIVC
        vivc_match = ""
        if norm_name in vivc_names:
            vivc_match = norm_name
        else:
            # Try alias map on the raw name
            candidate = alias_map.get(norm_name, "")
            if candidate and candidate in vivc_names:
                vivc_match = candidate

        if vivc_match:
            n_matched += 1

        # Count non-null loci (value > 0 and not -1)
        profile: dict = {"variety_name": norm_name, "vivc_match": vivc_match}
        for locus in loci:
            a1_col = f"{locus}_1"
            a2_col = f"{locus}_2"
            a1 = row.get(a1_col, "")
            a2 = row.get(a2_col, "")
            try:
                v1 = float(a1) if str(a1).strip() not in ("", "nan") else -1.0
            except ValueError:
                v1 = -1.0
            try:
                v2 = float(a2) if str(a2).strip() not in ("", "nan") else -1.0
            except ValueError:
                v2 = -1.0
            profile[locus] = f"{int(v1) if v1 > 0 else '?'}/{int(v2) if v2 > 0 else '?'}"

        profile_rows.append(profile)

    profiles_df = pd.DataFrame(profile_rows)

    OUT_PROFILES.parent.mkdir(parents=True, exist_ok=True)
    profiles_df.to_csv(OUT_PROFILES, index=False)
    print(f"\nSSR profiles saved: {len(profiles_df)} rows → {OUT_PROFILES}")
    print(f"  Matched to VIVC: {n_matched} / {len(profiles_df)}")

    # ── Build marker support rows ─────────────────────────────────────────────
    support_rows: list[dict] = []
    n_support_emitted = 0

    for _, row in df_raw.iterrows():
        raw_name = str(row.get(acc_col, "")).strip()
        if not raw_name or raw_name.lower() in ("nan", ""):
            continue

        norm_name = norm(raw_name, alias_map)
        vivc_match = norm_name if norm_name in vivc_names else alias_map.get(norm_name, "")

        if not vivc_match:
            continue

        # Count non-null SSR loci for this accession
        n_loci = 0
        for locus in loci:
            a1_col = f"{locus}_1"
            a1 = row.get(a1_col, "")
            try:
                v = float(a1)
                if v > 0:
                    n_loci += 1
            except (ValueError, TypeError):
                pass

        # Look up known parent pairs for this variety
        known_pairs = parent_pairs.get(vivc_match, [])
        if not known_pairs:
            continue

        for (p1, p2) in known_pairs:
            if not p1:
                continue
            notes = NOTES_BASE
            if n_loci > 0:
                notes += f"; {n_loci} loci typed"

            # Parent 1 row
            support_rows.append({
                "child_variety":   vivc_match,
                "parent_variety":  p1,
                "parent_role":     "Parent 1",
                "evidence_type":   "SSR",
                "marker_type":     "SSR",
                "n_markers":       n_loci if n_loci > 0 else np.nan,
                "lod_score":       np.nan,
                "study_reference": STUDY_REF,
                "doi":             DOI,
                "confirmed_year":  CONF_YEAR,
                "confidence_level": "probable",
                "notes":           notes,
            })
            n_support_emitted += 1

            # Parent 2 row
            if p2:
                support_rows.append({
                    "child_variety":   vivc_match,
                    "parent_variety":  p2,
                    "parent_role":     "Parent 2",
                    "evidence_type":   "SSR",
                    "marker_type":     "SSR",
                    "n_markers":       n_loci if n_loci > 0 else np.nan,
                    "lod_score":       np.nan,
                    "study_reference": STUDY_REF,
                    "doi":             DOI,
                    "confirmed_year":  CONF_YEAR,
                    "confidence_level": "probable",
                    "notes":           notes,
                })
                n_support_emitted += 1

    out_df = pd.DataFrame(support_rows, columns=SCHEMA_COLS) if support_rows else pd.DataFrame(columns=SCHEMA_COLS)
    out_df.to_csv(OUT_SUPPORT, index=False)
    print(f"\nMarker support rows: {n_support_emitted} → {OUT_SUPPORT}")
    print(f"  (corroborating SSR profiles for varieties with known parents)")


if __name__ == "__main__":
    main()
