"""
build_margaryan_2021_pedigree.py
=================================
Extract pedigree relationships from Margaryan et al. 2021 (Biology 10:1279,
DOI 10.3390/biology10121279) Tables S6 (full trios) and S7 (half duos)
and emit data/margaryan_2021_marker_support.csv in the standard long format.

Table S6 — full parentages (65 data rows → 130 parent-role rows):
  Row 3 = headers, row 4+ = data
  Col 0/1 = child name/vivc_no
  Col 2/3 = Parent 1 name/vivc_no
  Col 4/5 = Parent 2 name/vivc_no
  Col 6   = "Missing/Applied/Mismatch" (e.g. "0/24/0")
  Col 7   = LOD score
  Col 8   = Comments

Table S7 — half parentages (370 data rows → 1 row each, directional):
  Row 3 = headers, row 4+ = data
  Col 0/1 = Cultivar 1 name/vivc_no
  Col 2/3 = Cultivar 2 name/vivc_no
  Col 4   = Loci compared
  Col 5   = LOD score
  Col 6   = Relationship (P1/P2/Off/PO/NaN)
  Col 7   = Comments

Output schema (standard long format):
  child_variety, parent_variety, parent_role, evidence_type, marker_type,
  n_markers, lod_score, study_reference, doi, confirmed_year,
  confidence_level, notes
"""

from __future__ import annotations

import math
import re
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"
SUPP_DIR  = DATA_DIR / "biology-10-01279-s001"

S6_PATH   = SUPP_DIR / "Table S6. Complete list of full parentages of analyzed varieties.xlsx"
S7_PATH   = SUPP_DIR / "Table S7. Complete list of half parentages of analyzed varieties.xlsx"

OUT_PATH  = DATA_DIR / "margaryan_2021_marker_support.csv"
VIVC_MS   = DATA_DIR / "vivc_confirmed_marker_support.csv"

STUDY_REF    = "Margaryan et al. 2021"
DOI          = "10.3390/biology10121279"
CONF_YEAR    = 2021
SOURCE_NOTE  = "Margaryan 2021 Biology 10:1279; 24 nSSR loci"

# Keywords that indicate this is a confirmed/published result (not novel)
_PUBLICATION_REFS = ("lacombe", "laucou", "crespan", "ibañez", "ibanez")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(val) -> str:
    """Normalise a cell value to stripped uppercase string, or ''."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip().upper()


def _str(val) -> str:
    """Return string value stripped of whitespace, or ''."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip()


def _float_or_none(val) -> float | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_nssr(s: str) -> tuple[int | None, int | None]:
    """
    Parse "Missing/Applied/Mismatch" string, e.g. "0/24/0".
    Returns (n_applied, n_mismatch).
    """
    m = re.match(r"(\d+)/(\d+)/(\d+)", s.strip())
    if m:
        return int(m.group(2)), int(m.group(3))
    return None, None


def _has_publication_ref(comment: str) -> bool:
    """Return True if comment mentions a prior publication (Lacombe, Laucou, etc.)."""
    cl = comment.lower()
    return any(kw in cl for kw in _PUBLICATION_REFS)


def _is_invalidation(comment: str) -> bool:
    return "invalidation" in comment.lower()


def _is_first_confirmation(comment: str) -> bool:
    """
    True when this is likely the first molecular confirmation:
    - comment is empty, OR
    - "No information on the original cross", OR
    - "Identification of P1 and P2"
    AND does NOT reference a prior publication.
    """
    cl = comment.lower().strip()
    triggers = (
        cl == "",
        "no information on the original cross" in cl,
        "identification of p1 and p2" in cl,
    )
    return any(triggers) and not _has_publication_ref(comment)


def _trio_confidence(mismatch: int | None, comment: str) -> str:
    if _is_invalidation(comment):
        return "rejected"
    if mismatch is None:
        return "probable"
    if mismatch <= 1:
        return "confirmed"
    if mismatch <= 3:
        return "probable"
    return "rejected"


def _build_notes(comment: str, source_note: str, first_conf: bool) -> str:
    parts = []
    if comment:
        parts.append(comment)
    parts.append(source_note)
    if first_conf:
        parts.append("first_molecular_confirmation")
    return " | ".join(parts)


# ── Parse Table S6 (trios) ────────────────────────────────────────────────────

def parse_s6() -> pd.DataFrame:
    """
    Return a DataFrame with one row per (child, parent) pair.
    Each trio produces 2 rows: Parent 1 and Parent 2.
    """
    df = pd.read_excel(S6_PATH, sheet_name="Tabelle S6", header=None)
    # Row 0=title, 1=blank, 2=group headers, 3=column headers, 4+=data
    data = df.iloc[4:].copy().reset_index(drop=True)

    rows = []
    for _, row in data.iterrows():
        child      = _norm(row.iloc[0])
        parent1    = _norm(row.iloc[2])
        parent2    = _norm(row.iloc[4])
        nssr_str   = _str(row.iloc[6])
        lod_raw    = _float_or_none(row.iloc[7])
        comment    = _str(row.iloc[8])

        # Skip header echo row (can appear if 'Comments' literal is a value)
        if child == "VIVC PRIME NAME" or child == "":
            continue
        # Skip "Comments" echo rows from merged cells
        if nssr_str == "" and lod_raw is None and comment == "Comments":
            continue

        n_applied, n_mismatch = _parse_nssr(nssr_str)
        confidence = _trio_confidence(n_mismatch, comment)
        first_conf = _is_first_confirmation(comment)
        notes      = _build_notes(comment, SOURCE_NOTE, first_conf)

        for parent, role in [(parent1, "Parent 1"), (parent2, "Parent 2")]:
            if not parent:
                continue
            rows.append({
                "child_variety":    child,
                "parent_variety":   parent,
                "parent_role":      role,
                "evidence_type":    "SSR",
                "marker_type":      "SSR",
                "n_markers":        n_applied,
                "lod_score":        lod_raw,
                "study_reference":  STUDY_REF,
                "doi":              DOI,
                "confirmed_year":   CONF_YEAR,
                "confidence_level": confidence,
                "notes":            notes,
            })

    return pd.DataFrame(rows)


# ── Parse Table S7 (duos) ─────────────────────────────────────────────────────

def parse_s7() -> pd.DataFrame:
    """
    Return a DataFrame with one row per half-parentage relationship.

    Relationship interpretation (Cultivar 2 relative to Cultivar 1):
      P1  → child=Cultivar1, parent=Cultivar2, role="Parent 1"
      P2  → child=Cultivar1, parent=Cultivar2, role="Parent 2"
      Off → child=Cultivar2, parent=Cultivar1, role="Parent 1"
      PO  → parent-offspring direction unknown; emit as
             child=Cultivar1, parent=Cultivar2, role="Parent 1"
             with note that direction is uncertain
      NaN → ambiguous/speculative rows → skip

    Skip rows where Cultivar 2 vivc_no == 0 AND name contains "GENOTYPE"
    (unnamed accessions without VIVC representation).
    """
    df = pd.read_excel(S7_PATH, sheet_name="Tabelle S7", header=None)
    # Row 0=title, 1=blank, 2=group headers, 3=column headers, 4+=data
    data = df.iloc[4:].copy().reset_index(drop=True)

    rows = []
    for _, row in data.iterrows():
        cult1_name = _norm(row.iloc[0])
        cult1_vivc = row.iloc[1]
        cult2_name = _norm(row.iloc[2])
        cult2_vivc = row.iloc[3]
        loci_cmp   = _float_or_none(row.iloc[4])
        lod_raw    = _float_or_none(row.iloc[5])
        rel        = _str(row.iloc[6]).strip()
        comment    = _str(row.iloc[7])

        if cult1_name == "" or cult1_name == "VIVC PRIME NAME":
            continue

        # Skip NaN / ambiguous relationships
        if rel == "" or rel.upper() not in ("P1", "P2", "OFF", "PO"):
            continue

        # Skip unnamed GENOTYPE rows with vivc_no == 0 for Cultivar 2
        try:
            c2_no = int(float(str(cult2_vivc)))
        except (ValueError, TypeError):
            c2_no = -1

        if c2_no == 0 and "GENOTYPE" in cult2_name:
            continue

        # Determine direction
        rel_upper = rel.upper()
        if rel_upper == "P1":
            child   = cult1_name
            parent  = cult2_name
            role    = "Parent 1"
        elif rel_upper == "P2":
            child   = cult1_name
            parent  = cult2_name
            role    = "Parent 2"
        elif rel_upper == "OFF":
            child   = cult2_name
            parent  = cult1_name
            role    = "Parent 1"
        else:  # PO — direction unknown
            child   = cult1_name
            parent  = cult2_name
            role    = "Parent 1"
            # Append direction note
            if comment:
                comment = comment + "; parent-offspring direction unknown"
            else:
                comment = "parent-offspring direction unknown"

        first_conf = _is_first_confirmation_duo(comment)
        notes = _build_notes(comment, SOURCE_NOTE, first_conf)

        # S7 loci_cmp may be an int
        n_markers = int(loci_cmp) if loci_cmp is not None else None

        rows.append({
            "child_variety":    child,
            "parent_variety":   parent,
            "parent_role":      role,
            "evidence_type":    "SSR",
            "marker_type":      "SSR",
            "n_markers":        n_markers,
            "lod_score":        lod_raw,
            "study_reference":  STUDY_REF,
            "doi":              DOI,
            "confirmed_year":   CONF_YEAR,
            "confidence_level": "probable",
            "notes":            notes,
        })

    return pd.DataFrame(rows)


def _is_first_confirmation_duo(comment: str) -> bool:
    """
    First molecular confirmation for duo rows:
    - comment is empty, OR "no name new cross" (case-insensitive)
    AND does NOT reference a prior publication.
    """
    cl = comment.lower().strip()
    # Remove appended direction note for this check
    cl_base = cl.replace("; parent-offspring direction unknown", "").strip()
    triggers = (
        cl_base == "",
        "no name new cross" in cl_base,
        "no name most likely a new cross" in cl_base,
    )
    return any(triggers) and not _has_publication_ref(comment)


# ── Cross-check helpers ────────────────────────────────────────────────────────

def _load_vivc_confirmed_children() -> set[str]:
    if not VIVC_MS.exists():
        return set()
    df = pd.read_csv(VIVC_MS, low_memory=False)
    return set(df["child_variety"].dropna().str.strip().str.upper())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("build_margaryan_2021_pedigree.py")
    print("=" * 60)

    print("\n[1] Parsing Table S6 (full parentage trios)...")
    s6_df = parse_s6()
    print(f"  {len(s6_df)} parent-role rows from trios")

    print("\n[2] Parsing Table S7 (half parentage duos)...")
    s7_df = parse_s7()
    print(f"  {len(s7_df)} rows from duos")

    # Combine
    all_df = pd.concat([s6_df, s7_df], ignore_index=True)

    print(f"\n[3] Writing {len(all_df)} rows → {OUT_PATH.name}...")
    all_df.to_csv(OUT_PATH, index=False)
    print("  Done.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows written:    {len(all_df)}")

    # Trio stats
    trio_conf  = s6_df[s6_df["confidence_level"] == "confirmed"]
    trio_novel_mask = s6_df["notes"].apply(lambda n: "first_molecular_confirmation" in str(n))
    trio_novel = s6_df[trio_novel_mask]

    # Duo stats
    duo_novel_mask = s7_df["notes"].apply(lambda n: "first_molecular_confirmation" in str(n))
    duo_novel = s7_df[duo_novel_mask]

    # Unique trio children (before doubling into 2 rows per trio)
    trio_children = s6_df["child_variety"].nunique()
    print(f"\nTrios (Table S6):")
    print(f"  Unique trios (child varieties): {trio_children}")
    print(f"  Parent-role rows:               {len(s6_df)}")
    print(f"  Confirmed (mismatch ≤ 1):        {s6_df['confidence_level'].eq('confirmed').sum()}")
    print(f"  Probable (mismatch 2-3):          {s6_df['confidence_level'].eq('probable').sum()}")
    print(f"  Rejected (invalidation):          {s6_df['confidence_level'].eq('rejected').sum()}")
    print(f"  Novel (first molecular conf):     {len(trio_novel)} rows ({trio_novel['child_variety'].nunique()} children)")

    print(f"\nDuos (Table S7):")
    print(f"  Total rows:                       {len(s7_df)}")
    print(f"  Novel (first molecular conf):     {len(duo_novel)} rows")

    # Cross-check: child_varieties in vivc_confirmed_marker_support
    vivc_children = _load_vivc_confirmed_children()
    all_children  = set(all_df["child_variety"].str.strip().str.upper())
    overlap       = all_children & vivc_children
    print(f"\nCross-check vs vivc_confirmed_marker_support.csv:")
    print(f"  Child varieties in Margaryan output:    {len(all_children)}")
    print(f"  Already in vivc_confirmed_marker_support: {len(overlap)}")
    print(f"  New children not yet in VIVC confirmed:   {len(all_children - vivc_children)}")


if __name__ == "__main__":
    main()
