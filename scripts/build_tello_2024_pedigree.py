"""
build_tello_2024_pedigree.py
============================
Extract parent-offspring pedigree relationships from Tello et al. 2024
(Frontiers in Plant Science 15:1391679) and create tello_2024_marker_support.csv.

Sources:
  Table 4 (SM4): 35 confirmed parent-offspring trios (offspring + 2 parents)
  Table 5 (SM5): 31 confirmed parent-offspring duos (single-parent relationships)

Both tables contain VIVC variety numbers for matching to canonical prime names.
Evidence: SNP array (VIVC18K) + SSR (GENRES081 subset: 7 loci), 0 mismatches in most cases.

Output: data/tello_2024_marker_support.csv
Format mirrors laucou_2018_marker_support.csv:
  child_variety, parent_variety, parent_role, evidence_type, marker_type,
  n_markers, lod_score, study_reference, doi, confirmed_year, confidence_level, notes
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA    = Path(__file__).parent.parent / "data"
T4_PATH = DATA / "Tello et al 2024" / "Table 4.XLSX"
T5_PATH = DATA / "Tello et al 2024" / "Table 5.XLSX"
VIVC_SUP = DATA / "vivc_supplementary.csv"
OUT_PATH = DATA / "tello_2024_marker_support.csv"

STUDY_REF = "Tello et al. 2024"
DOI       = "10.3389/fpls.2024.1391679"
YEAR      = 2024


# ── helpers ──────────────────────────────────────────────────────────────────

def load_vivc_prime_names() -> dict[int, str]:
    df = pd.read_csv(VIVC_SUP, low_memory=False)
    id_col   = [c for c in df.columns if "vivc" in c.lower() and "no" in c.lower()][0]
    name_col = [c for c in df.columns if "prime" in c.lower() or "variety" in c.lower()][0]
    out = {}
    for _, row in df.iterrows():
        try:
            vivc_no = int(row[id_col])
            out[vivc_no] = str(row[name_col]).upper().strip()
        except (ValueError, TypeError):
            pass
    return out


def resolve_name(raw_name: str, vivc_no, vivc_map: dict) -> str:
    """Return the best canonical name: VIVC prime name if available, else normalised raw name."""
    try:
        vivc_int = int(vivc_no)
        prime = vivc_map.get(vivc_int)
        if prime:
            return prime
    except (ValueError, TypeError):
        pass
    return str(raw_name).upper().strip()


def safe_float(val) -> float | None:
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val) -> int | None:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


# ── Table 4: trios ────────────────────────────────────────────────────────────

def _is_novel(ref_str: str) -> bool:
    """Return True only if the reference indicates this is Tello's own new discovery."""
    s = str(ref_str).strip().lower()
    if s in ("", "nan", "-"):
        return True
    # "This work" = Tello's novel contribution
    if "this work" in s:
        # Also allow if it ONLY says "this work" (possibly combined with Röckel in-prep)
        # Exclude if it ALSO cites major prior databases (already loaded separately)
        excludes = ["lacombe", "laucou", "maras", "stajner", "cunha", "kiss", "vargas",
                    "crespan", "zinelabidine", "schneider", "d'onofrio", "ghaffari",
                    "margaryan", "zuhl", "zulj"]
        if any(e in s for e in excludes):
            return False
        return True
    return False


def load_trios(vivc_map: dict) -> list[dict]:
    """Returns a list of parent-child row dicts (2 rows per trio) — novel Tello discoveries only."""
    # Row 0 in original = title; Row 1 = NaN; Row 2 = headers; Row 3+ = data
    # pd.read_excel with skiprows=2 puts the header row (row 2) as column names
    df = pd.read_excel(T4_PATH, sheet_name="SM4", skiprows=2, header=0)

    # Rename columns for easy access (they may vary by Excel formatting)
    col_map = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if cs == "offspring":                              col_map[c] = "offspring_name"
        elif "offspring: icvv" in cs:                     col_map[c] = "offspring_icvv"
        elif "offspring: vivc" in cs:                     col_map[c] = "offspring_vivc"
        elif "offspring: chlorotype" in cs:               col_map[c] = "offspring_chlorotype"
        elif cs == "parent 1":                            col_map[c] = "p1_name"
        elif "parent 1: icvv" in cs:                      col_map[c] = "p1_icvv"
        elif "parent 1: vivc" in cs:                      col_map[c] = "p1_vivc"
        elif "parent 1: chlorotype" in cs:                col_map[c] = "p1_chlorotype"
        elif cs == "parent 2":                            col_map[c] = "p2_name"
        elif "parent 2: icvv" in cs:                      col_map[c] = "p2_icvv"
        elif "parent 2: vivc" in cs:                      col_map[c] = "p2_vivc"
        elif "parent 2: chlorotype" in cs:                col_map[c] = "p2_chlorotype"
        elif "compared snp" in cs:                        col_map[c] = "n_snps"
        elif "mismatching snp" in cs:                     col_map[c] = "n_mismatch_snps"
        elif "lod score" in cs:                           col_map[c] = "lod_score"
        elif "compared ssr" in cs:                        col_map[c] = "n_ssrs"
        elif "mismatching ssr" in cs:                     col_map[c] = "n_mismatch_ssrs"
        elif "bred cultivar" in cs:                       col_map[c] = "bred_cultivar"
        elif "described cross" in cs:                     col_map[c] = "described_cross"
        elif "references" in cs:                          col_map[c] = "references"
    df = df.rename(columns=col_map)

    rows = []
    for _, r in df.iterrows():
        offspring = resolve_name(r.get("offspring_name", ""), r.get("offspring_vivc"), vivc_map)
        p1        = resolve_name(r.get("p1_name", ""), r.get("p1_vivc"), vivc_map)
        p2        = resolve_name(r.get("p2_name", ""), r.get("p2_vivc"), vivc_map)

        if not offspring or offspring == "NAN":
            continue

        n_snps    = safe_int(r.get("n_snps"))
        lod       = safe_float(r.get("lod_score"))
        n_ssrs    = safe_int(r.get("n_ssrs"))
        n_mm_snp  = safe_int(r.get("n_mismatch_snps"))
        n_mm_ssr  = safe_int(r.get("n_mismatch_ssrs"))
        refs      = str(r.get("references", "")).strip()
        bred      = str(r.get("bred_cultivar", "")).strip()
        cross_desc = str(r.get("described_cross", "")).strip()

        # Skip relationships already published in prior literature
        # (Lacombe 2013, Laucou 2018, etc. are already loaded separately).
        # Only include parentages first reported by Tello 2024.
        if not _is_novel(refs):
            continue

        # Build notes string
        notes_parts = []
        if n_snps:       notes_parts.append(f"SNPs: {n_snps}, mismatches: {n_mm_snp or 0}")
        if n_ssrs:       notes_parts.append(f"SSRs: {n_ssrs}, mismatches: {n_mm_ssr or 0}")
        if bred == "Yes": notes_parts.append("Bred cultivar")
        if cross_desc and cross_desc != "nan": notes_parts.append(f"Cross: {cross_desc}")
        if refs and refs != "nan": notes_parts.append(f"Previously: {refs}")
        notes = "; ".join(notes_parts)

        # Marker type string
        marker_type = "SNP+SSR" if n_ssrs else "SNP"
        evidence    = "SNP (Vitis18K) + SSR" if n_ssrs else "SNP (Vitis18K)"

        base = dict(
            parent_role    = "",
            evidence_type  = evidence,
            marker_type    = marker_type,
            n_markers      = n_snps or "",
            lod_score      = lod or "",
            study_reference= STUDY_REF,
            doi            = DOI,
            confirmed_year = YEAR,
            confidence_level = "confirmed",
            notes          = notes,
        )

        if p1 and p1 != "NAN":
            rows.append({**base, "child_variety": offspring, "parent_variety": p1})
        if p2 and p2 != "NAN":
            rows.append({**base, "child_variety": offspring, "parent_variety": p2})

    return rows


# ── Table 5: duos ─────────────────────────────────────────────────────────────

def load_duos(vivc_map: dict) -> list[dict]:
    """Returns a list of parent-child row dicts (1 row per duo) — novel Tello discoveries only."""
    df = pd.read_excel(T5_PATH, skiprows=2, header=0)

    col_map = {}
    for c in df.columns:
        cs = str(c).strip().lower()
        if "cultivar 1: name" in cs:                col_map[c] = "c1_name"
        elif "cultivar 1: icvv" in cs:              col_map[c] = "c1_icvv"
        elif "cultivar 1: vivc" in cs:              col_map[c] = "c1_vivc"
        elif "cultivar 1: chlorotype" in cs:        col_map[c] = "c1_chlorotype"
        elif "cultivar 2: name" in cs:              col_map[c] = "c2_name"
        elif "cultivar 2: icvv" in cs:              col_map[c] = "c2_icvv"
        elif "cultivar 2: vivc" in cs:              col_map[c] = "c2_vivc"
        elif "cultivar 2: chlorotype" in cs:        col_map[c] = "c2_chlorotype"
        elif "compared snp" in cs:                  col_map[c] = "n_snps"
        elif "mismatching snp" in cs:               col_map[c] = "n_mismatch_snps"
        elif "lod score" in cs:                     col_map[c] = "lod_score"
        elif "compared ssr" in cs:                  col_map[c] = "n_ssrs"
        elif "mismatching ssr" in cs:               col_map[c] = "n_mismatch_ssrs"
        elif "references" in cs:                    col_map[c] = "references"
    df = df.rename(columns=col_map)

    rows = []
    for _, r in df.iterrows():
        c1 = resolve_name(r.get("c1_name", ""), r.get("c1_vivc"), vivc_map)
        c2 = resolve_name(r.get("c2_name", ""), r.get("c2_vivc"), vivc_map)

        if not c1 or c1 == "NAN" or not c2 or c2 == "NAN":
            continue

        n_snps   = safe_int(r.get("n_snps"))
        lod      = safe_float(r.get("lod_score"))
        n_ssrs   = safe_int(r.get("n_ssrs"))
        n_mm_snp = safe_int(r.get("n_mismatch_snps"))
        n_mm_ssr = safe_int(r.get("n_mismatch_ssrs"))
        refs     = str(r.get("references", "")).strip()

        # Skip relationships already in prior literature
        if not _is_novel(refs):
            continue

        notes_parts = []
        if n_snps:   notes_parts.append(f"SNPs: {n_snps}, mismatches: {n_mm_snp or 0}")
        if n_ssrs:   notes_parts.append(f"SSRs: {n_ssrs}, mismatches: {n_mm_ssr or 0}")
        if refs and refs not in ("nan", "-"): notes_parts.append(f"Previously: {refs}")
        notes = "; ".join(notes_parts) + "; direction not specified (duo)"

        marker_type = "SNP+SSR" if n_ssrs else "SNP"
        evidence    = "SNP (Vitis18K) + SSR" if n_ssrs else "SNP (Vitis18K)"

        rows.append(dict(
            child_variety   = c1,
            parent_variety  = c2,
            parent_role     = "",
            evidence_type   = evidence,
            marker_type     = marker_type,
            n_markers       = n_snps or "",
            lod_score       = lod or "",
            study_reference = STUDY_REF,
            doi             = DOI,
            confirmed_year  = YEAR,
            confidence_level= "confirmed",
            notes           = notes,
        ))

    return rows


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading VIVC prime-name map...")
    vivc_map = load_vivc_prime_names()
    print(f"  {len(vivc_map)} VIVC IDs loaded")

    print("Loading Table 4 (trios)...")
    trio_rows = load_trios(vivc_map)
    print(f"  {len(trio_rows)} child-parent rows from trios")

    print("Loading Table 5 (duos)...")
    duo_rows  = load_duos(vivc_map)
    print(f"  {len(duo_rows)} child-parent rows from duos")

    all_rows = trio_rows + duo_rows
    df_out = pd.DataFrame(all_rows, columns=[
        "child_variety", "parent_variety", "parent_role",
        "evidence_type", "marker_type", "n_markers", "lod_score",
        "study_reference", "doi", "confirmed_year", "confidence_level", "notes",
    ])

    df_out = df_out[
        df_out["child_variety"].fillna("").str.strip().ne("") &
        df_out["parent_variety"].fillna("").str.strip().ne("")
    ]

    df_out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df_out)} rows → {OUT_PATH.name}")

    print("\nSample rows:")
    print(df_out[["child_variety", "parent_variety", "n_markers", "lod_score"]].head(10).to_string(index=False))

    print("\nUnique varieties as children:", df_out["child_variety"].nunique())
    print("Unique varieties as parents :", df_out["parent_variety"].nunique())

    # Overlap with existing laucou_2018 support
    lac = pd.read_csv(DATA / "laucou_2018_marker_support.csv", low_memory=False)
    lac_pairs = set(zip(lac["child_variety"].str.upper(), lac["parent_variety"].str.upper()))
    tello_pairs = set(zip(df_out["child_variety"].str.upper(), df_out["parent_variety"].str.upper()))
    overlap = lac_pairs & tello_pairs
    novel   = tello_pairs - lac_pairs
    print(f"\nOverlap with Laucou 2018 (already confirmed elsewhere): {len(overlap)}")
    print(f"Novel Tello-only relationships: {len(novel)}")
    if novel:
        for c, p in sorted(novel)[:15]:
            print(f"  {c} ← {p}")


if __name__ == "__main__":
    main()
