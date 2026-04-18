"""
build_rancho_merced_2022_ssr.py
Extract individual SSR genotypes from Alvarez-Arbesú et al. 2022
(MDPI Plants 11(8):1088, doi: 10.3390/plants11081088)
Rancho de la Merced IFAPA Research Station, Jerez de la Frontera, Spain.

Source: plants-11-01088-s001/Supplementary Material 1.xlsx
  Sheet "unique genotypes" — 521 varieties, 479 with verified SSR profiles
  Sheet "global" — 930 accessions (for duplicate tracking)

13 SSR loci (column positions in the spreadsheet, data starts row index 2
after 2 header rows):
  col 5,6   → VVMD7
  col 9,10  → VVMD32
  col 13,14 → VrZAG62
  col 17,18 → VrZAG79
  col 21,22 → EVA2       (non-GENRES081; allele sizes unknown standard)
  col 23,24 → ISV2
  col 25,26 → VVS2
  col 29,30 → VVMD5
  col 33,34 → VVMD27
  col 37,38 → VVMD25
  col 41,42 → VVMD28
  col 45,46 → ISV4
  col 47,48 → ISV3

Size standard offsets vs VIVC (Rancho = VIVC + offset):
  VVS2    : -1   VVMD5: -4   VVMD7  : -1   VVMD25: 0
  VVMD27  : -3   VVMD28:  0  VVMD32 : -1
  VrZAG62 : -2   VrZAG79: -3
Verified on Cab Sauv, Pinot Noir, Chardonnay, Muscat a Petits Grains Blancs.

Output:
  data/ssr_rancho_merced_2022.csv  — raw profiles (Rancho bp sizes)
  data/ssr_rancho_merced_vivc.csv  — VIVC-standardized bp sizes, keyed on vivc_prime_name

Updates:
  data/node_molecular_profile.csv  — fills empty slots, adds new varieties
  data/datasource_inventory.csv    — adds/updates Rancho entry
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA      = BASE_DIR / "data"
XLSX_PATH = DATA / "plants-11-01088-s001" / "Supplementary Material 1.xlsx"

OUT_RAW   = DATA / "ssr_rancho_merced_2022.csv"
OUT_VIVC  = DATA / "ssr_rancho_merced_vivc.csv"

VIVC_PATH    = DATA / "vivc_supplementary.csv"
ALIAS_PATH   = DATA / "marker_name_aliases.csv"
PROFILE_PATH = DATA / "node_molecular_profile.csv"
INV_PATH     = DATA / "datasource_inventory.csv"

# ── Loci column positions (0-indexed, after skiprows=2) ───────────────────────
# Each locus has (a1_col, a2_col)
LOCUS_COLS: dict[str, tuple[int, int]] = {
    "VVMD7":   (5,  6),
    "VVMD32":  (9,  10),
    "VrZAG62": (13, 14),
    "VrZAG79": (17, 18),
    "EVA2":    (21, 22),   # non-standard, kept as-is
    "ISV2":    (23, 24),
    "VVS2":    (25, 26),
    "VVMD5":   (29, 30),
    "VVMD27":  (33, 34),
    "VVMD25":  (37, 38),
    "VVMD28":  (41, 42),
    "ISV4":    (45, 46),
    "ISV3":    (47, 48),
}

# Offsets to ADD to Rancho allele sizes to get VIVC-standardized sizes
# Confirmed on 4 reference varieties (Cab Sauv, Pinot Noir, Chardonnay, Muscat)
VIVC_OFFSETS: dict[str, int] = {
    "VVS2":    +1,
    "VVMD5":   +4,
    "VVMD7":   +1,
    "VVMD25":   0,
    "VVMD27":  +3,
    "VVMD28":   0,
    "VVMD32":  +1,
    "VrZAG62": +2,
    "VrZAG79": +3,
    # non-GENRES loci: no offset applied (unknown standard)
    "EVA2":     0,
    "ISV2":     0,
    "ISV4":     0,
    "ISV3":     0,
}

GENRES081  = ["VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28",
               "VVMD32", "VrZAG62", "VrZAG79"]
EXTRA_LOCI = ["EVA2", "ISV2", "ISV4", "ISV3"]
ALL_LOCI   = GENRES081 + EXTRA_LOCI

SSR_SOURCE = "Alvarez-Arbesú et al. 2022"


# ── Name normalisation ─────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", ascii_str).strip().upper()


def build_vivc_lookup() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Returns (vivc_no_to_prime, norm_to_prime, syn_to_prime)."""
    df = pd.read_csv(VIVC_PATH, dtype=str, low_memory=False).fillna("")
    vivc_no_to_prime: dict[str, str] = {}
    norm_to_prime:    dict[str, str] = {}
    syn_to_prime:     dict[str, str] = {}

    for _, row in df.iterrows():
        prime  = str(row.get("prime_name", "")).strip()
        vno    = str(row.get("vivc_no", "")).strip()
        if not prime:
            continue
        prime_up = prime.upper()
        if vno:
            vivc_no_to_prime[vno] = prime_up
        norm_to_prime[normalize(prime)] = prime_up
        for s in re.split(r";|\|", str(row.get("synonyms", ""))):
            s = s.strip()
            if s and s.upper() != prime_up:
                syn_to_prime.setdefault(normalize(s), prime_up)

    return vivc_no_to_prime, norm_to_prime, syn_to_prime


def build_alias_map() -> dict[str, str]:
    if not ALIAS_PATH.exists():
        return {}
    df = pd.read_csv(ALIAS_PATH, dtype=str).fillna("")
    return {
        str(r["raw_name"]).strip().upper(): str(r["vivc_prime_name"]).strip().upper()
        for _, r in df.iterrows()
        if r.get("raw_name") and r.get("vivc_prime_name")
    }


def resolve_name(
    raw_name: str,
    vivc_no: str,
    vivc_no_to_prime: dict[str, str],
    norm_to_prime: dict[str, str],
    syn_to_prime:  dict[str, str],
    alias_map:     dict[str, str],
) -> tuple[str, str]:
    raw_upper = str(raw_name).strip().upper()
    raw_norm  = normalize(raw_name)

    # 0. VIVC number (most reliable for a 2022 paper with current VIVC IDs)
    if vivc_no and vivc_no in vivc_no_to_prime:
        return vivc_no_to_prime[vivc_no], "vivc_no"

    # 1. Alias map
    if raw_upper in alias_map:
        return alias_map[raw_upper], "alias"

    # 2. Exact normalised prime name
    if raw_norm in norm_to_prime:
        return norm_to_prime[raw_norm], "exact_normalized"

    # 3. Synonym
    if raw_norm in syn_to_prime:
        return syn_to_prime[raw_norm], "synonym"

    # 4. Compound splitting
    parts = [p.strip() for p in re.split(r"\s*[=()/,]\s*", str(raw_name)) if p.strip()]
    for part in parts:
        if part.upper() == raw_upper:
            continue
        pu = part.upper()
        if pu in alias_map:
            return alias_map[pu], "alias_part"
        pn = normalize(part)
        if pn in norm_to_prime:
            return norm_to_prime[pn], "exact_part"

    # 5. Fuzzy
    try:
        from rapidfuzz import process as rfp, fuzz as rff
        hit = rfp.extractOne(raw_norm, list(norm_to_prime.keys()),
                             scorer=rff.token_sort_ratio, score_cutoff=93)
        if hit:
            return norm_to_prime[hit[0]], f"fuzzy_{hit[1]:.0f}"
    except ImportError:
        pass

    return raw_upper, "name_fallback"


def apply_offset(allele_str: str, offset: int) -> str:
    """Apply +offset to each allele in 'a1/a2' string. Returns '' if input is ''."""
    if not allele_str:
        return ""
    parts = allele_str.split("/")
    try:
        adjusted = [str(int(p) + offset) for p in parts]
        return "/".join(adjusted)
    except ValueError:
        return allele_str  # non-numeric; leave as-is


# ── Extract ──────────────────────────────────────────────────────────────────

def extract_profiles(df: pd.DataFrame) -> list[dict]:
    """
    Extract SSR profiles from the 'unique genotypes' sheet DataFrame
    (already skipped 2 header rows).
    Only processes verified rows (before 'genotypes with unmatching SSR profiles').
    """
    records = []
    in_verified = False
    stopped_at = None

    for i, row in df.iterrows():
        section_marker = str(row.iloc[0]).strip()
        if section_marker == "genotypes with verified SSR profiles":
            in_verified = True
            continue
        if section_marker == "genotypes with unmatching SSR profiles":
            stopped_at = i
            break  # stop — don't include unmatching profiles

        if not in_verified:
            # Before first section marker but after skip — still include
            pass

        variety = str(row.iloc[1]).strip()
        if not variety or variety.lower() in ("nan", "none", "variety prime name", "accession name"):
            continue

        vivc_no = str(row.iloc[2]).strip()
        # Remove .0 suffix from Excel float conversion
        if vivc_no.endswith(".0"):
            vivc_no = vivc_no[:-2]

        rec: dict = {
            "variety_name":    variety,
            "vivc_no_source":  vivc_no,
            "vivc_name_source": str(row.iloc[3]).strip(),  # 'VIVC', 'synonym', etc.
        }

        for locus, (c1, c2) in LOCUS_COLS.items():
            try:
                a1_raw = row.iloc[c1]
                a2_raw = row.iloc[c2]
            except IndexError:
                rec[f"ssr_{locus}_raw"] = ""
                continue

            # Convert float → int string if needed
            def _clean(v) -> str:
                if v is None or str(v).strip() in ("nan", "None", "", "0", "-"):
                    return ""
                try:
                    return str(int(float(str(v))))
                except ValueError:
                    return str(v).strip()

            a1 = _clean(a1_raw)
            a2 = _clean(a2_raw)

            if a1 and a2:
                # Sort numerically
                try:
                    if int(a1) > int(a2):
                        a1, a2 = a2, a1
                except ValueError:
                    pass
                rec[f"ssr_{locus}_raw"] = f"{a1}/{a2}"
            else:
                rec[f"ssr_{locus}_raw"] = ""

        records.append(rec)

    if stopped_at is not None:
        print(f"  Stopped at 'unmatching SSR profiles' section (row {stopped_at})")
    return records


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== build_rancho_merced_2022_ssr.py ===\n")

    if not XLSX_PATH.exists():
        sys.exit(f"[ERROR] Not found: {XLSX_PATH}")

    print(f"Reading: {XLSX_PATH.name}")
    df_raw = pd.read_excel(XLSX_PATH, sheet_name="unique genotypes",
                           header=None, skiprows=2, dtype=str)
    df_raw = df_raw.fillna("")
    print(f"  {len(df_raw)} rows loaded")

    # ── Extract raw profiles ─────────────────────────────────────────────────
    records = extract_profiles(df_raw)
    print(f"  {len(records)} variety profiles extracted")

    df = pd.DataFrame(records)

    # Report coverage
    for locus in ALL_LOCI:
        col = f"ssr_{locus}_raw"
        if col in df.columns:
            n = (df[col] != "").sum()
            print(f"  {locus:12s}: {n}/{len(df)}")

    # ── VIVC name resolution ─────────────────────────────────────────────────
    print("\n[VIVC name resolution]")
    vivc_no_to_prime, norm_to_prime, syn_to_prime = build_vivc_lookup()
    alias_map = build_alias_map()
    print(f"  VIVC prime names: {len(norm_to_prime)}")

    method_counts: dict[str, int] = {}
    vivc_rows = []
    for _, row in df.iterrows():
        prime, method = resolve_name(
            row["variety_name"], row["vivc_no_source"],
            vivc_no_to_prime, norm_to_prime, syn_to_prime, alias_map,
        )
        method_counts[method] = method_counts.get(method, 0) + 1

        vivc_row: dict = {
            "vivc_prime_name":      prime,
            "vivc_no_source":       row["vivc_no_source"],
            "rancho_variety_name":  row["variety_name"],
            "match_method":         method,
            "ssr_source":           SSR_SOURCE,
        }

        # Raw allele values
        for locus in ALL_LOCI:
            vivc_row[f"ssr_{locus}_raw"] = row.get(f"ssr_{locus}_raw", "")

        # VIVC-standardized allele values (offset-corrected)
        for locus in GENRES081:
            raw = row.get(f"ssr_{locus}_raw", "")
            offset = VIVC_OFFSETS.get(locus, 0)
            vivc_row[f"ssr_{locus}"] = apply_offset(raw, offset) if raw else ""

        vivc_rows.append(vivc_row)

    vivc_df = pd.DataFrame(vivc_rows)

    print("  Resolution methods:")
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        print(f"    {method:25s}: {count}")

    fallbacks = (vivc_df["match_method"] == "name_fallback").sum()
    print(f"\n  VIVC resolved:   {len(vivc_df) - fallbacks}/{len(vivc_df)}")
    print(f"  Name fallbacks:  {fallbacks}")

    # Save raw CSV
    df.to_csv(OUT_RAW, index=False)
    print(f"\nRaw CSV → {OUT_RAW}")

    # Save VIVC-keyed CSV
    vivc_df.to_csv(OUT_VIVC, index=False)
    print(f"VIVC CSV → {OUT_VIVC}")

    # ── Update node_molecular_profile.csv ────────────────────────────────────
    print("\n[Updating node_molecular_profile.csv]")
    profile = pd.read_csv(PROFILE_PATH, dtype=str).fillna("")
    print(f"  Current rows: {len(profile)}")

    GENRES_COLS = [f"ssr_{l}" for l in GENRES081]
    for col in GENRES_COLS + ["ssr_source"]:
        if col not in profile.columns:
            profile[col] = ""

    profile_names = set(profile["vivc_prime_name"].str.strip().str.upper())

    vivc_df_up = vivc_df.copy()
    vivc_df_up["vivc_prime_name"] = vivc_df_up["vivc_prime_name"].str.strip().str.upper()
    vivc_df_up = vivc_df_up[vivc_df_up["vivc_prime_name"] != ""]

    # Backfill empty SSR slots in existing rows
    backfill_count = 0
    for col in GENRES_COLS:
        profile[col] = profile[col].str.strip()
        missing_mask = profile[col] == ""
        fill_map = (
            vivc_df_up[vivc_df_up[col] != ""]
            .set_index("vivc_prime_name")[col]
            .to_dict()
        )
        filled = (
            profile.loc[missing_mask, "vivc_prime_name"]
            .str.strip().str.upper()
            .map(fill_map)
            .fillna("")
        )
        n = (filled != "").sum()
        backfill_count += n
        profile.loc[missing_mask, col] = filled.values

    # Backfill ssr_source for rows that just got SSR data
    src_fill = vivc_df_up.set_index("vivc_prime_name")["ssr_source"].to_dict()
    has_ssr = profile[GENRES_COLS[0]].str.strip() != ""
    no_src  = profile["ssr_source"].str.strip() == ""
    profile.loc[has_ssr & no_src, "ssr_source"] = (
        profile.loc[has_ssr & no_src, "vivc_prime_name"]
        .str.strip().str.upper()
        .map(src_fill)
        .fillna("")
        .values
    )

    print(f"  Backfilled {backfill_count} SSR cells in existing rows")

    # Add new rows
    new_df = vivc_df_up[~vivc_df_up["vivc_prime_name"].isin(profile_names)].copy()
    has_data = new_df[GENRES_COLS].apply(lambda r: any(v for v in r if v), axis=1)
    new_df = new_df[has_data]
    print(f"  New rows to add:  {len(new_df)}")

    if len(new_df) > 0:
        new_rows = pd.DataFrame(index=range(len(new_df)))
        for col in profile.columns:
            if col in new_df.columns:
                new_rows[col] = new_df[col].values
            else:
                new_rows[col] = ""
        new_rows["ssr_source"] = SSR_SOURCE
        updated = pd.concat([profile, new_rows], ignore_index=True)
        updated.to_csv(PROFILE_PATH, index=False)
        print(f"  Saved: {len(updated)} rows (+{len(new_df)})")
    else:
        profile.to_csv(PROFILE_PATH, index=False)
        print(f"  Saved: {len(profile)} rows (backfills only)")

    # ── Update datasource_inventory.csv ──────────────────────────────────────
    if INV_PATH.exists():
        inv = pd.read_csv(INV_PATH, dtype=str).fillna("")
        row_id = "RanchoMerced2022_SSR"
        mask = inv["source_id"] == row_id
        n_total = len(vivc_df)
        n_resolved = n_total - fallbacks
        n_genre = (vivc_df[GENRES_COLS[0]] != "").sum()

        entry = {
            "source_id":         row_id,
            "citation":          "Alvarez-Arbesú et al. 2022, MDPI Plants 11(8):1088 (doi: 10.3390/plants11081088)",
            "year":              "2022",
            "marker_type":       "SSR (microsatellite)",
            "platform":          "CE / GenomeLabTM GeXP (Beckman Coulter)",
            "loci":              "VVMD7, VVMD32, VrZAG62, VrZAG79, EVA2, ISV2, VVS2, VVMD5, VVMD27, VVMD25, VVMD28, ISV4, ISV3 (9 GENRES081 + 4 extra)",
            "n_varieties_total": str(n_total),
            "n_varieties_obtained": str(n_resolved),
            "access_method":     "Supplementary Material 1.xlsx from PMC (open access CC BY)",
            "raw_data_in_supp":  "True",
            "notes":             (
                f"{n_total} verified unique genotypes from 930 accessions, mainly Iberian (Spanish/Portuguese) varieties. "
                f"VIVC-matched via vivc_no (2022 data, current IDs). "
                f"Systematic size standard offset vs VIVC: VVS2=-1, VVMD5=-4, VVMD7=-1, VVMD25=0, VVMD27=-3, "
                f"VVMD28=0, VVMD32=-1, VrZAG62=-2, VrZAG79=-3 (confirmed on 4 reference varieties). "
                f"Offset-corrected values stored in ssr_LOCUS columns of ssr_rancho_merced_vivc.csv."
            ),
            "availability":      "Full (locally available)",
            "data_type_for_pedigree": "Direct SSR parentage testing",
        }

        if mask.any():
            for k, v in entry.items():
                if k in inv.columns:
                    inv.loc[mask, k] = v
            print(f"\nUpdated {row_id} in datasource_inventory.csv")
        else:
            new_entry = pd.DataFrame([entry])
            for col in inv.columns:
                if col not in new_entry.columns:
                    new_entry[col] = ""
            inv = pd.concat([inv, new_entry[inv.columns]], ignore_index=True)
            print(f"\nAdded {row_id} to datasource_inventory.csv")

        inv.to_csv(INV_PATH, index=False)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
