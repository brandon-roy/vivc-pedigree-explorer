"""
build_lacombe_ssr_profiles.py
Extract individual SSR genotypes (20 nSSR loci) from Lacombe et al. 2013
Online Resource 4 (MOESM4) — 828 progenies + 434 genitors = 1,262 varieties.

Loci order (decoded from rotated column headers, left-to-right):
  VMC1b11, VMC4f3, VVIb01, VVIh54, VVIn16, VVIn73, VVIp31, VVIp60,
  VVIq52, VVIv37, VVIv67, VVMD21, VVMD24, VVMD25, VVMD27, VVMD28,
  VVMD32, VVMD5, VVMD7, VVS2

GENRES081 overlap loci: VVMD5, VVMD7, VVMD25, VVMD27, VVMD28, VVMD32, VVS2
(VrZAG62 and VrZAG79 are NOT in this panel)

Output:
  data/ssr_lacombe_2013.csv   — raw profiles (all 20 loci, allele1/allele2)
  data/ssr_lacombe_vivc.csv   — GENRES081 overlap loci keyed on vivc_prime_name

doi: 10.1007/s00122-012-1988-2
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("pip install pdfplumber")

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "_lacombe_pdfs" / "MOESM4.pdf"
OUT_RAW  = BASE_DIR / "data" / "ssr_lacombe_2013.csv"
OUT_VIVC = BASE_DIR / "data" / "ssr_lacombe_vivc.csv"

# ── Loci (20 total, in PDF column order) ───────────────────────────────────────
LOCI_20 = [
    "VMC1b11", "VMC4f3", "VVIb01", "VVIh54",
    "VVIn16",  "VVIn73", "VVIp31", "VVIp60",
    "VVIq52",  "VVIv37", "VVIv67", "VVMD21",
    "VVMD24",  "VVMD25", "VVMD27", "VVMD28",
    "VVMD32",  "VVMD5",  "VVMD7",  "VVS2",
]
# Number of digits for each allele (most are 3-digit; VVIq52 alleles are 2-digit)
ALLELE_SIZES = [3, 3, 3, 3, 3, 3, 3, 3,  # VMC1b11..VVIp60
                2,                          # VVIq52
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # VVIv37..VVS2

GENRES081_OVERLAP = ["VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28", "VVMD32", "VVS2"]

# ─── Data-row pattern ─────────────────────────────────────────────────────────
# "VarietyName (#12345)allele1:allele2allele1:allele2..."
# Adjacent pairs are concatenated without separator, so we split on ':'
ROW_RE = re.compile(r"^(.+?)\s*\(#(\d+)\)\s*(\d[\d:]+\d)$", re.UNICODE)


def parse_allele_colons(allele_str: str) -> list[tuple[str, str]] | None:
    """
    Parse 20 allele pairs from a colon-separated string where adjacent pairs
    are concatenated: '165:184171:181...'

    Uses ALLELE_SIZES to determine the digit-count of each allele2, which
    determines where the next allele1 begins within each colon-separated segment.
    """
    segs = allele_str.split(":")
    if len(segs) != 21:
        return None   # expect exactly 20 colons → 21 segments

    alleles: list[str | None] = [None] * 40  # 20 loci × (a1, a2)
    alleles[0] = segs[0]  # a1 of locus 0

    for i in range(1, 20):
        seg = segs[i]
        a2_size = ALLELE_SIZES[i - 1]
        if len(seg) < a2_size:
            return None    # malformed
        alleles[2 * (i - 1) + 1] = seg[:a2_size]   # a2 of locus i-1
        alleles[2 * i]            = seg[a2_size:]    # a1 of locus i

    alleles[39] = segs[20]  # a2 of locus 19

    # Validate: all slots filled, all non-empty
    if any(v is None or v == "" for v in alleles):
        return None

    return [(alleles[2 * i], alleles[2 * i + 1]) for i in range(20)]


def parse_row(text: str) -> dict | None:
    text = text.replace("\xa0", " ").strip()
    m = ROW_RE.match(text)
    if not m:
        return None
    variety_name = m.group(1).strip()
    vivc_no      = m.group(2)
    allele_str   = m.group(3)

    pairs = parse_allele_colons(allele_str)
    if pairs is None:
        return None

    rec = {
        "variety_name": variety_name,
        "vivc_no":       vivc_no,
    }
    for i, locus in enumerate(LOCI_20):
        rec[f"ssr_{locus}_a1"] = pairs[i][0]
        rec[f"ssr_{locus}_a2"] = pairs[i][1]
    return rec


def main():
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    records: list[dict] = []
    skipped = 0

    with pdfplumber.open(PDF_PATH) as pdf:
        for pg_idx, page in enumerate(pdf.pages):
            chars = page.chars
            row_map: dict[float, list] = defaultdict(list)
            for c in chars:
                y = round(c["top"], 0)
                row_map[y].append(c)

            pg_records = 0
            for y in sorted(row_map.keys()):
                if y <= 128:
                    continue    # skip header
                rc = sorted(row_map[y], key=lambda c: c["x0"])
                text = "".join(c["text"] for c in rc)
                rec = parse_row(text)
                if rec:
                    records.append(rec)
                    pg_records += 1
                else:
                    skipped += 1

            print(f"  Page {pg_idx+1:>2}: {pg_records} parsed")

    print(f"\nTotal parsed: {len(records)}, skipped: {skipped}")

    # Build raw CSV
    df = pd.DataFrame(records)
    df.to_csv(OUT_RAW, index=False)
    print(f"Raw CSV saved → {OUT_RAW}")

    # Build VIVC-keyed CSV (prime_name lookup via vivc_no)
    supp = pd.read_csv(
        BASE_DIR / "data" / "vivc_supplementary.csv",
        dtype=str, low_memory=False,
    )
    vivc_no_to_name: dict[str, str] = {}
    for _, row in supp.iterrows():
        vn = str(row.get("vivc_no", "")).strip()
        pn = str(row.get("prime_name", "")).strip()
        if vn and pn:
            vivc_no_to_name[vn] = pn.upper()

    df["vivc_prime_name"] = df["vivc_no"].map(vivc_no_to_name)
    matched = df[df["vivc_prime_name"].notna()]
    print(f"Matched to VIVC prime name: {len(matched)} / {len(df)}")

    # Build the GENRES081-overlap keyed table
    vivc_rows = []
    for _, row in matched.iterrows():
        vivc_row = {"vivc_prime_name": row["vivc_prime_name"]}
        for locus in GENRES081_OVERLAP:
            a1 = row.get(f"ssr_{locus}_a1", "")
            a2 = row.get(f"ssr_{locus}_a2", "")
            if a1 and a2 and a1 not in ("-", "0") and a2 not in ("-", "0"):
                vivc_row[f"ssr_{locus}"] = f"{a1}/{a2}"
            else:
                vivc_row[f"ssr_{locus}"] = ""
        vivc_row["ssr_source"] = "Lacombe et al. 2013"
        vivc_rows.append(vivc_row)

    vivc_df = pd.DataFrame(vivc_rows)
    vivc_df.to_csv(OUT_VIVC, index=False)
    print(f"VIVC-keyed CSV saved → {OUT_VIVC}")
    print(vivc_df.head(5).to_string())

    # ── Update node_molecular_profile.csv ─────────────────────────────────────
    profile_path = BASE_DIR / "data" / "node_molecular_profile.csv"
    profile = pd.read_csv(profile_path)
    existing_names = set(profile["vivc_prime_name"].str.strip().str.upper())

    # Only add rows NOT already in profile
    new_rows = vivc_df[~vivc_df["vivc_prime_name"].isin(existing_names)].copy()
    # Check they have at least one SSR value
    ssr_cols = [f"ssr_{l}" for l in GENRES081_OVERLAP]
    has_data = new_rows[ssr_cols].apply(lambda row: any(v for v in row), axis=1)
    new_rows = new_rows[has_data]

    print(f"\nNew SSR profile rows to add: {len(new_rows)}")

    # Also backfill SSR data for existing profile rows that lack it
    profile_up = profile["vivc_prime_name"].str.strip().str.upper()
    vivc_df_up = vivc_df.copy()
    vivc_df_up["vivc_prime_name"] = vivc_df_up["vivc_prime_name"].str.strip().str.upper()

    ssr_profile_cols = [f"ssr_{l}" for l in GENRES081_OVERLAP]
    backfill_count = 0
    for col in ssr_profile_cols:
        if col not in profile.columns:
            profile[col] = ""
        # Find profile rows missing this column's value
        profile[col] = profile[col].fillna("").astype(str).str.strip()
        missing_mask = profile[col] == ""
        # Build lookup from vivc_df_up
        fill_map = vivc_df_up.set_index("vivc_prime_name")[col].to_dict()
        filled = profile.loc[missing_mask, "vivc_prime_name"].str.strip().str.upper().map(fill_map)
        filled = filled.fillna("").astype(str)
        n = (filled != "").sum()
        backfill_count += n
        profile.loc[missing_mask, col] = filled.values

    print(f"Backfilled {backfill_count} SSR cells in existing profile rows")

    if len(new_rows) > 0:
        for col in profile.columns:
            if col not in new_rows.columns:
                new_rows[col] = ""
        new_rows = new_rows.reindex(columns=profile.columns, fill_value="")
        updated = pd.concat([profile, new_rows], ignore_index=True)
        updated.to_csv(profile_path, index=False)
        print(f"Updated node_molecular_profile.csv: rows +{len(new_rows)}, total {len(updated)}")
    else:
        profile.to_csv(profile_path, index=False)
        print(f"Profile saved with backfilled SSR data ({len(profile)} rows)")


if __name__ == "__main__":
    main()
