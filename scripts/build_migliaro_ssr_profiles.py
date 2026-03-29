"""
build_migliaro_ssr_profiles.py
Extracts SSR genotype profiles from Migliaro et al. 2019 (AJEV 70:390-397)
Supplemental Table 3 — 232 rootstock genotypes × 17 SSR loci.

doi: 10.5344/ajev.2019.18054

Output: data/ssr_migliaro_2019.csv
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
PDF_PATH = (
    BASE_DIR.parent
    / "VIVC App"
    / "Supplemental Literature Data"
    / "Supplemental_Table_3.pdf"
)
OUT_PATH = BASE_DIR / "data" / "ssr_migliaro_2019.csv"

# ── Loci (17 total, order from PDF header) ─────────────────────────────────────
LOCI = [
    "VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28",
    "VrZAG62", "VrZAG64", "VrZAG79", "VrZAG83",
    "VChr3a", "VChr4a", "VChr7b", "VChr10b", "VChr19a", "VMC6E1", "VMCNG4b9",
]
N_LOCI = len(LOCI)          # 17
N_ALLELES = N_LOCI * 2      # 34
ALLELE_STR_LEN = N_ALLELES * 3   # 102 (all alleles are 3-digit integers)


def parse_allele_string(allele_str: str) -> dict | None:
    """Split a 102-char allele string into per-locus a1/a2 values."""
    if len(allele_str) != ALLELE_STR_LEN:
        return None
    result = {}
    for i, locus in enumerate(LOCI):
        a1 = allele_str[i * 6 : i * 6 + 3]
        a2 = allele_str[i * 6 + 3 : i * 6 + 6]
        result[f"ssr_{locus}_a1"] = a1
        result[f"ssr_{locus}_a2"] = a2
    return result


def extract_rows_from_page(page) -> list[dict]:
    """
    Returns a list of dicts with keys: migliaro_id, variety_name, allele_str.

    PDF layout:
      - ID column starts at x ≈ 36 (1-3 digit sequential integer)
      - Variety name column starts at x ≈ 63
      - Allele columns follow to the right edge (~x 963)

    We split by x-position: chars x < 58 → ID; chars x ≥ 58 → name+allele text.
    The allele string is always the rightmost 102 digits of the name+allele text.
    Some long variety names wrap to a continuation line (x0 ≈ 63-64).
    """
    chars = page.chars
    row_map: dict[float, list] = defaultdict(list)
    for c in chars:
        y = round(c["top"], 0)
        row_map[y].append(c)

    # Build ordered list of (y, x0, id_text, rest_text) tuples
    lines = []
    for y in sorted(row_map.keys()):
        rc = sorted(row_map[y], key=lambda c: c["x0"])
        id_chars   = [c for c in rc if c["x0"] < 58]
        rest_chars = [c for c in rc if c["x0"] >= 58]
        id_text   = "".join(c["text"] for c in id_chars).strip()
        rest_text = "".join(c["text"] for c in rest_chars).strip()
        x0 = rc[0]["x0"]
        lines.append((y, x0, id_text, rest_text))

    records: list[dict] = []
    pending: dict | None = None

    for _y, x0, id_text, rest_text in lines:
        if not id_text and not rest_text:
            continue

        # Skip header / title / footer lines
        combined = (id_text + " " + rest_text).strip()
        if any(kw in combined for kw in (
            "Supplemental", "SSR ", "profile", "Allelic", "Online",
            "Cultivar prime", "Am. J.", "—",
        )):
            continue

        # --- Data row: id_text is a pure integer ---
        if id_text and re.match(r"^\d+$", id_text):
            # Flush previous record
            if pending is not None:
                records.append(pending)

            # Extract allele string (last 102 digits) from rest_text
            m_alleles = re.search(r"(\d{102})$", rest_text)
            if not m_alleles:
                # Retry after removing spaces embedded in the numeric suffix
                # (rare PDF artefact; preserve spaces in variety name)
                # Find where the trailing digit block starts
                m_start = re.search(r"(\d[\d\s]+\d)$", rest_text)
                if m_start:
                    numeric_block = re.sub(r"\s+", "", m_start.group(1))
                    rest_text = rest_text[: m_start.start()] + numeric_block
                    m_alleles = re.search(r"(\d{102})$", rest_text)
            if not m_alleles:
                pending = None
                continue

            allele_str = m_alleles.group(1)
            variety_name = rest_text[: m_alleles.start()].strip()
            pending = {
                "migliaro_id": id_text,
                "variety_name": variety_name,
                "allele_str": allele_str,
            }
            continue

        # --- Continuation line: no id_text, rest_text is variety name overflow ---
        if not id_text and pending is not None:
            # Skip footnote markers: purely numeric refs or isolated words like "and"
            skip_footnote = (
                re.match(r"^\d+\s*$", rest_text)          # bare number reference
                or re.match(r"^and\s+\d+", rest_text)     # "and 174" style footnote
                or len(rest_text) <= 3                     # too short to be a name
            )
            if rest_text and not re.match(r"^\d{6,}$", rest_text) and not skip_footnote:
                pending["variety_name"] = (
                    (pending["variety_name"] + " " + rest_text).strip()
                )
            continue

        # Flush on anything else
        if pending is not None:
            records.append(pending)
            pending = None

    if pending is not None:
        records.append(pending)

    return records


def main():
    if not PDF_PATH.exists():
        sys.exit(f"PDF not found: {PDF_PATH}")

    all_records: list[dict] = []
    with pdfplumber.open(PDF_PATH) as pdf:
        for pg_idx, page in enumerate(pdf.pages):
            rows = extract_rows_from_page(page)
            print(f"  Page {pg_idx+1}: {len(rows)} varieties parsed")
            all_records.extend(rows)

    print(f"\nTotal raw records: {len(all_records)}")

    # Build DataFrame
    out_rows = []
    parse_errors = 0
    for rec in all_records:
        alleles = parse_allele_string(rec["allele_str"])
        if alleles is None:
            print(f"  PARSE ERROR: id={rec['migliaro_id']} len={len(rec['allele_str'])}")
            parse_errors += 1
            continue
        row = {
            "migliaro_id": rec["migliaro_id"],
            "variety_name": rec["variety_name"],
        }
        row.update(alleles)
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    print(f"Parse errors: {parse_errors}")
    print(f"Clean records: {len(df)}")

    if df.empty:
        sys.exit("No records extracted.")

    # Deduplicate by migliaro_id (keep first)
    dups = df[df.duplicated("migliaro_id", keep=False)]
    if not dups.empty:
        print(f"Duplicate migliaro_ids: {dups['migliaro_id'].tolist()}")
    df = df.drop_duplicates("migliaro_id", keep="first")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows → {OUT_PATH}")
    print(df[["migliaro_id", "variety_name", "ssr_VVS2_a1", "ssr_VVS2_a2",
              "ssr_VVMD5_a1", "ssr_VVMD5_a2"]].head(10).to_string())


if __name__ == "__main__":
    main()
