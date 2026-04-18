"""
build_cipriani_2010_pedigree.py
================================
Parse MOESM4 (pedigree table) from Cipriani et al. 2010 TAG 121:1569-1585
and write data/cipriani_2010_marker_support.csv in standard long format.

Three sections in the table:
  • "confirmed pedigrees"          — existing pedigrees validated by this study
  • "revised or completed"         — pedigrees corrected vs. breeder/VIVC records
  • "new pedigrees"                — first-time discoveries

Names in the file are formatted as "VARIETY_NAME (accession_code)"; we strip the
accession number, replace underscores with spaces, and resolve to VIVC prime names.
"""

import re
import pathlib
import unicodedata
import xlrd
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = pathlib.Path(__file__).resolve().parents[1]
XLS_PED   = BASE_DIR / "data" / "Cipriani et al 2010" / "122_2010_1411_MOESM4_ESM.xls"
XLS_NAMES = BASE_DIR / "data" / "Cipriani et al 2010" / "122_2010_1411_MOESM1_ESM.xls"
MAS_PATH  = BASE_DIR / "data" / "master_accession_sheet.csv"
OUT_PATH  = BASE_DIR / "data" / "cipriani_2010_marker_support.csv"

DOI    = "10.1007/s00122-010-1411-9"
YEAR   = 2010
STUDY  = "Cipriani et al. 2010"

# ── Name normalisation ────────────────────────────────────────────────────────
def norm(s: str) -> str:
    s = str(s).strip().upper()
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s)

def parse_cipriani_name(raw: str) -> tuple[str, int | None]:
    """'ALICANTE_BOUSCHET (12)' → ('ALICANTE BOUSCHET', 12)"""
    raw = str(raw).strip()
    m = re.match(r"^(.+?)\s*\((\d+)\)\s*$", raw)
    if m:
        name = m.group(1).replace("_", " ").strip()
        acc  = int(m.group(2))
    else:
        name = raw.replace("_", " ").strip()
        acc  = None
    return name, acc

# ── Build VIVC name lookup from master accession sheet ────────────────────────
print("Loading VIVC name list…")
mas = pd.read_csv(MAS_PATH, low_memory=False)
vivc_norm: dict[str, str] = {}
for _, row in mas.iterrows():
    prime = str(row.get("vivc_prime_name") or "").strip()
    if prime:
        vivc_norm[norm(prime)] = prime

def resolve(name: str) -> str:
    n = norm(name)
    return vivc_norm.get(n, name.upper())

# ── Parse MOESM4 pedigree table ───────────────────────────────────────────────
print(f"Reading {XLS_PED.name}…")
wb  = xlrd.open_workbook(str(XLS_PED))
sh  = wb.sheet_by_index(0)

SECTION_MARKERS = {
    "confirmed pedigrees":            "confirmed",
    "revised or completed pedigrees": "revised",
    "new pedigrees":                  "new",
}
current_section = "confirmed"   # default before first marker is seen

out_rows   = []
section_counts = {"confirmed": 0, "revised": 0, "new": 0}

for i in range(1, sh.nrows):
    row = sh.row_values(i)

    # Check for section header row
    cell0 = str(row[0] or "").strip().lower()
    for marker, sec in SECTION_MARKERS.items():
        if marker in cell0:
            current_section = sec
            break
    else:
        # Skip if offspring cell is empty or looks like a header
        offspring_raw = str(row[0] or "").strip()
        if not offspring_raw or offspring_raw.lower() in ("offspring", ""):
            continue

        child_name, child_acc = parse_cipriani_name(offspring_raw)
        p1_name,    _         = parse_cipriani_name(str(row[1] or ""))
        p2_name,    _         = parse_cipriani_name(str(row[2] or ""))

        # Skip degenerate rows
        if not p1_name.strip() or not p2_name.strip():
            continue

        # n_markers: use "further loci" count if available, else primary loci
        n_primary  = row[3] if row[3] != "" else None
        n_further  = row[6] if len(row) > 6 and row[6] != "" else None
        try:
            n_markers = int(n_further if n_further else n_primary)
        except (TypeError, ValueError):
            n_markers = None

        # mismatches
        try:
            mm_primary = int(row[4]) if row[4] != "" else 0
            mm_further = int(row[7]) if len(row) > 7 and row[7] != "" else 0
            total_mm   = mm_primary + mm_further
        except (TypeError, ValueError):
            total_mm = 0

        # LOD score — stored as very large float (e.g. 3.08e15); convert safely
        try:
            lod_raw = float(row[5]) if row[5] != "" else None
            # Cipriani LOD values are Cervus-format; store as-is (log-odds × 10^14 approx)
            # For display use scientific notation
            lod = f"{lod_raw:.3e}" if lod_raw else None
        except (TypeError, ValueError):
            lod = None

        notes_raw = str(row[8] or "").strip() if len(row) > 8 else ""

        # Confidence level
        if current_section == "revised":
            confidence = "revised"   # custom tag — pedigree corrected vs prior record
        elif total_mm <= 1:
            confidence = "confirmed"
        elif total_mm <= 3:
            confidence = "probable"
        else:
            confidence = "low_confidence"

        # Novel flag
        is_novel = current_section == "new"
        novel_prefix = "first_molecular_confirmation; " if is_novel else ""
        section_label = {
            "confirmed": "Confirmed pedigree",
            "revised":   "Revised/corrected pedigree",
            "new":       "New pedigree discovery",
        }[current_section]

        child_v = resolve(child_name)
        p1_v    = resolve(p1_name)
        p2_v    = resolve(p2_name)

        note = (f"{novel_prefix}{section_label}; {notes_raw}; "
                f"Cipriani 2010 TAG 121:1569; {n_markers} SSR loci; "
                f"mismatches={total_mm}").strip("; ")

        base = dict(
            evidence_type    = "SSR",
            marker_type      = "SSR",
            n_markers        = n_markers,
            lod_score        = lod,
            study_reference  = STUDY,
            doi              = DOI,
            confirmed_year   = YEAR,
            confidence_level = confidence,
            notes            = note,
        )
        out_rows.append({**base, "child_variety": child_v,
                         "parent_variety": p1_v, "parent_role": "Parent 1"})
        out_rows.append({**base, "child_variety": child_v,
                         "parent_variety": p2_v, "parent_role": "Parent 2"})
        section_counts[current_section] += 1

print(f"Sections parsed: confirmed={section_counts['confirmed']}, "
      f"revised={section_counts['revised']}, new={section_counts['new']}")

# ── Write output ──────────────────────────────────────────────────────────────
df = pd.DataFrame(out_rows, columns=[
    "child_variety", "parent_variety", "parent_role",
    "evidence_type", "marker_type", "n_markers", "lod_score",
    "study_reference", "doi", "confirmed_year", "confidence_level", "notes",
])
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} rows → {OUT_PATH.relative_to(BASE_DIR)}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=== SUMMARY ===")
print(f"Total trios:  {len(df)//2}")
print(f"  Confirmed:  {section_counts['confirmed']}")
print(f"  Revised:    {section_counts['revised']}")
print(f"  New:        {section_counts['new']}")
print()
print("New pedigrees (first molecular confirmation):")
new_rows = df[(df["notes"].str.contains("first_molecular_confirmation")) &
              (df["parent_role"] == "Parent 1")]
for _, r in new_rows.iterrows():
    p2 = df[(df["child_variety"] == r["child_variety"]) &
            (df["parent_role"] == "Parent 2")]["parent_variety"].iloc[0]
    print(f"  {r['child_variety']} × {r['parent_variety']} × {p2}")

print()
print("Revised/corrected pedigrees:")
rev_rows = df[(df["notes"].str.contains("Revised")) &
              (df["parent_role"] == "Parent 1")]
for _, r in rev_rows.iterrows():
    p2 = df[(df["child_variety"] == r["child_variety"]) &
            (df["parent_role"] == "Parent 2")]["parent_variety"].iloc[0]
    print(f"  {r['child_variety']} × {r['parent_variety']} × {p2}")

# ── Cross-check against VIVC ──────────────────────────────────────────────────
vivc = pd.read_csv(BASE_DIR / "data" / "vivc_confirmed_marker_support.csv",
                   low_memory=False)
vivc_pairs  = set(zip(vivc["child_variety"].str.upper(),
                      vivc["parent_variety"].str.upper()))
our_pairs   = set(zip(df["child_variety"].str.upper(),
                      df["parent_variety"].str.upper()))
overlap     = our_pairs & vivc_pairs
print()
print(f"Cross-check vs VIVC: {len(overlap)}/{len(our_pairs)} pairs already in VIVC")
print(f"Cipriani-only (not yet in VIVC): {len(our_pairs - vivc_pairs)}")
