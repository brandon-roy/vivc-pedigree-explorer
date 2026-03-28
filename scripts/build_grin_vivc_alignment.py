"""
build_grin_vivc_alignment.py
----------------------------
Constructs a GRIN accession → VIVC prime_name alignment table using five
complementary confirmatory methods, in descending confidence order.

Methods
-------
M1  Exact case-insensitive match          (GRIN_PLANT NAME == prime_name)
M2  Alias table bridge                    (GRIN name maps via marker_name_aliases)
M3  Umlaut-expansion normalisation        (Ö→OE, Ü→UE, Ä→AE applied to both sides)
M4  Hybrid-series spacing normalisation   (strip/collapse spaces around hyphens)
M5  High-confidence fuzzy match           (difflib SequenceMatcher ratio ≥ 0.92)

For each GRIN accession the first method that produces a single unambiguous
VIVC match is recorded.  Multi-method agreement is tracked in `match_methods`.

Origin + taxonomy cross-validation is applied after all methods to flag
potentially ambiguous matches (same name but conflicting country / species).

Output
------
data/grin_vivc_alignment.csv  — one row per aligned GRIN accession × VIVC variety
"""

from __future__ import annotations

import re
import difflib
import unicodedata
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
GRIN_PATH  = BASE / "data" / "DVIT_GVIT_BI_sample_info_summary_cleaned.xlsx"
VIVC_PATH  = BASE / "vivc_passport_table.csv"
ALIAS_PATH = BASE / "data" / "marker_name_aliases.csv"
OUT_PATH   = BASE / "data" / "grin_vivc_alignment.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _expand_umlauts(s: str) -> str:
    """Ö→OE, Ü→UE, Ä→AE (upper); strip remaining diacritics via NFD."""
    s = str(s)   # guard against float NaN
    s = (s
         .replace("Ö", "OE").replace("ö", "OE")
         .replace("Ü", "UE").replace("ü", "UE")
         .replace("Ä", "AE").replace("ä", "AE"))
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _norm_hybrid(s: str) -> str:
    """Collapse whitespace around hyphens: 'SV 12- 401' → 'SV 12-401'."""
    return re.sub(r"\s*-\s*", "-", str(s))


# ══════════════════════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════════════════════
print("Loading GRIN genotype file…")
grin_raw = pd.read_excel(GRIN_PATH)

# De-duplicate: one row per unique GRIN accession, keeping pass-QC row first
grin = (grin_raw
        .sort_values("passQC", ascending=False)
        .drop_duplicates(subset=["GRIN_accessionID"], keep="first")
        .reset_index(drop=True))
grin = grin[grin["GRIN_PLANT NAME"].notna()].copy()
grin["_grin_name"] = grin["GRIN_PLANT NAME"].str.strip().str.upper()
print(f"  {len(grin):,} unique GRIN accessions with plant names")

# ── VIVC passport ──────────────────────────────────────────────────────────
print("Loading VIVC passport…")
vivc_raw = pd.read_csv(VIVC_PATH, dtype=str, low_memory=False)
vivc_raw.columns = [
    "prime_name", "color", "vivc_number", "utilization",
    "country_origin", "species", "parent1", "parent2",
    "pedigree_confirmed_markers", "breeder", "year_crossing",
    "accessions", "page_scraped", "source_url",
]
vivc = vivc_raw[
    vivc_raw["prime_name"].notna() & (vivc_raw["prime_name"] != "Prime name")
].copy()
vivc["_vivc_name"] = vivc["prime_name"].str.strip().str.upper()
# Keep first occurrence per name for cross-validation lookup
vivc_lookup = vivc.drop_duplicates(subset=["_vivc_name"], keep="first").set_index("_vivc_name")
vivc_set = set(vivc_lookup.index)
print(f"  {len(vivc_set):,} unique VIVC prime names")

# ── Alias table ────────────────────────────────────────────────────────────
print("Loading alias table…")
aliases = pd.read_csv(ALIAS_PATH, dtype=str).fillna("")
alias_map: dict[str, str] = {
    row["alias"].strip().upper(): row["canonical_vivc_name"].strip().upper()
    for _, row in aliases.iterrows()
    if row["alias"].strip() and row["canonical_vivc_name"].strip()
}
print(f"  {len(alias_map):,} alias entries")

# Pre-build normalised VIVC lookup dictionaries
# Each maps a normalised key → original VIVC name
vivc_umlaut = {_expand_umlauts(n): n for n in vivc_set}
vivc_hybrid  = {_norm_hybrid(n): n for n in vivc_set}
vivc_both    = {_norm_hybrid(_expand_umlauts(n)): n for n in vivc_set}

# difflib lists (lowercase)
vivc_list_lc = [n.lower() for n in vivc_set]
vivc_list_uc = list(vivc_set)

FUZZY_THRESHOLD = 0.92


# ══════════════════════════════════════════════════════════════════════════════
# Alignment engine — five methods applied in order
# ══════════════════════════════════════════════════════════════════════════════
print("Running alignment…")
results: list[dict] = []

for _, row in grin.iterrows():
    gname = row["_grin_name"]
    if not isinstance(gname, str) or not gname.strip():
        continue
    match: str | None = None
    methods: list[str] = []

    # ── M1: Exact match ───────────────────────────────────────────────────
    if gname in vivc_set:
        match = gname
        methods.append("M1_exact")

    # ── M2: Alias table bridge ────────────────────────────────────────────
    if gname in alias_map:
        canonical = alias_map[gname]
        if canonical in vivc_set:
            if match is None:
                match = canonical
            if canonical == match:
                methods.append("M2_alias")

    # ── M3: Umlaut-expansion normalisation ───────────────────────────────
    g_uml = _expand_umlauts(gname)
    if g_uml in vivc_umlaut:
        m3 = vivc_umlaut[g_uml]
        if match is None:
            match = m3
        if m3 == match:
            methods.append("M3_umlaut")
    # Also: normalise the VIVC side (GRIN keeps native letters, VIVC expands)
    if gname in vivc_both:
        m3b = vivc_both[gname]
        if match is None:
            match = m3b
        if m3b == match and "M3_umlaut" not in methods:
            methods.append("M3_umlaut")

    # ── M4: Hybrid-series spacing normalisation ───────────────────────────
    g_hyb = _norm_hybrid(gname)
    if g_hyb in vivc_hybrid:
        m4 = vivc_hybrid[g_hyb]
        if match is None:
            match = m4
        if m4 == match:
            methods.append("M4_hybrid_spacing")
    g_both = _norm_hybrid(g_uml)
    if g_both in vivc_both:
        m4b = vivc_both[g_both]
        if match is None:
            match = m4b
        if m4b == match and "M4_hybrid_spacing" not in methods:
            methods.append("M4_hybrid_spacing")

    # ── M5: High-confidence fuzzy match ───────────────────────────────────
    if match is None:
        candidates = difflib.get_close_matches(
            gname.lower(), vivc_list_lc, n=1, cutoff=FUZZY_THRESHOLD
        )
        if candidates:
            idx   = vivc_list_lc.index(candidates[0])
            ratio = difflib.SequenceMatcher(
                None, gname.lower(), candidates[0]
            ).ratio()
            m5 = vivc_list_uc[idx]
            match = m5
            methods.append(f"M5_fuzzy(r={ratio:.3f})")

    if match is None:
        continue

    # ── Cross-validation: origin + taxonomy ───────────────────────────────
    vr = vivc_lookup.loc[match] if match in vivc_lookup.index else None

    origin_cv  = None
    species_cv = None
    ambiguous  = False

    if vr is not None:
        grin_origin = str(row.get("GRIN_ORIGIN", "")).strip().upper()
        vivc_origin = str(vr.get("country_origin", "")).strip().upper()
        if grin_origin not in ("", "NAN") and vivc_origin not in ("", "NAN"):
            origin_cv = "OK" if (
                grin_origin in vivc_origin or vivc_origin in grin_origin
            ) else "MISMATCH"

        grin_tax = str(row.get("GRIN_TAXONOMY", "")).strip().lower()
        vivc_sp  = str(vr.get("species", "")).strip().lower()
        if grin_tax not in ("", "nan") and vivc_sp not in ("", "nan"):
            grin_vv = "vinifera" in grin_tax
            vivc_vv = "vinifera" in vivc_sp
            species_cv = "OK" if grin_vv == vivc_vv else "MISMATCH"
            if species_cv == "MISMATCH":
                ambiguous = True

        if (vivc["_vivc_name"] == match).sum() > 1:
            ambiguous = True   # homonym in VIVC

    results.append({
        # GRIN identity
        "GRIN_accessionID":        row["GRIN_accessionID"],
        "GRIN_ACCESSION":          row["GRIN_ACCESSION"],
        "GRIN_PLANT_NAME":         row["GRIN_PLANT NAME"],
        "GRIN_TAXONOMY":           row.get("GRIN_TAXONOMY"),
        "GRIN_GENEBANK":           row.get("GRIN_GENEBANK"),
        "GRIN_ORIGIN":             row.get("GRIN_ORIGIN"),
        "GRIN_DbID":               row.get("GRIN_DbID"),
        # Genotype QC
        "passQC":                  row.get("passQC"),
        "missingness_rate":        row.get("missingness_rate"),
        # Ancestry proportions
        "Vv_ancestry_pct":         row.get("Vv_ancestry(%)"),
        "NA1_ancestry_pct":        row.get("NA1_ancestry(%)"),
        "NA2_ancestry_pct":        row.get("NA2_ancestry(%)"),
        "Mus_ancestry_pct":        row.get("Mus_ancestry(%)"),
        "EA_ancestry_pct":         row.get("EA_ancestry(%)"),
        # VIVC match
        "vivc_prime_name":         match,
        "vivc_number":             vr["vivc_number"] if vr is not None else None,
        "vivc_country_origin":     vr["country_origin"] if vr is not None else None,
        "vivc_species":            vr["species"] if vr is not None else None,
        "vivc_color":              vr["color"] if vr is not None else None,
        "vivc_parent1":            vr["parent1"] if vr is not None else None,
        "vivc_parent2":            vr["parent2"] if vr is not None else None,
        "vivc_pedigree_confirmed": vr["pedigree_confirmed_markers"] if vr is not None else None,
        # Alignment metadata
        "match_methods":           "|".join(methods),
        "n_confirming_methods":    len({m.split("(")[0] for m in methods}),
        "origin_crossval":         origin_cv,
        "species_crossval":        species_cv,
        "ambiguous_flag":          ambiguous,
    })

# ══════════════════════════════════════════════════════════════════════════════
# Output & summary
# ══════════════════════════════════════════════════════════════════════════════
out = pd.DataFrame(results)

print(f"\n{'='*60}")
print("GRIN → VIVC Alignment Summary")
print(f"{'='*60}")
print(f"Unique GRIN accessions (with plant name) : {len(grin):,}")
print(f"Successfully aligned to VIVC             : {len(out):,}  ({len(out)/len(grin)*100:.1f}%)")
print(f"Unmatched                                : {len(grin)-len(out):,}")

method_counts: dict[str, int] = {}
for s in out["match_methods"]:
    for m in s.split("|"):
        k = m.split("(")[0]
        method_counts[k] = method_counts.get(k, 0) + 1

print("\nContributions per method:")
for m, c in sorted(method_counts.items()):
    print(f"  {m:<35} {c:>5}")

multi = out[out["n_confirming_methods"] >= 2]
print(f"\nMulti-method confirmed (≥2 methods)      : {len(multi):,}  ({len(multi)/len(out)*100:.1f}%)")

o_ok  = (out["origin_crossval"] == "OK").sum()
o_mis = (out["origin_crossval"] == "MISMATCH").sum()
print(f"Origin cross-validation: {o_ok} agree, {o_mis} mismatch")
s_mis = (out["species_crossval"] == "MISMATCH").sum()
print(f"Species cross-validation mismatches      : {s_mis}")
print(f"Ambiguous flags                          : {out['ambiguous_flag'].sum()}")

print(f"\nSaving → {OUT_PATH}")
out.to_csv(OUT_PATH, index=False)
print("Done.")
