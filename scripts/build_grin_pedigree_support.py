"""
build_grin_pedigree_support.py
-------------------------------
Derives a new pedigree marker-support layer from GRIN SNP admixture data.

For every VIVC parentage trio (child, parent1, parent2) where all three members
appear in the GRIN alignment table, we test whether the child's five ancestry
proportions are consistent with a 50:50 mixture of the two parents (the genetic
expectation for a cross).  This produces an "ancestry consistency score" (ACS)
that serves as independent SNP-based support for the pedigree, complementary to
the existing SSR and Axiom Vitis22K records.

Ancestry consistency score (ACS, 0–1)
--------------------------------------
ACS = 1 − mean_absolute_deviation(child_obs, child_expected) / 100

  child_expected[k] = 0.5 × parent1[k] + 0.5 × parent2[k]   for each component k

ACS = 1.0  → perfect mixture match
ACS ≥ 0.95 → strong support (deviation < 5 pp across all 5 components)
ACS ≥ 0.90 → moderate support
ACS < 0.80 → conflicting (ancestry inconsistent with claimed parentage)

The script also flags cases where all three members share the same GRIN
accession confirmation methods (multi-method confidence) and where the
VIVC pedigree is already corroborated by SSR/SNP records in the existing
marker support files.

Output
------
data/grin_pedigree_support.csv  — one row per validated trio
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
ALN_PATH     = BASE / "data" / "grin_vivc_alignment.csv"
VIVC_PATH    = BASE / "vivc_passport_table.csv"
MARKER_PATHS = [
    BASE / "data" / "marker_support.csv",
    BASE / "data" / "vivc_confirmed_marker_support.csv",
    BASE / "data" / "constantini_2026_marker_support.csv",
]
OUT_PATH     = BASE / "data" / "grin_pedigree_support.csv"

ANC_COLS = ["Vv_ancestry_pct", "NA1_ancestry_pct",
            "NA2_ancestry_pct", "Mus_ancestry_pct", "EA_ancestry_pct"]

# ── Load alignment ──────────────────────────────────────────────────────────
print("Loading GRIN–VIVC alignment…")
aln = pd.read_csv(ALN_PATH)
# Index by VIVC prime name (upper); keep best QC row per variety
aln["_key"] = aln["vivc_prime_name"].str.strip().str.upper()
aln_idx = (aln
           .sort_values("passQC", ascending=False)
           .drop_duplicates(subset=["_key"], keep="first")
           .set_index("_key"))
print(f"  {len(aln_idx):,} VIVC varieties with GRIN ancestry data")

# ── Load VIVC passport ──────────────────────────────────────────────────────
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
vivc["_key"]    = vivc["prime_name"].str.strip().str.upper()
vivc["_parent1"] = vivc["parent1"].fillna("").str.strip().str.upper()
vivc["_parent2"] = vivc["parent2"].fillna("").str.strip().str.upper()

# Trios with both parents known
trios = vivc[(vivc["_parent1"] != "") & (vivc["_parent2"] != "")].copy()
print(f"  {len(trios):,} VIVC entries with both parents recorded")

# ── Load existing marker support ────────────────────────────────────────────
print("Loading existing marker support files…")
ms_frames = []
for p in MARKER_PATHS:
    if p.exists():
        df = pd.read_csv(p, dtype=str, low_memory=False, keep_default_na=False)
        ms_frames.append(df)
ms = pd.concat(ms_frames, ignore_index=True)
ms["_child_key"]  = ms["child_variety"].str.strip().str.upper()
ms["_parent_key"] = ms["parent_variety"].str.strip().str.upper()
# Build a set of (child, parent) pairs already in existing support
existing_pairs = set(zip(ms["_child_key"], ms["_parent_key"]))
print(f"  {len(existing_pairs):,} child–parent pairs in existing marker support")

# ── Ancestry consistency scoring ───────────────────────────────────────────
print("Scoring ancestry consistency for GRIN-covered trios…")

aln_keys = set(aln_idx.index)
results: list[dict] = []

for _, trio in trios.iterrows():
    child   = trio["_key"]
    parent1 = trio["_parent1"]
    parent2 = trio["_parent2"]

    if child not in aln_keys or parent1 not in aln_keys or parent2 not in aln_keys:
        continue   # not all three are in the GRIN alignment

    c_row = aln_idx.loc[child]
    p1_row = aln_idx.loc[parent1]
    p2_row = aln_idx.loc[parent2]

    # Extract ancestry vectors
    c  = np.array([c_row[k]  for k in ANC_COLS], dtype=float)
    p1 = np.array([p1_row[k] for k in ANC_COLS], dtype=float)
    p2 = np.array([p2_row[k] for k in ANC_COLS], dtype=float)

    # Expected child ancestry under 50:50 mixture
    expected = 0.5 * (p1 + p2)
    deviation = np.abs(c - expected)
    mean_dev  = float(deviation.mean())
    max_dev   = float(deviation.max())
    acs       = round(1.0 - mean_dev / 100.0, 4)

    # Support category
    if acs >= 0.95:
        support_category = "strong"
    elif acs >= 0.90:
        support_category = "moderate"
    elif acs >= 0.80:
        support_category = "weak"
    else:
        support_category = "conflicting"

    # Is this trio already in existing marker support?
    already_p1 = (child, parent1) in existing_pairs
    already_p2 = (child, parent2) in existing_pairs
    existing_support = (
        "SSR/SNP+ancestry" if (already_p1 or already_p2) else "ancestry_only"
    )

    # Multi-method confidence of the three alignment members
    def _n_methods(key):
        return int(aln_idx.loc[key, "n_confirming_methods"])
    min_conf = min(_n_methods(child), _n_methods(parent1), _n_methods(parent2))

    results.append({
        # Trio identity
        "child":    trio["prime_name"],
        "parent1":  trio["parent1"],
        "parent2":  trio["parent2"],
        # VIVC metadata
        "vivc_number":      trio["vivc_number"],
        "vivc_color":       trio["color"],
        "vivc_species":     trio["species"],
        "vivc_pedigree_confirmed_markers": trio["pedigree_confirmed_markers"],
        # Child ancestry
        "child_Vv_pct":   float(c_row["Vv_ancestry_pct"]),
        "child_NA1_pct":  float(c_row["NA1_ancestry_pct"]),
        "child_NA2_pct":  float(c_row["NA2_ancestry_pct"]),
        "child_Mus_pct":  float(c_row["Mus_ancestry_pct"]),
        "child_EA_pct":   float(c_row["EA_ancestry_pct"]),
        "child_passQC":   c_row["passQC"],
        # Parent 1 ancestry
        "p1_Vv_pct":   float(p1_row["Vv_ancestry_pct"]),
        "p1_NA1_pct":  float(p1_row["NA1_ancestry_pct"]),
        "p1_NA2_pct":  float(p1_row["NA2_ancestry_pct"]),
        "p1_Mus_pct":  float(p1_row["Mus_ancestry_pct"]),
        "p1_EA_pct":   float(p1_row["EA_ancestry_pct"]),
        "p1_passQC":   p1_row["passQC"],
        # Parent 2 ancestry
        "p2_Vv_pct":   float(p2_row["Vv_ancestry_pct"]),
        "p2_NA1_pct":  float(p2_row["NA1_ancestry_pct"]),
        "p2_NA2_pct":  float(p2_row["NA2_ancestry_pct"]),
        "p2_Mus_pct":  float(p2_row["Mus_ancestry_pct"]),
        "p2_EA_pct":   float(p2_row["EA_ancestry_pct"]),
        "p2_passQC":   p2_row["passQC"],
        # Expected vs observed deviation
        "expected_Vv_pct":   round(float(expected[0]), 2),
        "expected_NA1_pct":  round(float(expected[1]), 2),
        "expected_NA2_pct":  round(float(expected[2]), 2),
        "expected_Mus_pct":  round(float(expected[3]), 2),
        "expected_EA_pct":   round(float(expected[4]), 2),
        "mean_abs_deviation_pp": round(mean_dev, 2),
        "max_abs_deviation_pp":  round(max_dev, 2),
        # Scoring
        "ancestry_consistency_score": acs,
        "support_category":           support_category,
        "existing_marker_support":    existing_support,
        # Alignment confidence
        "child_match_methods":   c_row["match_methods"],
        "p1_match_methods":      p1_row["match_methods"],
        "p2_match_methods":      p2_row["match_methods"],
        "min_confirming_methods": min_conf,
        "child_ambiguous":       c_row["ambiguous_flag"],
        "p1_ambiguous":          p1_row["ambiguous_flag"],
        "p2_ambiguous":          p2_row["ambiguous_flag"],
    })

out = pd.DataFrame(results)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("GRIN Ancestry Pedigree Support — Summary")
print(f"{'='*60}")
print(f"VIVC trios with both parents known        : {len(trios):,}")
print(f"Trios where all 3 members have GRIN data  : {len(out):,}")
print()
print("Support category breakdown:")
cats = out["support_category"].value_counts()
for cat, n in cats.items():
    print(f"  {cat:<15} {n:>5}  ({n/len(out)*100:.1f}%)")

print()
already = (out["existing_marker_support"] == "SSR/SNP+ancestry").sum()
print(f"Overlap with existing SSR/SNP support     : {already:,}")
new_only = (out["existing_marker_support"] == "ancestry_only").sum()
print(f"New ancestry-only evidence                : {new_only:,}")

strong = out[out["support_category"].isin(["strong", "moderate"])]
strong_new = strong[strong["existing_marker_support"] == "ancestry_only"]
print(f"Strong/moderate NEW ancestry support      : {len(strong_new):,}")

print()
print("Top 25 trios by ancestry consistency score:")
cols = ["child", "parent1", "parent2", "ancestry_consistency_score",
        "support_category", "mean_abs_deviation_pp", "existing_marker_support"]
print(out.nlargest(25, "ancestry_consistency_score")[cols].to_string(index=False))

print()
print("Potentially conflicting trios (ACS < 0.80):")
conf = out[out["support_category"] == "conflicting"].sort_values("ancestry_consistency_score")
print(conf[cols].head(20).to_string(index=False))

print(f"\nSaving → {OUT_PATH}")
out.to_csv(OUT_PATH, index=False)
print("Done.")
