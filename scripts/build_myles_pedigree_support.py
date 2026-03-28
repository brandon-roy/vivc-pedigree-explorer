"""
Build pedigree support evidence from Myles et al. 2011 Vitis9KSNP array.

For each VIVC pedigree edge (child → parent) where both individuals are present
in the Myles 2011 dataset, this script computes an IBD Consistency Score (ICS)
using Mendelian-exclusion testing at homozygous-parent SNP loci:

    ICS = 1 − (N_Mendelian_exclusions / N_informative_loci)

where an "informative locus" is one where the proposed parent is homozygous and
neither individual has a missing call.  At such loci, the child must carry at
least one copy of the parent's allele — otherwise it is a Mendelian exclusion.

Expected ICS values
───────────────────
    True parent–offspring  : ≥ 0.97  (error rate ~ 1–3 %)
    Unrelated individuals  : ~ 0.50–0.75  (varies with allele frequencies)

Confidence mapping
──────────────────
    ICS ≥ 0.97  →  confirmed   (< 3 % exclusions, strong genomic support)
    ICS ≥ 0.93  →  probable    (< 7 % exclusions)
    ICS ≥ 0.85  →  probable    (< 15 %, weaker but above random)
    ICS <  0.85 →  disputed    (≥ 15 % exclusions)
    n_informative < MIN_INFO   →  skipped (insufficient data)

Name-matching pipeline (in priority order)
──────────────────────────────────────────
    1. Myles local_id (DVIT_XXX) → grin_vivc_alignment.csv → vivc_prime_name
    2. Myles cultivar name normalised (replace _ → space, uppercase)
       → direct match against VIVC passport prime_name set
    3. Fallback via marker_name_aliases.csv

Output
──────
    data/myles_2011_marker_support.csv  (12-column marker-support schema)

Run with:  python scripts/build_myles_pedigree_support.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
SUPP_DIR    = Path("/Users/brandonroy/Desktop/Informatique/Projects/VIVC App"
                   "/Supplemental Literature Data")

PED_PATH    = SUPP_DIR / "20110117_pnas_grape_rawdat/vitis9ksnp_finalfilt.ped"
MAP_PATH    = SUPP_DIR / "20110117_pnas_grape_rawdat/vitis9ksnp_finalfilt.map"
INFO_PATH   = SUPP_DIR / "20110117_pnas_grape_rawdat/sample_info.txt"

PASSPORT_PATH  = BASE_DIR / "vivc_passport_table.csv"
ALIAS_PATH     = BASE_DIR / "data" / "marker_name_aliases.csv"
GRIN_ALIGN_PATH = BASE_DIR / "data" / "grin_vivc_alignment.csv"
OUT_PATH       = BASE_DIR / "data" / "myles_2011_marker_support.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
N_SNPS        = 6114
DOI           = "10.1073/pnas.1009363108"
STUDY_REF     = "Myles et al. 2011 (PNAS)"
CONFIRMED_YEAR = "2011"
MIN_INFORMATIVE = 200   # minimum homozygous-parent loci required to emit a row

ICS_CONFIRMED = 0.97
ICS_PROBABLE  = 0.93
ICS_WEAK      = 0.85   # still "probable" but noted as lower-confidence

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalise a variety name: replace _ with space, strip, uppercase."""
    return re.sub(r"_+", " ", str(name or "")).strip().upper()


def load_alias_map() -> dict[str, str]:
    if not ALIAS_PATH.exists():
        return {}
    df = pd.read_csv(ALIAS_PATH, dtype=str).fillna("")
    if "alias" not in df.columns or "canonical_vivc_name" not in df.columns:
        return {}
    return {
        row["alias"].strip().upper(): row["canonical_vivc_name"].strip().upper()
        for _, row in df.iterrows()
        if row["alias"].strip()
    }


def build_dvit_to_vivc(alias_map: dict[str, str]) -> dict[str, str]:
    """
    Return {DVIT_XXX: vivc_prime_name} from grin_vivc_alignment.csv.
    Only includes rows where vivc_prime_name is non-null.
    """
    if not GRIN_ALIGN_PATH.exists():
        print("  [WARN] grin_vivc_alignment.csv not found — skipping DVIT lookup")
        return {}
    ga = pd.read_csv(GRIN_ALIGN_PATH, dtype=str, low_memory=False)
    dvit_map: dict[str, str] = {}
    for _, row in ga.iterrows():
        acc_id  = str(row.get("GRIN_accessionID", "") or "").strip()
        vivc_nm = str(row.get("vivc_prime_name",  "") or "").strip().upper()
        if acc_id and vivc_nm and vivc_nm not in {"NAN", "NONE", ""}:
            dvit_map[acc_id] = vivc_nm
    print(f"  DVIT → VIVC map: {len(dvit_map)} entries")
    return dvit_map


def load_vivc_passport() -> tuple[set[str], list[tuple[str, str, str]]]:
    """
    Parse the VIVC passport CSV.
    Returns:
        vivc_names : set of all VIVC prime_names (upper)
        edges      : list of (child, parent, role) triples
    """
    pp = pd.read_csv(PASSPORT_PATH, dtype=str, low_memory=False, keep_default_na=False)
    # The real header is in the first data row
    real_cols = [str(v).strip() for v in pp.iloc[0].values]
    pp.columns = [str(c).strip() for c in pp.columns]
    pp = pp.rename(columns=dict(zip(pp.columns, real_cols)))
    pp = pp.iloc[1:].reset_index(drop=True)
    pp.columns = [
        c.lower().replace(" ", "_").replace("-", "_")
         .replace("(", "").replace(")", "").replace("/", "_")
        for c in pp.columns
    ]

    col_name = next((c for c in pp.columns if c == "prime_name"), None)
    col_p1   = next((c for c in pp.columns if "parent_1" in c), None)
    col_p2   = next((c for c in pp.columns if "parent_2" in c), None)

    def _clean(s: str) -> str:
        s = str(s or "").strip().upper()
        return "" if s in {"", "NAN", "NONE", "NA", "N/A"} else s

    vivc_names: set[str] = set()
    edges: list[tuple[str, str, str]] = []
    for _, row in pp.iterrows():
        child = _clean(row.get(col_name, ""))
        p1    = _clean(row.get(col_p1, ""))
        p2    = _clean(row.get(col_p2, ""))
        if child:
            vivc_names.add(child)
        if child and p1:
            edges.append((child, p1, "Parent 1"))
        if child and p2:
            edges.append((child, p2, "Parent 2"))

    print(f"  VIVC varieties: {len(vivc_names):,}")
    print(f"  VIVC pedigree edges: {len(edges):,}")
    return vivc_names, edges


def map_myles_samples(
    dvit_to_vivc: dict[str, str],
    vivc_names: set[str],
    alias_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Build two maps from the Myles sample_info.txt:
        illum_to_vivc  : {illum_id → vivc_prime_name}
        vivc_to_illums : {vivc_prime_name → [illum_ids]}

    Matching priority:
        1. local_id (DVIT_XXX) → dvit_to_vivc
        2. normalised cultivar name → direct VIVC prime-name match
        3. normalised cultivar name → alias_map lookup
    """
    info = pd.read_csv(INFO_PATH, sep=r"\s+", dtype=str, engine="python").fillna("")
    # Keep only V. vinifera
    info = info[info["species"].str.lower().str.contains("vinifera", na=False)].copy()

    illum_to_vivc: dict[str, str]       = {}
    vivc_to_illums: dict[str, list[str]] = {}
    n_dvit, n_direct, n_alias, n_fail   = 0, 0, 0, 0

    for _, row in info.iterrows():
        iid      = str(row.get("illum_id", "")).strip()
        dvit_id  = str(row.get("local_id", "")).strip()   # e.g. "DVIT_677"
        cultivar = _norm(row.get("cultivar", ""))

        vivc_nm: str | None = None

        # Strategy 1 — DVIT lookup
        if dvit_id and dvit_id.upper().startswith("DVIT_") and dvit_id in dvit_to_vivc:
            vivc_nm = dvit_to_vivc[dvit_id]
            n_dvit += 1

        # Strategy 2 — direct cultivar name match
        elif cultivar and cultivar in vivc_names:
            vivc_nm = cultivar
            n_direct += 1

        # Strategy 3 — alias map
        elif cultivar and cultivar in alias_map:
            vivc_nm = alias_map[cultivar]
            n_alias += 1

        else:
            n_fail += 1

        if vivc_nm and iid:
            illum_to_vivc[iid] = vivc_nm
            vivc_to_illums.setdefault(vivc_nm, []).append(iid)

    print(f"  Myles name matching: DVIT={n_dvit}, direct={n_direct}, "
          f"alias={n_alias}, unmatched={n_fail}")
    print(f"  Unique VIVC varieties in Myles: {len(vivc_to_illums)}")
    return illum_to_vivc, vivc_to_illums


def load_ped_subset(needed_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """
    Scan the PED file and load genotype vectors for the requested illum_ids.
    Returns {illum_id: [(a1, a2), ...]} with length N_SNPS per entry.
    """
    genotypes: dict[str, list[tuple[str, str]]] = {}
    total = len(needed_ids)
    print(f"  Scanning PED for {total} samples …", end="", flush=True)
    with open(PED_PATH) as fh:
        for line in fh:
            parts = line.split()
            iid = parts[1]
            if iid not in needed_ids:
                continue
            alleles = parts[6:]
            genos = [
                (alleles[i * 2], alleles[i * 2 + 1])
                for i in range(N_SNPS)
            ]
            genotypes[iid] = genos
            if len(genotypes) == total:
                break  # found all we need
    print(f" loaded {len(genotypes)}/{total}")
    return genotypes


def pick_best_per_variety(
    vivc_to_illums: dict[str, list[str]],
    genotypes: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    """
    For each VIVC variety with multiple accessions in the dataset, select the
    single accession with the fewest missing genotype calls ('0').
    Returns {vivc_name → best_illum_id}.
    """
    best: dict[str, str] = {}
    for vivc_nm, illum_ids in vivc_to_illums.items():
        available = [iid for iid in illum_ids if iid in genotypes]
        if not available:
            continue
        best_iid = min(
            available,
            key=lambda iid: sum(1 for a1, _ in genotypes[iid] if a1 == "0"),
        )
        best[vivc_nm] = best_iid
    return best


def compute_ics(
    parent_genos: list[tuple[str, str]],
    child_genos:  list[tuple[str, str]],
) -> tuple[float | None, int, int]:
    """
    Compute IBD Consistency Score for a proposed parent–offspring pair.

    Returns (ics, n_informative, n_excluded).
    Returns (None, n_informative, 0) if n_informative < MIN_INFORMATIVE.
    """
    n_info = 0
    n_excl = 0
    for (p1, p2), (c1, c2) in zip(parent_genos, child_genos):
        if p1 == "0" or c1 == "0":
            continue                    # skip missing
        if p1 != p2:
            continue                    # parent is het → always compatible
        # Parent is homozygous — informative locus
        n_info += 1
        if p1 not in (c1, c2):         # child doesn't carry parent's allele
            n_excl += 1
    if n_info < MIN_INFORMATIVE:
        return None, n_info, 0
    return 1.0 - n_excl / n_info, n_info, n_excl


def ics_to_confidence(ics: float, n_info: int) -> str:
    if ics >= ICS_CONFIRMED and n_info >= 500:
        return "confirmed"
    if ics >= ICS_PROBABLE:
        return "probable"
    if ics >= ICS_WEAK:
        return "probable"   # weaker support, but still above random
    return "disputed"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== build_myles_pedigree_support.py ===\n")

    # 1. Load helper maps
    print("[1/6] Loading alias map and DVIT→VIVC map …")
    alias_map   = load_alias_map()
    dvit_to_vivc = build_dvit_to_vivc(alias_map)

    # 2. Load VIVC passport
    print("\n[2/6] Loading VIVC passport …")
    vivc_names, all_edges = load_vivc_passport()

    # 3. Map Myles samples → VIVC names
    print("\n[3/6] Matching Myles sample names → VIVC prime names …")
    illum_to_vivc, vivc_to_illums = map_myles_samples(
        dvit_to_vivc, vivc_names, alias_map
    )

    # 4. Find covered edges (both child and parent in Myles)
    print("\n[4/6] Finding covered pedigree edges …")
    covered = [
        (child, parent, role)
        for child, parent, role in all_edges
        if child in vivc_to_illums and parent in vivc_to_illums
    ]
    print(f"  Edges with both members in Myles dataset: {len(covered)}")
    if not covered:
        print("  No covered edges found — check name matching. Exiting.")
        sys.exit(0)

    # 5. Load PED genotypes for all needed samples
    print("\n[5/6] Loading PED genotypes …")
    needed_ids: set[str] = set()
    for child, parent, _ in covered:
        needed_ids.update(vivc_to_illums[child])
        needed_ids.update(vivc_to_illums[parent])
    genotypes = load_ped_subset(needed_ids)

    # For each variety, pick the best-quality (fewest missing) accession
    best_acc = pick_best_per_variety(vivc_to_illums, genotypes)

    # 6. Compute ICS and build output rows
    print("\n[6/6] Computing IBD Consistency Scores …")
    rows: list[dict] = []
    skipped_missing = 0
    skipped_low_n   = 0
    by_conf: dict[str, int] = {}

    for child, parent, role in covered:
        child_iid  = best_acc.get(child)
        parent_iid = best_acc.get(parent)
        if not child_iid or not parent_iid:
            skipped_missing += 1
            continue

        parent_gt = genotypes.get(parent_iid)
        child_gt  = genotypes.get(child_iid)
        if parent_gt is None or child_gt is None:
            skipped_missing += 1
            continue

        ics, n_info, n_excl = compute_ics(parent_gt, child_gt)
        if ics is None:
            skipped_low_n += 1
            continue

        conf = ics_to_confidence(ics, n_info)
        by_conf[conf] = by_conf.get(conf, 0) + 1

        rows.append({
            "child_variety":   child,
            "parent_variety":  parent,
            "parent_role":     role,
            "evidence_type":   "SNP array",
            "marker_type":     "Vitis9KSNP",
            "n_markers":       N_SNPS,
            "lod_score":       np.nan,
            "study_reference": STUDY_REF,
            "doi":             DOI,
            "confirmed_year":  CONFIRMED_YEAR,
            "confidence_level": conf,
            "notes": (
                f"ICS={ics:.4f}; informative_loci={n_info}; "
                f"exclusions={n_excl}; "
                f"accession_child={child_iid}; accession_parent={parent_iid}"
            ),
        })

    print(f"\n  Results:")
    print(f"    Edges scored   : {len(rows)}")
    print(f"    Skipped (no genotype / missing accession): {skipped_missing}")
    print(f"    Skipped (< {MIN_INFORMATIVE} informative loci): {skipped_low_n}")
    for conf, n in sorted(by_conf.items()):
        print(f"      {conf:12s}: {n}")

    if not rows:
        print("\n  No rows to write. Exiting.")
        sys.exit(0)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n✓ Written {len(df)} rows → {OUT_PATH}")

    # Summary: show unique children added
    unique_children = df["child_variety"].nunique()
    print(f"  Unique children with new evidence: {unique_children}")
    print(f"\n  Sample output (first 10 rows):")
    print(df[["child_variety","parent_variety","confidence_level","notes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
