"""
Build pedigree support evidence from Myles et al. 2011 Vitis9KSNP array — v2.

Improvements over v1:
  - Uses GRIN_accessionID column from grin_vivc_alignment.csv (which already
    includes DVIT_ and PI_ prefixes) to bridge Myles local_id → vivc_prime_name.
  - Also adds GVIT_ prefix handling (some sample_info rows have GVIT_ local_ids).
  - Keeps fallback strategies: direct cultivar-name match and alias-map lookup.
  - Draws pedigree edges from BOTH vivc_supplementary.csv AND grin_pedigree_support.csv.

ICS constants (unchanged from v1):
    ICS_CONFIRMED = 0.97  (≥ 0.97  → confirmed if n ≥ 500)
    ICS_PROBABLE  = 0.93  (≥ 0.93  → probable)
    ICS_WEAK      = 0.85  (≥ 0.85  → probable, lower confidence)
    MIN_INFORMATIVE = 200 (minimum homozygous-parent loci required)

Output: data/myles_2011_marker_support.csv  (12-column schema)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
BASE_DIR   = SCRIPT_DIR.parent
SUPP_DIR   = Path("/Users/brandonroy/Desktop/Informatique/Projects/VIVC App"
                  "/Supplemental Literature Data")

PED_PATH  = SUPP_DIR / "20110117_pnas_grape_rawdat/vitis9ksnp_finalfilt.ped"
MAP_PATH  = SUPP_DIR / "20110117_pnas_grape_rawdat/vitis9ksnp_finalfilt.map"
INFO_PATH = SUPP_DIR / "20110117_pnas_grape_rawdat/sample_info.txt"

PASSPORT_PATH    = BASE_DIR / "vivc_passport_table.csv"
SLIM_PATH        = BASE_DIR / "data"  / "vivc_passport_slim.csv"
ALIAS_PATH       = BASE_DIR / "data" / "marker_name_aliases.csv"
GRIN_ALIGN_PATH  = BASE_DIR / "data" / "grin_vivc_alignment.csv"
VIVC_SUPP_PATH   = BASE_DIR / "data" / "vivc_supplementary.csv"
GRIN_PED_PATH    = BASE_DIR / "data" / "grin_pedigree_support.csv"
OUT_PATH         = BASE_DIR / "data" / "myles_2011_marker_support.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
N_SNPS          = 6114
DOI             = "10.1073/pnas.1009363108"
STUDY_REF       = "Myles et al. 2011 (PNAS)"
CONFIRMED_YEAR  = "2011"
MIN_INFORMATIVE = 200

ICS_CONFIRMED = 0.97
ICS_PROBABLE  = 0.93
ICS_WEAK      = 0.85

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(name: str) -> str:
    """Normalise a variety name: replace _ with space, strip, uppercase."""
    return re.sub(r"_+", " ", str(name or "")).strip().upper()


def _clean(s) -> str:
    s = str(s or "").strip().upper()
    return "" if s in {"", "NAN", "NONE", "NA", "N/A"} else s


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


def build_id_to_vivc() -> dict[str, str]:
    """
    Return {GRIN_accessionID: vivc_prime_name} from grin_vivc_alignment.csv.
    Covers DVIT_XXX, GVIT_XXX, PI_XXXXXX, etc.
    """
    if not GRIN_ALIGN_PATH.exists():
        print("  [WARN] grin_vivc_alignment.csv not found — skipping ID lookup")
        return {}
    ga = pd.read_csv(GRIN_ALIGN_PATH, dtype=str, low_memory=False)
    id_map: dict[str, str] = {}
    for _, row in ga.iterrows():
        acc_id  = str(row.get("GRIN_accessionID", "") or "").strip()
        vivc_nm = str(row.get("vivc_prime_name", "")  or "").strip().upper()
        if acc_id and vivc_nm and vivc_nm not in {"NAN", "NONE", ""}:
            id_map[acc_id] = vivc_nm
    print(f"  GRIN ID → VIVC map: {len(id_map)} entries")
    return id_map


def load_vivc_names() -> set[str]:
    """Load the set of all VIVC prime_names (uppercase)."""
    for path in (PASSPORT_PATH, SLIM_PATH):
        if path.exists():
            try:
                raw = pd.read_csv(path, dtype=str, low_memory=False,
                                  keep_default_na=False)
                break
            except Exception:
                continue
    else:
        print("  [WARN] Passport CSV not found — name set will be empty")
        return set()

    # Handle embedded header row
    if "prime_name" not in raw.columns and len(raw) >= 1:
        first_row = raw.iloc[0].fillna("").astype(str)
        if (first_row != "").sum() >= 4:
            raw.columns = [c.strip().lower().replace(" ", "_").replace("-", "_")
                           for c in first_row.tolist()]
            raw = raw.iloc[2:].reset_index(drop=True)

    raw.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
         .replace("(", "").replace(")", "").replace("/", "_")
        for c in raw.columns
    ]
    col_name = next((c for c in raw.columns if c == "prime_name"), None)
    if col_name is None:
        print("  [WARN] prime_name column not found in passport")
        return set()
    names = {_clean(v) for v in raw[col_name] if _clean(v)}
    print(f"  VIVC varieties: {len(names):,}")
    return names


def load_all_edges() -> list[tuple[str, str, str]]:
    """
    Load pedigree edges from multiple sources.  Priority / source order:
        1. vivc_passport_table.csv (or slim) — primary parent columns
        2. grin_pedigree_support.csv         — child / parent1 / parent2 columns
    Returns deduplicated list of (child, parent, role).
    """
    edges: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(child: str, parent: str, role: str) -> None:
        key = (child, parent)
        if key not in seen and child and parent:
            seen.add(key)
            edges.append((child, parent, role))

    # 1. VIVC passport
    for path in (PASSPORT_PATH, SLIM_PATH):
        if path.exists():
            try:
                raw = pd.read_csv(path, dtype=str, low_memory=False,
                                  keep_default_na=False)
                break
            except Exception:
                continue
    else:
        raw = pd.DataFrame()

    if not raw.empty:
        # Handle embedded header row
        if "prime_name" not in raw.columns and len(raw) >= 1:
            first_row = raw.iloc[0].fillna("").astype(str)
            if (first_row != "").sum() >= 4:
                raw.columns = [c.strip().lower().replace(" ", "_").replace("-", "_")
                               for c in first_row.tolist()]
                raw = raw.iloc[2:].reset_index(drop=True)
        raw.columns = [
            c.strip().lower().replace(" ", "_").replace("-", "_")
             .replace("(", "").replace(")", "").replace("/", "_")
            for c in raw.columns
        ]
        col_name = next((c for c in raw.columns if c == "prime_name"), None)
        col_p1   = next((c for c in raw.columns if "parent_1" in c), None)
        col_p2   = next((c for c in raw.columns if "parent_2" in c), None)
        if col_name and (col_p1 or col_p2):
            for _, row in raw.iterrows():
                child = _clean(row.get(col_name, ""))
                p1    = _clean(row.get(col_p1, "")) if col_p1 else ""
                p2    = _clean(row.get(col_p2, "")) if col_p2 else ""
                if child and p1:
                    _add(child, p1, "Parent 1")
                if child and p2:
                    _add(child, p2, "Parent 2")
    print(f"  After passport: {len(edges)} edges")

    # 2. grin_pedigree_support.csv  (columns: child, parent1, parent2)
    if GRIN_PED_PATH.exists():
        df = pd.read_csv(GRIN_PED_PATH, dtype=str, low_memory=False).fillna("")
        for _, row in df.iterrows():
            # Try both possible column naming conventions
            child = _clean(row.get("child",          row.get("child_variety",  "")))
            p1    = _clean(row.get("parent1",        row.get("parent_variety", "")))
            p2    = _clean(row.get("parent2",        ""))
            if child and p1:
                _add(child, p1, "Parent 1")
            if child and p2:
                _add(child, p2, "Parent 2")
        print(f"  After grin_pedigree_support: {len(edges)} total edges")

    return edges


def map_myles_samples(
    id_to_vivc: dict[str, str],
    vivc_names: set[str],
    alias_map: dict[str, str],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Build:
        illum_to_vivc  : {illum_id → vivc_prime_name}
        vivc_to_illums : {vivc_prime_name → [illum_ids]}

    Priority:
        1. local_id (DVIT_XXX / GVIT_XXX / PI_XXXXXX) → id_to_vivc
        2. normalised cultivar name → direct VIVC prime-name match
        3. normalised cultivar name → alias_map
    """
    info = pd.read_csv(INFO_PATH, sep=r"\s+", dtype=str, engine="python").fillna("")
    # Keep only V. vinifera (includes subsp. sativa etc.)
    info = info[info["species"].str.lower().str.contains("vinifera", na=False)].copy()
    print(f"  Myles V. vinifera samples: {len(info)}")

    illum_to_vivc: dict[str, str]        = {}
    vivc_to_illums: dict[str, list[str]] = {}
    n_id, n_direct, n_alias, n_fail      = 0, 0, 0, 0

    for _, row in info.iterrows():
        iid      = str(row.get("illum_id", "")).strip()
        local_id = str(row.get("local_id",  "")).strip()   # e.g. DVIT_677, GVIT_12
        cultivar = _norm(row.get("cultivar", ""))

        vivc_nm: str | None = None

        # Strategy 1 — GRIN accession ID lookup (DVIT_ / GVIT_ / PI_ / ...)
        if local_id and local_id in id_to_vivc:
            vivc_nm = id_to_vivc[local_id]
            n_id += 1

        # Strategy 2 — direct cultivar name match against VIVC prime-names
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

    print(f"  Myles name matching: ID={n_id}, direct={n_direct}, "
          f"alias={n_alias}, unmatched={n_fail}")
    print(f"  Unique VIVC varieties in Myles: {len(vivc_to_illums)}")
    return illum_to_vivc, vivc_to_illums


def load_ped_subset(needed_ids: set[str]) -> dict[str, list[tuple[str, str]]]:
    """
    Scan PED file; return {illum_id: [(a1,a2), ...]} for requested IDs.
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
            genos   = [(alleles[i * 2], alleles[i * 2 + 1]) for i in range(N_SNPS)]
            genotypes[iid] = genos
            if len(genotypes) == total:
                break
    print(f" loaded {len(genotypes)}/{total}")
    return genotypes


def pick_best_per_variety(
    vivc_to_illums: dict[str, list[str]],
    genotypes: dict[str, list[tuple[str, str]]],
) -> dict[str, str]:
    """Select the accession with fewest missing calls per VIVC variety."""
    best: dict[str, str] = {}
    for vivc_nm, illum_ids in vivc_to_illums.items():
        available = [iid for iid in illum_ids if iid in genotypes]
        if not available:
            continue
        best[vivc_nm] = min(
            available,
            key=lambda iid: sum(1 for a1, _ in genotypes[iid] if a1 == "0"),
        )
    return best


def compute_ics(
    parent_genos: list[tuple[str, str]],
    child_genos:  list[tuple[str, str]],
) -> tuple[float | None, int, int]:
    """
    IBD Consistency Score for a proposed parent–offspring pair.
    Returns (ics, n_informative, n_excluded) or (None, n_informative, 0).
    """
    n_info = n_excl = 0
    for (p1, p2), (c1, c2) in zip(parent_genos, child_genos):
        if p1 == "0" or c1 == "0":
            continue
        if p1 != p2:
            continue          # parent het → always compatible
        n_info += 1
        if p1 not in (c1, c2):
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
        return "probable"
    return "disputed"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== build_myles_pedigree_support_v2.py ===\n")

    # 1. Helper maps
    print("[1/6] Loading alias map and GRIN-ID → VIVC map …")
    alias_map  = load_alias_map()
    id_to_vivc = build_id_to_vivc()

    # 2. VIVC variety name set
    print("\n[2/6] Loading VIVC variety names …")
    vivc_names = load_vivc_names()

    # 3. Myles sample → VIVC mapping
    print("\n[3/6] Matching Myles samples → VIVC prime names …")
    illum_to_vivc, vivc_to_illums = map_myles_samples(
        id_to_vivc, vivc_names, alias_map
    )

    # 4. Pedigree edges covered by Myles
    print("\n[4/6] Loading pedigree edges …")
    all_edges = load_all_edges()
    print(f"  Total pedigree edges: {len(all_edges)}")

    covered = [
        (child, parent, role)
        for child, parent, role in all_edges
        if child in vivc_to_illums and parent in vivc_to_illums
    ]
    print(f"  Edges with both members in Myles: {len(covered)}")
    if not covered:
        print("  No covered edges — check name matching. Exiting.")
        sys.exit(0)

    # 5. Load genotypes
    print("\n[5/6] Loading PED genotypes …")
    needed_ids: set[str] = set()
    for child, parent, _ in covered:
        needed_ids.update(vivc_to_illums[child])
        needed_ids.update(vivc_to_illums[parent])
    genotypes = load_ped_subset(needed_ids)
    best_acc  = pick_best_per_variety(vivc_to_illums, genotypes)

    # 6. Compute ICS
    print("\n[6/6] Computing IBD Consistency Scores …")
    rows: list[dict]   = []
    skipped_missing    = 0
    skipped_low_n      = 0
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
            "child_variety":    child,
            "parent_variety":   parent,
            "parent_role":      role,
            "evidence_type":    "SNP array",
            "marker_type":      "Vitis9KSNP",
            "n_markers":        N_SNPS,
            "lod_score":        np.nan,
            "study_reference":  STUDY_REF,
            "doi":              DOI,
            "confirmed_year":   CONFIRMED_YEAR,
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
    print(f"\nWritten {len(df)} rows → {OUT_PATH}")

    unique_children = df["child_variety"].nunique()
    print(f"  Unique children with evidence: {unique_children}")
    print(f"\n  Sample output (first 10 rows):")
    print(df[["child_variety", "parent_variety", "confidence_level", "notes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
