"""
Build node_molecular_profile.csv — one row per unique vivc_prime_name.

Columns:
    vivc_prime_name,
    vv_ancestry_pct, na1_ancestry_pct, na2_ancestry_pct, mus_ancestry_pct, ea_ancestry_pct,
    ancestry_n_samples, ancestry_source,
    ssr_VVS2, ssr_VVMD5, ssr_VVMD7, ssr_VVMD25, ssr_VVMD27, ssr_VVMD28,
    ssr_VVMD32, ssr_VrZAG62, ssr_VrZAG79, ssr_source,
    has_snp_myles, has_snp_constantini,
    data_sources

Run with:  python scripts/build_node_molecular_profile.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR       = Path(__file__).parent
BASE_DIR         = SCRIPT_DIR.parent

GRIN_ALIGN_PATH  = BASE_DIR / "data" / "grin_vivc_alignment.csv"
SSR_PATH         = BASE_DIR / "data" / "ssr_vivc.csv"
MYLES_PATH       = BASE_DIR / "data" / "myles_2011_marker_support.csv"
CONSTANTINI_PATH = BASE_DIR / "data" / "constantini_2026_marker_support.csv"
OUT_PATH         = BASE_DIR / "data" / "node_molecular_profile.csv"

ANCESTRY_SOURCE = "Myles et al. 2011 / GRIN (DVIT admixture)"
SSR_SOURCE      = "VIVC"

# The 9 standard SSR loci for GENRES081
SSR_LOCI = ["VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28",
            "VVMD32", "VrZAG62", "VrZAG79"]


# ── Ancestry section ───────────────────────────────────────────────────────────

def build_ancestry(grin_align: pd.DataFrame) -> pd.DataFrame:
    """
    From grin_vivc_alignment.csv, build one ancestry row per vivc_prime_name.

    For varieties with multiple accessions, select the QC-passing accession with
    the lowest missingness_rate. If no QC-passing accession exists, use the one
    with the lowest missingness_rate regardless.

    Returns DataFrame indexed by vivc_prime_name (uppercase).
    """
    df = grin_align.copy()

    # Normalise vivc_prime_name
    df["vivc_prime_name"] = df["vivc_prime_name"].str.strip().str.upper()
    df = df[df["vivc_prime_name"].notna() & (df["vivc_prime_name"] != "")].copy()

    # Cast numeric columns
    df["passQC"]          = df["passQC"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    df["missingness_rate"] = pd.to_numeric(df["missingness_rate"], errors="coerce")

    anc_cols = ["Vv_ancestry_pct", "NA1_ancestry_pct", "NA2_ancestry_pct",
                "Mus_ancestry_pct", "EA_ancestry_pct"]
    for col in anc_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort: QC-passing first, then by missingness ascending
    df["_passQC_int"] = (~df["passQC"]).astype(int)   # 0 = passes QC
    df = df.sort_values(["vivc_prime_name", "_passQC_int", "missingness_rate"])

    # Count samples per variety before dedup
    counts = df.groupby("vivc_prime_name").size().rename("ancestry_n_samples")

    # Take the first (best) row per variety
    best = df.groupby("vivc_prime_name", as_index=False).first()
    best = best.merge(counts, on="vivc_prime_name", how="left")

    # Build output
    out = pd.DataFrame()
    out["vivc_prime_name"]  = best["vivc_prime_name"]
    out["vv_ancestry_pct"]  = best["Vv_ancestry_pct"].round(1)
    out["na1_ancestry_pct"] = best["NA1_ancestry_pct"].round(1)
    out["na2_ancestry_pct"] = best["NA2_ancestry_pct"].round(1)
    out["mus_ancestry_pct"] = best["Mus_ancestry_pct"].round(1)
    out["ea_ancestry_pct"]  = best["EA_ancestry_pct"].round(1)
    out["ancestry_n_samples"] = best["ancestry_n_samples"].fillna(0).astype(int)
    out["ancestry_source"]    = ANCESTRY_SOURCE

    return out.set_index("vivc_prime_name")


# ── SSR section ────────────────────────────────────────────────────────────────

def build_ssr(ssr_path: Path) -> pd.DataFrame:
    """
    Parse ssr_vivc.csv.  The first data row is a header artefact ('Alleles:',…)
    and must be dropped.  For each variety, take the pre-combined 'A1/A2' column
    for each locus.

    Returns DataFrame indexed by variety name (uppercase).
    """
    if not ssr_path.exists():
        print(f"  [WARN] {ssr_path.name} not found — SSR data will be empty")
        return pd.DataFrame()

    df = pd.read_csv(ssr_path, dtype=str).fillna("")

    # Drop the 'Alleles:' artefact row (first row where variety == 'Alleles:')
    df = df[df["variety"].str.strip().str.upper() != "ALLELES:"].copy()

    df["variety"] = df["variety"].str.strip().str.upper()
    df = df[df["variety"] != ""].copy()

    out_rows = []
    for _, row in df.iterrows():
        rec = {"vivc_prime_name": row["variety"], "ssr_source": SSR_SOURCE}
        for locus in SSR_LOCI:
            # The pre-combined column has the same name as the locus (e.g. 'VVS2')
            val = str(row.get(locus, "")).strip()
            rec[f"ssr_{locus}"] = val if val not in ("", "nan", "NA", "0/0") else None
        out_rows.append(rec)

    out = pd.DataFrame(out_rows).set_index("vivc_prime_name")
    return out


# ── SNP flags ─────────────────────────────────────────────────────────────────

def build_snp_flags(
    myles_path: Path,
    constantini_path: Path,
) -> pd.DataFrame:
    """
    Return DataFrame with columns [vivc_prime_name, has_snp_myles, has_snp_constantini].
    """
    myles_varieties: set[str] = set()
    if myles_path.exists():
        m = pd.read_csv(myles_path, dtype=str).fillna("")
        for col in ("child_variety", "parent_variety"):
            if col in m.columns:
                myles_varieties.update(
                    v.strip().upper() for v in m[col] if v.strip()
                )

    const_varieties: set[str] = set()
    if constantini_path.exists():
        c = pd.read_csv(constantini_path, dtype=str).fillna("")
        for col in ("child_variety", "parent_variety"):
            if col in c.columns:
                const_varieties.update(
                    v.strip().upper() for v in c[col] if v.strip()
                )

    all_names = myles_varieties | const_varieties
    rows = []
    for name in all_names:
        rows.append({
            "vivc_prime_name":    name,
            "has_snp_myles":       name in myles_varieties,
            "has_snp_constantini": name in const_varieties,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index("vivc_prime_name")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== build_node_molecular_profile.py ===\n")

    # Load source data
    print("[1/4] Loading grin_vivc_alignment …")
    if not GRIN_ALIGN_PATH.exists():
        print(f"  [ERROR] {GRIN_ALIGN_PATH} not found. Aborting.")
        return
    grin_align = pd.read_csv(GRIN_ALIGN_PATH, dtype=str, low_memory=False)
    print(f"  {len(grin_align)} rows")

    print("\n[2/4] Building ancestry table …")
    ancestry_df = build_ancestry(grin_align)
    print(f"  {len(ancestry_df)} unique VIVC varieties with ancestry data")

    print("\n[3/4] Building SSR table …")
    ssr_df = build_ssr(SSR_PATH)
    print(f"  {len(ssr_df)} varieties with SSR data")

    print("\n[4/4] Building SNP flags …")
    snp_df = build_snp_flags(MYLES_PATH, CONSTANTINI_PATH)
    print(f"  Myles:       {snp_df['has_snp_myles'].sum() if not snp_df.empty else 0} varieties")
    print(f"  Constantini: {snp_df['has_snp_constantini'].sum() if not snp_df.empty else 0} varieties")

    # ── Merge all sections ────────────────────────────────────────────────────
    print("\nMerging …")

    # Union of all variety names
    all_names: set[str] = set(ancestry_df.index) | set(ssr_df.index) | set(snp_df.index)
    base = pd.DataFrame({"vivc_prime_name": sorted(all_names)})
    base = base.set_index("vivc_prime_name")

    # Ancestry
    anc_cols = ["vv_ancestry_pct", "na1_ancestry_pct", "na2_ancestry_pct",
                "mus_ancestry_pct", "ea_ancestry_pct", "ancestry_n_samples",
                "ancestry_source"]
    base = base.join(ancestry_df[anc_cols], how="left")

    # SSR
    ssr_cols = [f"ssr_{l}" for l in SSR_LOCI] + ["ssr_source"]
    if not ssr_df.empty:
        base = base.join(ssr_df[ssr_cols], how="left")
    else:
        for col in ssr_cols:
            base[col] = None

    # SNP flags
    if not snp_df.empty:
        base = base.join(snp_df[["has_snp_myles", "has_snp_constantini"]], how="left")
        base["has_snp_myles"]       = base["has_snp_myles"].infer_objects(copy=False).fillna(False).astype(bool)
        base["has_snp_constantini"] = base["has_snp_constantini"].infer_objects(copy=False).fillna(False).astype(bool)
    else:
        base["has_snp_myles"]       = False
        base["has_snp_constantini"] = False

    # data_sources
    def _sources(row: pd.Series) -> str:
        parts = []
        if pd.notna(row.get("vv_ancestry_pct")):
            parts.append("ancestry")
        if pd.notna(row.get("ssr_VVS2")):
            parts.append("SSR")
        if row.get("has_snp_myles"):
            parts.append("SNP_Myles")
        if row.get("has_snp_constantini"):
            parts.append("SNP_Constantini")
        return ",".join(parts) if parts else ""

    base["data_sources"] = base.apply(_sources, axis=1)

    base = base.reset_index()

    # Ensure canonical column order
    out_cols = [
        "vivc_prime_name",
        "vv_ancestry_pct", "na1_ancestry_pct", "na2_ancestry_pct",
        "mus_ancestry_pct", "ea_ancestry_pct",
        "ancestry_n_samples", "ancestry_source",
        "ssr_VVS2", "ssr_VVMD5", "ssr_VVMD7", "ssr_VVMD25", "ssr_VVMD27",
        "ssr_VVMD28", "ssr_VVMD32", "ssr_VrZAG62", "ssr_VrZAG79", "ssr_source",
        "has_snp_myles", "has_snp_constantini",
        "data_sources",
    ]
    for col in out_cols:
        if col not in base.columns:
            base[col] = None
    base = base[out_cols]

    base.to_csv(OUT_PATH, index=False)
    print(f"\nWritten {len(base)} rows x {len(base.columns)} cols → {OUT_PATH}")
    print(f"  Varieties with ancestry:  {base['vv_ancestry_pct'].notna().sum()}")
    print(f"  Varieties with SSR:       {base['ssr_VVS2'].notna().sum()}")
    print(f"  Varieties with SNP Myles: {base['has_snp_myles'].sum()}")
    print(f"\nSample (first 5 rows):")
    print(base[["vivc_prime_name", "vv_ancestry_pct", "ssr_VVS2",
                "has_snp_myles", "data_sources"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
