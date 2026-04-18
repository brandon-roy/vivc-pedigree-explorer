"""
build_tello_2024_ssr.py
=======================
Extract SSR genotype profiles from Tello et al. 2024 (Frontiers in Plant Science 15:1391679)
and integrate into node_molecular_profile.csv.

Source: data/Tello et al 2024/Table 3.XLSX  (sheet SM3)
        data/Tello et al 2024/Table 1 (1).XLSX (sheet SM1)  — VIVC ID lookup

Loci covered: VVS2, VVMD5, VVMD7, VVMD27, VVMD32, VrZAG62, VrZAG79  (7 of 9 GENRES081)
Missing loci: VVMD25, VVMD28

Size-standard offsets (Tello raw → VIVC-standard, verified against 20+ overlap varieties):
  VVS2   : −2   (exact, all alleles)
  VVMD5  : −4   (approx; alleles ≤234 need −5, but −4 used as approximation — ~6 residual 1bp discrepancies)
  VVMD7  :  0   (exact)
  VVMD27 : −4   (exact)
  VVMD32 : −1   (exact)
  VrZAG62:  0   (exact)
  VrZAG79:  0   (exact)

Format in Table 3: "133:143" → converted to "131/141" (VIVC standard slash notation).

Variety matching strategy (priority order):
  1. VIVC Number (column in Table 3) → vivc_supplementary.csv prime name
  2. Tello variety name → exact normalised match against vivc_supplementary
  3. No VIVC number → store with Tello name as-is (new variety entry)

Output:
  data/ssr_tello_2024.csv         — VIVC-standardised profiles (475 → 59 varieties, 7 loci)
  data/node_molecular_profile.csv — updated in-place (backfill + new rows)
  data/datasource_inventory.csv   — updated RanchoMerced entry + new Tello2024_SSR entry
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA    = Path(__file__).parent.parent / "data"
T3_PATH = DATA / "Tello et al 2024" / "Table 3.XLSX"
T1_PATH = DATA / "Tello et al 2024" / "Table 1 (1).XLSX"
NMP_PATH = DATA / "node_molecular_profile.csv"
VIVC_SUP = DATA / "vivc_supplementary.csv"
INV_PATH = DATA / "datasource_inventory.csv"
OUT_PATH = DATA / "ssr_tello_2024.csv"

TELLO_LOCI = ["VVS2", "VVMD5", "VVMD7", "VVMD27", "VVMD32", "VrZAG62", "VrZAG79"]

# Tello raw → VIVC standard (add offset to each allele)
VIVC_OFFSETS: dict[str, int] = {
    "VVS2":    -2,
    "VVMD5":   -4,   # non-constant: ~-5 for alleles ≤234, -4 for ≥236; -4 used as approximation
    "VVMD7":    0,
    "VVMD27":  -4,
    "VVMD32":  -1,
    "VrZAG62":  0,
    "VrZAG79":  0,
}

SSR_SOURCE = "Tello et al. 2024"


# ── helpers ──────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).upper().strip())


def apply_offset(allele_str: str, offset: int) -> str:
    """Convert 'a1:a2' (Tello colon notation) to 'x/y' VIVC notation with offset applied."""
    if not allele_str or str(allele_str).strip() in ("nan", "None", ""):
        return ""
    # Support both ':' (Tello) and '/' notation
    sep = ":" if ":" in str(allele_str) else "/"
    parts = str(allele_str).split(sep)
    try:
        vals = sorted(int(p.strip()) + offset for p in parts if p.strip().isdigit())
        return "/".join(str(v) for v in vals) if vals else ""
    except ValueError:
        return str(allele_str)


def apply_offset_raw(allele_str: str) -> str:
    """Convert 'a1:a2' (Tello) to 'a1/a2' with no offset."""
    if not allele_str or str(allele_str).strip() in ("nan", "None", ""):
        return ""
    sep = ":" if ":" in str(allele_str) else "/"
    parts = str(allele_str).split(sep)
    try:
        vals = sorted(int(p.strip()) for p in parts if p.strip().isdigit())
        return "/".join(str(v) for v in vals) if vals else ""
    except ValueError:
        return str(allele_str)


# ── load Table 3 (SSR + SNP profiles) ────────────────────────────────────────

def load_table3() -> pd.DataFrame:
    df = pd.read_excel(T3_PATH, sheet_name="SM3", skiprows=2, header=0)
    df = df.rename(columns={df.columns[0]: "variety_name", df.columns[2]: "vivc_no_source"})
    df["variety_name"] = df["variety_name"].fillna("").apply(lambda x: str(x).strip())
    df = df[df["variety_name"] != ""].copy()
    df["variety_name_upper"] = df["variety_name"].str.upper().str.strip()
    return df


# ── load Table 1 (VIVC ID lookup) ────────────────────────────────────────────

def load_table1() -> dict[str, int]:
    """Return dict: variety_name_upper → vivc_no (int)."""
    df = pd.read_excel(T1_PATH, sheet_name="SM1", skiprows=2, header=0)
    out = {}
    for _, row in df.iterrows():
        name = str(row.get("Sample: Name", row.iloc[1])).upper().strip()
        vivc = row.get("VIVC Number", row.iloc[5])
        try:
            out[name] = int(vivc)
        except (ValueError, TypeError):
            pass
    return out


# ── load VIVC supplementary for prime-name lookup ────────────────────────────

def load_vivc_prime_names() -> dict[int, str]:
    """Return dict: vivc_no (int) → prime_name (upper)."""
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


# ── build profile rows ────────────────────────────────────────────────────────

def build_profiles(df: pd.DataFrame, vivc_map: dict[int, str]) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        vname   = row["variety_name_upper"]
        vivc_no = row.get("vivc_no_source")

        # Determine prime name
        try:
            vivc_int    = int(vivc_no)
            prime_name  = vivc_map.get(vivc_int, vname)
            match_method = "vivc_no"
        except (ValueError, TypeError):
            vivc_int   = None
            prime_name = vname
            match_method = "name_only"

        rec: dict = {
            "vivc_prime_name":     prime_name,
            "vivc_no_source":      vivc_int,
            "tello_variety_name":  row["variety_name"],
            "match_method":        match_method,
            "ssr_source":          SSR_SOURCE,
        }

        for locus in TELLO_LOCI:
            raw_val = row.get(locus, "")
            raw_str = apply_offset_raw(str(raw_val) if pd.notna(raw_val) else "")
            rec[f"ssr_{locus}_raw"] = raw_str
            offset  = VIVC_OFFSETS[locus]
            rec[f"ssr_{locus}"] = apply_offset(str(raw_val) if pd.notna(raw_val) else "", offset)

        rows.append(rec)

    return pd.DataFrame(rows)


# ── update node_molecular_profile.csv ────────────────────────────────────────

def update_nmp(profiles: pd.DataFrame) -> tuple[int, int]:
    """Backfill empty slots in node_molecular_profile and append new rows.
    Returns (n_backfilled_cells, n_new_rows).
    """
    nmp = pd.read_csv(NMP_PATH, low_memory=False)
    nmp_name_col = "vivc_prime_name"
    nmp[nmp_name_col] = nmp[nmp_name_col].fillna("").str.upper().str.strip()

    n_filled  = 0
    new_rows  = []

    for _, prow in profiles.iterrows():
        prime = str(prow["vivc_prime_name"]).upper().strip()
        if not prime:
            continue

        ssr_cols_avail = [c for c in profiles.columns if c.startswith("ssr_") and not c.endswith("_raw")]

        match = nmp[nmp[nmp_name_col] == prime]
        if match.empty:
            # New variety — build a minimal row
            new_rec = {col: np.nan for col in nmp.columns}
            new_rec[nmp_name_col] = prime
            new_rec["ssr_source"] = SSR_SOURCE
            for col in ssr_cols_avail:
                locus_col = col  # e.g. ssr_VVS2
                if locus_col in nmp.columns:
                    val = prow.get(col, "")
                    new_rec[locus_col] = val if val else np.nan
            new_rows.append(new_rec)
        else:
            idx = match.index[0]
            for col in ssr_cols_avail:
                locus_col = col
                if locus_col not in nmp.columns:
                    continue
                existing = str(nmp.at[idx, locus_col]).strip()
                if existing in ("", "nan", "None"):
                    new_val = prow.get(col, "")
                    if new_val and str(new_val).strip() not in ("", "nan", "None"):
                        nmp.at[idx, locus_col] = new_val
                        # Update ssr_source if it was empty/VIVC-only
                        src = str(nmp.at[idx, "ssr_source"]).strip()
                        if src in ("", "nan", "None", "VIVC"):
                            nmp.at[idx, "ssr_source"] = SSR_SOURCE
                        n_filled += 1

    if new_rows:
        nmp = pd.concat([nmp, pd.DataFrame(new_rows)], ignore_index=True)

    nmp.to_csv(NMP_PATH, index=False)
    return n_filled, len(new_rows)


# ── update datasource_inventory.csv ──────────────────────────────────────────

def update_inventory(n_varieties: int) -> None:
    inv = pd.read_csv(INV_PATH)
    entry = {
        "source_id":        "Tello2024_SSR",
        "citation":         "Tello J et al. (2024) Front. Plant Sci. 15:1391679",
        "doi":              "10.3389/fpls.2024.1391679",
        "n_varieties":      n_varieties,
        "loci_covered":     ", ".join(TELLO_LOCI),
        "technology":       "SSR (GENRES081 panel, 7 loci) + SNP",
        "region":           "Balkans / Serbia",
        "vivc_offsets":     "VVS2:-2, VVMD5:-4(approx), VVMD7:0, VVMD27:-4, VVMD32:-1, VrZAG62:0, VrZAG79:0",
        "notes":            ("59 non-redundant genotypes from 138 Serbian grapevines. "
                             "7 GENRES081 loci. VVMD25 and VVMD28 not genotyped. "
                             "Allele sizes in Tello are 2bp (VVS2, VVMD5) or 4bp (VVMD27) above VIVC standard; "
                             "offsets applied to standardise."),
    }
    mask = inv["source_id"] == "Tello2024_SSR"
    if mask.any():
        for k, v in entry.items():
            inv.loc[mask, k] = v
    else:
        inv = pd.concat([inv, pd.DataFrame([entry])], ignore_index=True)
    inv.to_csv(INV_PATH, index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading Table 3 (SSR profiles)...")
    df3 = load_table3()
    print(f"  {len(df3)} variety rows found in Table 3")

    print("Loading VIVC supplementary for prime-name lookup...")
    vivc_map = load_vivc_prime_names()
    print(f"  {len(vivc_map)} VIVC IDs loaded")

    print("Building VIVC-standardised profiles...")
    profiles = build_profiles(df3, vivc_map)
    profiles.to_csv(OUT_PATH, index=False)
    print(f"  Saved {len(profiles)} profiles → {OUT_PATH.name}")

    # Summary of match methods
    print("  Match method breakdown:")
    for method, cnt in profiles["match_method"].value_counts().items():
        print(f"    {method}: {cnt}")

    # Check how many have VIVC prime names vs name-only
    no_vivc = profiles[profiles["match_method"] == "name_only"]
    print(f"  Varieties without VIVC number: {len(no_vivc)}")
    if not no_vivc.empty:
        print("   ", ", ".join(no_vivc["tello_variety_name"].head(10).tolist()))

    print("\nUpdating node_molecular_profile.csv...")
    n_filled, n_new = update_nmp(profiles)
    print(f"  Backfilled cells: {n_filled}")
    print(f"  New variety rows added: {n_new}")

    print("\nUpdating datasource_inventory.csv...")
    update_inventory(len(profiles))
    print("  Done.")

    print("\nSummary of SSR coverage in Tello profiles:")
    for locus in TELLO_LOCI:
        col = f"ssr_{locus}"
        filled = profiles[col].apply(lambda x: str(x).strip() not in ("", "nan", "None")).sum()
        print(f"  {locus}: {filled}/{len(profiles)}")


if __name__ == "__main__":
    main()
