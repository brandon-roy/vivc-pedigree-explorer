"""
build_margaryan_2021_ssr.py
============================
Extract SSR profiles from Margaryan et al. 2021 (Biology 10:1279,
DOI 10.3390/biology10121279) Table S4 and integrate into
node_molecular_profile.csv.

Source files:
  data/biology-10-01279-s001/
    Table S4. Microsatellite profiles for 24 nSSR loci...xlsx  (sheet "Table S 4.")
    Table S2. MCPD data...xlsx  (sheet "Table S 2-a")

Loci covered (9 GENRES081): VVS2, VVMD5, VVMD7, VVMD25, VVMD27, VVMD28,
                             VVMD32, VrZAG62, VrZAG79
No size-standard offsets needed — alleles are already VIVC-standardised
(verified by cross-check against Lacombe NMP data with 0 difference).

Output:
  data/ssr_margaryan_2021.csv         — 222 VIVC-standardised profiles
  data/node_molecular_profile.csv     — backfilled + new rows appended
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
DATA_DIR   = BASE_DIR / "data"
SUPP_DIR   = DATA_DIR / "biology-10-01279-s001"

S4_PATH    = SUPP_DIR / "Table S4. Microsatellite profiles for 24 nSSR loci of analyzed grapevine varieties and unknown genotypes.xlsx"
S2_PATH    = SUPP_DIR / "Table S2. MCPD data, assessment of truneness to type and bibliography of analyzed grapevine varieties.xlsx"

NMP_PATH   = DATA_DIR / "node_molecular_profile.csv"
OUT_PATH   = DATA_DIR / "ssr_margaryan_2021.csv"

SSR_SOURCE = "Margaryan 2021"

# Columns in Table S4 row 1 (0-indexed):
# 0  = VIVC prime name
# 1  = VVS2a, 2  = VVS2b
# 3  = VVMD5a, 4 = VVMD5b
# 5  = VVMD7a, 6 = VVMD7b
# 7  = VVMD25a, 8 = VVMD25b
# 9  = VVMD27a, 10 = VVMD27b
# 11 = VVMD28a, 12 = VVMD28b
# 13 = VVMD32a, 14 = VVMD32b
# 15 = VrZAG62a, 16 = VrZAG62b
# 17 = VrZAG79a, 18 = VrZAG79b

LOCI = [
    ("VVS2",    1,  2),
    ("VVMD5",   3,  4),
    ("VVMD7",   5,  6),
    ("VVMD25",  7,  8),
    ("VVMD27",  9,  10),
    ("VVMD28",  11, 12),
    ("VVMD32",  13, 14),
    ("VrZAG62", 15, 16),
    ("VrZAG79", 17, 18),
]

# NMP locus column names (slash-format strings: "133/149")
NMP_LOCUS_COLS = {
    "VVS2":    "ssr_VVS2",
    "VVMD5":   "ssr_VVMD5",
    "VVMD7":   "ssr_VVMD7",
    "VVMD25":  "ssr_VVMD25",
    "VVMD27":  "ssr_VVMD27",
    "VVMD28":  "ssr_VVMD28",
    "VVMD32":  "ssr_VVMD32",
    "VrZAG62": "ssr_VrZAG62",
    "VrZAG79": "ssr_VrZAG79",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_empty(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return str(val).strip().lower() in ("", "nan", "none")


def _allele_int(val) -> int | None:
    """Convert a cell value to an integer allele size, or None."""
    if _is_empty(val):
        return None
    try:
        f = float(val)
        if math.isnan(f):
            return None
        return int(round(f))
    except (ValueError, TypeError):
        return None


def _make_profile(a: int | None, b: int | None) -> str:
    """Return 'a/b' VIVC slash notation, or empty string."""
    if a is None and b is None:
        return ""
    if a is None:
        return str(b)
    if b is None:
        return str(a)
    lo, hi = (a, b) if a <= b else (b, a)
    return f"{lo}/{hi}"


# ── Load Table S2 → VIVC variety number lookup ────────────────────────────────

def load_vivc_numbers() -> dict[str, int]:
    """Return dict: VIVC prime name (upper) → VIVC variety number (int)."""
    df = pd.read_excel(S2_PATH, sheet_name="Table S 2-a", header=None)
    # Row 2 = headers, row 3+ = data
    # Col 0 = VIVC variety number, col 1 = Prime name
    data = df.iloc[3:, [0, 1]].copy()
    data.columns = ["vivc_no", "prime_name"]
    out: dict[str, int] = {}
    matched = 0
    for _, row in data.iterrows():
        name = str(row["prime_name"]).strip().upper()
        try:
            no = int(float(str(row["vivc_no"])))
            if name and no > 0:
                out[name] = no
                matched += 1
        except (ValueError, TypeError):
            pass
    print(f"  Table S2: {matched} VIVC number entries loaded")
    return out


# ── Load Table S4 → SSR profiles ──────────────────────────────────────────────

def load_ssr_profiles(vivc_no_map: dict[str, int]) -> pd.DataFrame:
    """Parse Table S4 and return a DataFrame of SSR profiles."""
    df = pd.read_excel(S4_PATH, sheet_name="Table S 4.", header=None)
    # Row 0 = title, row 1 = headers, rows 2+ = data
    data = df.iloc[2:].reset_index(drop=True)

    rows = []
    s2_matched = 0

    for _, row in data.iterrows():
        raw_name = str(row.iloc[0]).strip()
        if not raw_name or raw_name.lower() in ("nan", "none", ""):
            continue

        vivc_prime_name = raw_name.upper().strip()
        vivc_no = vivc_no_map.get(vivc_prime_name)
        if vivc_no is not None:
            s2_matched += 1

        rec: dict = {
            "vivc_prime_name": vivc_prime_name,
            "vivc_no":         vivc_no,
            "ssr_source":      SSR_SOURCE,
        }

        for locus, col_a, col_b in LOCI:
            a = _allele_int(row.iloc[col_a])
            b = _allele_int(row.iloc[col_b])
            rec[f"ssr_{locus}_a"] = a
            rec[f"ssr_{locus}_b"] = b

        rows.append(rec)

    print(f"  Table S4: {len(rows)} variety rows extracted")
    print(f"  Matched to Table S2 VIVC number: {s2_matched}/{len(rows)}")
    return pd.DataFrame(rows)


# ── Build output CSV (a/b columns) ────────────────────────────────────────────

def build_output_csv(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape to the requested output schema:
      vivc_prime_name, vivc_no, ssr_source,
      ssr_vvs2_a, ssr_vvs2_b, ssr_vvmd5_a, ssr_vvmd5_b, ...
    (lowercase column names)
    """
    out_cols = ["vivc_prime_name", "vivc_no", "ssr_source"]
    for locus, _, _ in LOCI:
        locus_lc = locus.lower()
        out_cols.append(f"ssr_{locus_lc}_a")
        out_cols.append(f"ssr_{locus_lc}_b")

    rows = []
    for _, prow in profiles.iterrows():
        rec: dict = {
            "vivc_prime_name": prow["vivc_prime_name"],
            "vivc_no":         prow["vivc_no"],
            "ssr_source":      prow["ssr_source"],
        }
        for locus, _, _ in LOCI:
            locus_lc = locus.lower()
            rec[f"ssr_{locus_lc}_a"] = prow.get(f"ssr_{locus}_a")
            rec[f"ssr_{locus_lc}_b"] = prow.get(f"ssr_{locus}_b")
        rows.append(rec)

    return pd.DataFrame(rows, columns=out_cols)


# ── Update node_molecular_profile.csv ─────────────────────────────────────────

def update_nmp(profiles: pd.DataFrame) -> tuple[int, int]:
    """
    Backfill empty locus slots in NMP for existing varieties;
    append new rows for varieties not yet in NMP.

    NMP stores slash-format strings: e.g. "133/149"
    profiles stores integer a/b columns: ssr_VVS2_a, ssr_VVS2_b, etc.

    Returns (n_cells_backfilled, n_new_rows).
    """
    nmp = pd.read_csv(NMP_PATH, low_memory=False)
    nmp["vivc_prime_name"] = nmp["vivc_prime_name"].fillna("").str.upper().str.strip()

    nmp_index: dict[str, int] = {
        name: idx for idx, name in enumerate(nmp["vivc_prime_name"])
    }

    n_filled = 0
    new_rows: list[dict] = []

    for _, prow in profiles.iterrows():
        prime = str(prow["vivc_prime_name"]).upper().strip()
        if not prime:
            continue

        if prime not in nmp_index:
            # New variety — build minimal NMP row
            new_rec: dict = {col: np.nan for col in nmp.columns}
            new_rec["vivc_prime_name"] = prime
            new_rec["ssr_source"] = SSR_SOURCE
            for locus, _, _ in LOCI:
                nmp_col = NMP_LOCUS_COLS[locus]
                a = prow.get(f"ssr_{locus}_a")
                b = prow.get(f"ssr_{locus}_b")
                profile_str = _make_profile(
                    None if _is_empty(a) else int(a),
                    None if _is_empty(b) else int(b),
                )
                new_rec[nmp_col] = profile_str if profile_str else np.nan
            new_rows.append(new_rec)
        else:
            idx = nmp_index[prime]
            for locus, _, _ in LOCI:
                nmp_col = NMP_LOCUS_COLS[locus]
                if nmp_col not in nmp.columns:
                    continue
                existing = str(nmp.at[idx, nmp_col]).strip()
                if existing not in ("", "nan", "None"):
                    continue  # already filled
                a = prow.get(f"ssr_{locus}_a")
                b = prow.get(f"ssr_{locus}_b")
                profile_str = _make_profile(
                    None if _is_empty(a) else int(a),
                    None if _is_empty(b) else int(b),
                )
                if profile_str:
                    nmp.at[idx, nmp_col] = profile_str
                    # Update ssr_source if blank
                    src = str(nmp.at[idx, "ssr_source"]).strip()
                    if src in ("", "nan", "None", "VIVC"):
                        nmp.at[idx, "ssr_source"] = SSR_SOURCE
                    n_filled += 1

    if new_rows:
        nmp = pd.concat([nmp, pd.DataFrame(new_rows)], ignore_index=True)

    nmp.to_csv(NMP_PATH, index=False)
    return n_filled, len(new_rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("build_margaryan_2021_ssr.py")
    print("=" * 60)

    print("\n[1] Loading Table S2 (VIVC variety numbers)...")
    vivc_no_map = load_vivc_numbers()

    print("\n[2] Loading Table S4 (SSR profiles)...")
    profiles = load_ssr_profiles(vivc_no_map)

    print("\n[3] Building output CSV...")
    out_df = build_output_csv(profiles)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"  Saved {len(out_df)} rows → {OUT_PATH.name}")

    print("\n[4] Updating node_molecular_profile.csv...")
    n_filled, n_new = update_nmp(profiles)
    print(f"  Cells backfilled in existing NMP rows: {n_filled}")
    print(f"  New variety rows appended to NMP:      {n_new}")

    print("\n[5] SSR coverage summary:")
    for locus, _, _ in LOCI:
        locus_lc = locus.lower()
        a_col = f"ssr_{locus_lc}_a"
        b_col = f"ssr_{locus_lc}_b"
        filled = out_df[[a_col, b_col]].apply(
            lambda row: not (_is_empty(row[a_col]) and _is_empty(row[b_col])),
            axis=1,
        ).sum()
        print(f"  {locus:<10}: {filled}/{len(out_df)} varieties")

    print("\nDone.")


if __name__ == "__main__":
    main()
