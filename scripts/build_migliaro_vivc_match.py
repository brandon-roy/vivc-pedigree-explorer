"""
build_migliaro_vivc_match.py
Match Migliaro 2019 rootstock SSR profiles to VIVC prime names and append
SSR data to node_molecular_profile.csv.

Rules:
 1. Direct match on prime_name (uppercased, stripped)
 2. Synonym lookup
 3. Rootstock name flip: '<number> <breeder>' → '<BREEDER> <number>'
    - Normalise Millardet: '101.14' → '101- 14'  /  '106.8' → '106- 8' etc.
    - Normalise multi-word periods/hyphens as needed
 4. Partial token search as fallback (only if exactly one VIVC match)
"""
import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MIG_PATH     = BASE_DIR / "data" / "ssr_migliaro_2019.csv"
SUPP_PATH    = BASE_DIR / "data" / "vivc_supplementary.csv"
PROFILE_PATH = BASE_DIR / "data" / "node_molecular_profile.csv"

# SSR loci in Migliaro 2019 (17 loci, stored as ssr_<LOCUS>_a1/a2 in CSV)
LOCI_17 = [
    "VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28",
    "VrZAG62", "VrZAG64", "VrZAG79", "VrZAG83",
    "VChr3a", "VChr4a", "VChr7b", "VChr10b", "VChr19a", "VMC6E1", "VMCNG4b9",
]
# GENRES081 SSR loci we carry in the main node profile
GENRES081 = ["VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28",
             "VVMD32", "VrZAG62", "VrZAG79"]
# Overlap between Migliaro panel and GENRES081 (VVMD32 missing in Migliaro)
PANEL_OVERLAP = [l for l in GENRES081 if l in LOCI_17]


# ── Name normalisation helpers ──────────────────────────────────────────────────

def _norm_sp(s: str) -> str:
    """Collapse internal whitespace."""
    return re.sub(r"\s+", " ", s.strip())


def vivc_millardet_number(num: str) -> str:
    """
    Convert Migliaro-style number to VIVC-style Millardet number.
    '101.14' → '101- 14'
    '106.8'  → '106-  8'
    '41'     → ' 41'   (bare integer)
    '125-1'  → '125- 1'
    """
    # Period separator → hyphen, e.g. 101.14 → 101-14
    num = num.replace(".", "-")
    # If already has hyphen, ensure VIVC-style spacing: '101-14' → '101- 14'
    if "-" in num:
        parts = num.split("-", 1)
        left, right = parts[0], parts[1]
        # VIVC pads single-digit right side differently
        return f"{left}- {right}"
    return num


def rootstock_vivc_name(variety_name: str, vivc_names_upper: set) -> str | None:
    """
    Try to derive the VIVC prime name from a Migliaro rootstock name.
    Returns the matching VIVC name (as stored), or None.
    """
    name_up = variety_name.upper().strip()

    # Rootstock pattern: starts with number/code, followed by breeder/word
    m = re.match(r"^([\d./\-]+[A-Z]?)\s+(.+)$", name_up)
    if m:
        num_raw, breeder = m.group(1), m.group(2).strip()

        # Normalise Millardet suffix
        breeder = re.sub(r"\s+ET\s+DE\s+GRASSET", " ET GRASSET", breeder)

        # Millardet number normalisation
        if "MILLARDET" in breeder or "GRASSET" in breeder:
            vivc_num = vivc_millardet_number(num_raw)
        else:
            # For Paulsen / Richter / Couderc / Ruggeri etc.,
            # replace period with nothing or keep as-is
            vivc_num = num_raw.replace(".", "")

        # Try permutations of BREEDER + space variants + NUMBER
        for breeder_name in [breeder, breeder.split()[0]]:
            for sep in ["  ", " ", "   "]:
                candidate = _norm_sp(f"{breeder_name}{sep}{vivc_num}")
                for vc in vivc_names_upper:
                    if _norm_sp(vc) == candidate:
                        return vc
            # Also try contains (unique match only)
            matches = [vc for vc in vivc_names_upper
                       if num_raw.replace(".", "") in re.sub(r"\s+", "", vc)
                       and breeder_name.split()[0] in vc]
            if len(matches) == 1:
                return matches[0]

    return None


def match_to_vivc(
    variety_name: str,
    vivc_upper_to_raw: dict,   # upper → original case prime_name
    syn_to_prime: dict,        # synonym upper → prime_name upper
) -> tuple[str | None, str]:
    """Returns (matched_vivc_prime_name_original_case, match_method)."""
    up = variety_name.upper().strip()

    # 1. Direct
    if up in vivc_upper_to_raw:
        return vivc_upper_to_raw[up], "direct"

    # 2. Synonym
    if up in syn_to_prime:
        prime_up = syn_to_prime[up]
        return vivc_upper_to_raw.get(prime_up, prime_up), "synonym"

    # 3. Rootstock flip
    candidate = rootstock_vivc_name(variety_name, set(vivc_upper_to_raw.keys()))
    if candidate:
        return vivc_upper_to_raw.get(candidate, candidate), "rootstock_flip"

    return None, "no_match"


def main():
    mig    = pd.read_csv(MIG_PATH)
    supp   = pd.read_csv(SUPP_PATH, dtype=str, low_memory=False)
    profile = pd.read_csv(PROFILE_PATH)

    # Build VIVC lookup structures
    vivc_upper_to_raw: dict[str, str] = {}
    for _, row in supp.iterrows():
        pname = str(row["prime_name"]).strip()
        vivc_upper_to_raw[pname.upper()] = pname

    syn_to_prime: dict[str, str] = {}
    for _, row in supp.iterrows():
        pname_up = str(row["prime_name"]).strip().upper()
        syns = str(row.get("synonyms", "") or "")
        for syn in syns.split("|"):
            s = syn.strip().upper()
            if s:
                syn_to_prime[s] = pname_up

    # Match all Migliaro entries
    match_results = []
    for _, row in mig.iterrows():
        vname = str(row["variety_name"]).strip()
        prime, method = match_to_vivc(vname, vivc_upper_to_raw, syn_to_prime)
        match_results.append({"migliaro_id": row["migliaro_id"],
                              "variety_name": vname,
                              "vivc_prime_name": prime,
                              "match_method": method})

    match_df = pd.DataFrame(match_results)
    matched   = match_df[match_df["vivc_prime_name"].notna()]
    unmatched = match_df[match_df["vivc_prime_name"].isna()]

    print(f"Migliaro entries: {len(mig)}")
    print(f"  Matched: {len(matched)} ({len(matched)/len(mig)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched)}")
    print("\nMethod breakdown:")
    print(match_df["match_method"].value_counts().to_string())
    print("\nSample matched:")
    print(matched[["variety_name", "vivc_prime_name", "match_method"]].head(20).to_string())
    print("\nUnmatched sample:")
    print(unmatched["variety_name"].tolist()[:20])

    # ── Build SSR columns for matched entries ────────────────────────────────────
    # For node_molecular_profile.csv we store the GENRES081 overlap loci.
    # ssr_<LOCUS> column format: "a1/a2"  (same convention as existing profile)
    profile_up = profile["vivc_prime_name"].str.strip().str.upper()
    new_rows = []

    for _, mrow in matched.iterrows():
        prime_name_orig = mrow["vivc_prime_name"]
        prime_up = str(prime_name_orig).strip().upper()

        # Skip if already in profile
        already = (profile_up == prime_up).any()
        if already:
            # Could update SSR; for now just note it
            continue

        # Get SSR values for this Migliaro entry
        mig_row = mig[mig["migliaro_id"] == mrow["migliaro_id"]].iloc[0]
        ssr_vals = {}
        for locus in PANEL_OVERLAP:
            a1 = str(mig_row[f"ssr_{locus}_a1"]).strip()
            a2 = str(mig_row[f"ssr_{locus}_a2"]).strip()
            if a1 != "nan" and a2 != "nan" and a1 != "0" and a2 != "0":
                ssr_vals[f"ssr_{locus}"] = f"{a1}/{a2}"

        if not ssr_vals:
            continue

        new_row = {"vivc_prime_name": prime_up}
        new_row.update(ssr_vals)
        new_row["ssr_source"] = "Migliaro et al. 2019"
        new_rows.append(new_row)

    print(f"\nNew profile rows to add: {len(new_rows)}")
    if new_rows:
        print(pd.DataFrame(new_rows).head(5).to_string())

    # Persist an updated match table for reference
    match_out = BASE_DIR / "data" / "migliaro_vivc_match.csv"
    match_df.to_csv(match_out, index=False)
    print(f"\nMatch table saved → {match_out}")

    # Append new SSR rows to node_molecular_profile.csv
    if new_rows:
        additions = pd.DataFrame(new_rows)
        # Ensure all profile columns exist
        for col in profile.columns:
            if col not in additions.columns:
                additions[col] = ""
        additions = additions.reindex(columns=profile.columns, fill_value="")
        updated = pd.concat([profile, additions], ignore_index=True)
        updated.to_csv(PROFILE_PATH, index=False)
        print(f"Updated node_molecular_profile.csv: {len(profile)} → {len(updated)} rows")
    else:
        print("No new rows to add to node_molecular_profile.csv.")


if __name__ == "__main__":
    main()
