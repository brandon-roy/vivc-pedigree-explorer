"""
apply_tier1_aliases.py
======================
Applies reviewed tier-1 alias suggestions to marker_name_aliases.csv.
Writes flagged/rejected entries to data/tier1_aliases_flagged.csv for
manual follow-up.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

# ── Decisions ────────────────────────────────────────────────────────────────
# Verified safe to add: spelling/umlaut corrections, unambiguous color/region
# qualifiers for well-known varieties where VIVC has only one matching entry.
APPLY = {
    # Umlaut / encoding normalisations
    "HARSLEVELU":         "HARSLEVELUE",
    "HOSSZUNYELU":        "HOSSZUNYELUE",
    "KEKNYELU":           "KEKNYELUE",
    "SIBIRKOVY":          "SIBIRKOVYI",
    "YALOVA CEKIRDEKSIZ": "YALOVA CEKIRDEKSIZI",
    "MUSKATELLER GELB":   "MUSKATELLER GELBER",
    "PLAVAI":             "PLAVAIE",

    # Gender/color qualifier — single VIVC entry, well-known variety
    "HUMAGNE BLANC":      "HUMAGNE BLANCHE",
    "PRIETO PICUDO":      "PRIETO PICUDO TINTO",
    "PLAVINA":            "PLAVINA CRNA",
    "TORTOZON":           "TORTOZONA TINTA",
    "BACHET":             "BACHET NOIR",
    "REDONDAL":           "REDONDAL BLANC",
    "RAZAKLIJA":          "RAZAKLIJA CRNA",
    "PELOURSIN N":        "PELOURSIN NOIR",   # N = Noir abbreviation
    "ASKERI":             "ASKERI SARI",
    "BALINT":             "BALINT WEISS",
    "JAVOR GROSS":        "JAVOR GROSS WEISS",
    "MALVASIA DI SARDEGNA": "MALVASIA DI SARDEGNA ROSADA",
    "MOSCATEL DE HAMBURGO": "MOSCATEL DE HAMBURGO BRANCO",

    # Seedless / oval qualifier — single match, well-known variety
    "HUBBARD":            "HUBBARD SEEDLESS",
    "NIAGARA BRANCA":     "NIAGARA BRANCA OVAL",
    "SULTANINA BIANCA":   "SULTANINA BIANCA DI RODI",

    # Regional / locality qualifier — single match
    "FRAPPATO":           "FRAPPATO DI VITTORIA",
    "ZAGARESE":           "ZAGARESE DI PUGLIA",
    "MANTONICO NERO":     "MANTONICO NERO INZENGA",
    "CALLET CAS CONCOS":  "CALLET CAS CONCOS BLANCO",
    "VITOUSKA":           "VITOUSKA GARGANIJA",
    "DARANYI IGNAC":      "DARANYI IGNAC MUSKOTALY",
    "BENEDICTO FALSO":    "BENEDICTO FALSO DE ARAGON",
    "CRIOLLA GRANDE":     "CRIOLLA GRANDE SANJUANINA",

    # Minor spelling difference (not umlaut)
    "DAPHNI":             "DAPHNIA",
    "MUSCATE":            "MUSCATEDDU",
    "TORTOZON":           "TORTOZONA TINTA",

    # Clone / selection designator where VIVC only registers the clonal entry
    "BELDI":              "BELDI CT2",
    "EKIM KARA FAUX":     "EKIM KARA FAUX (COLLECTION CONEGLIANO)",

    # Vitis species author-name expansions (botanical nomenclature standard)
    "VITIS BERLANDIERI":    "VITIS BERLANDIERI PLANCHON",
    "VITIS CALIFORNICA":    "VITIS CALIFORNICA BENTHAM",
    "VITIS CANDICANS":      "VITIS CANDICANS ENGELMANN",
    "VITIS CINEREA":        "VITIS CINEREA ENGELMANN VAR. CINEREA",
    "VITIS DOANIANA":       "VITIS DOANIANA MUNSON",
    "VITIS GIRDIANA":       "VITIS GIRDIANA MUNSON",
    "VITIS MONTICOLA":      "VITIS MONTICOLA BUCKLEY",
    "VITIS SHUTTLEWORTHII": "VITIS SHUTTLEWORTHII HOUSE",
}

# ── Flagged — do NOT apply, require domain expert review ─────────────────────
FLAGGED = {
    "MACABEO":         ("MACABEO NEGRO",
                        "Macabeo/Viura is white — MACABEO NEGRO is a different black variety; "
                        "need to check what name the source actually uses"),
    "FRANKENTHAL":     ("FRANKENTHAL BLANC MUSQUE",
                        "Frankenthal is typically NOIR — FRANKENTHAL BLANC MUSQUE is a white muscat variant; "
                        "likely the wrong mapping for a source citing Frankenthal as a parent"),
    "GRENACHE N":      ("GRENACHE NOIR A FLEURS FEMELLES",
                        "'N' = Noir abbreviation, but GRENACHE NOIR A FLEURS FEMELLES is the female-flower sex "
                        "variant, not the main Grenache Noir — would misidentify the variety"),
    "MISSION":         ("MISSION MALE",
                        "MISSION MALE is Muscadinia rotundifolia (different genus) — "
                        "not the California Mission grape which is V. vinifera"),
    "PIROVANO 59":     ("PIROVANO 595",
                        "Number does not match: 59 ≠ 595; likely a different accession"),
    "PIROVANO 60":     ("PIROVANO 604",
                        "Number does not match: 60 ≠ 604; likely a different accession"),
    "DIAGALVES":       ("DIAGALVES FAUX",
                        "FAUX = 'false Diagalves' — a distinct variety misidentified as Diagalves; "
                        "sources citing 'DIAGALVES' likely mean the true variety, not this one"),
    "MOUCHKETNY":      ("MOUCHKETNY FAUX",
                        "Same FAUX issue: sources likely reference true Mouchketny, not the false-labeled entry"),
    "MOSCATEL DE OEIRAS": ("MOSCATEL DE OEIRAS FAUX (COLLECTION BORDEAUX)",
                        "FAUX variety in Bordeaux collection — sources may mean true Moscatel de Oeiras"),
    "BLANC DE DELLYS": ("BLANC DE DELLYS X RHODITES",
                        "X RHODITES suffix marks a specific hybrid cross, not the base variety"),
    "DUNAVSKI MISKET": ("DUNAVSKI MISKET X CARDINAL",
                        "X CARDINAL suffix marks a hybrid cross, not the base variety"),
    "FUJIMINORI":      ("FUJIMINORI (4N)",
                        "(4N) = tetraploid — genetically distinct from the diploid Fujiminori; "
                        "sources likely reference the standard diploid"),
    "NIABELL":         ("NIABELL (4N)",
                        "(4N) = tetraploid — same concern as Fujiminori"),
    "UVA DI TROJA":    ("UVA DI TROJANERO",
                        "TROJANERO suffix substantially changes the name — unclear if these are the same variety"),
    "MOSCATEL DE OEIRAS": ("MOSCATEL DE OEIRAS FAUX (COLLECTION BORDEAUX)",
                        "FAUX + collection qualifier — likely a different accession from what literature cites"),
}

# ── Load existing alias map ───────────────────────────────────────────────────
alias_path = DATA / "marker_name_aliases.csv"
existing = pd.read_csv(alias_path, dtype=str).fillna("")
existing_keys = set(existing["alias"].str.strip().str.upper())

# ── Build new rows ────────────────────────────────────────────────────────────
new_rows = []
skipped = []
for alias, canon in APPLY.items():
    if alias in existing_keys:
        skipped.append(alias)
        continue
    new_rows.append({
        "alias":               alias,
        "canonical_vivc_name": canon,
        "source":              "tier1_auto",
        "match_score":         95.0,
    })

if new_rows:
    new_df = pd.DataFrame(new_rows)
    updated = pd.concat([existing, new_df], ignore_index=True)
    updated.to_csv(alias_path, index=False)
    print(f"Added {len(new_rows)} new aliases to {alias_path.name}")
    for r in new_rows:
        print(f"  {r['alias']:<35} → {r['canonical_vivc_name']}")
else:
    print("No new aliases to add (all already present).")

if skipped:
    print(f"\nSkipped {len(skipped)} already-present: {', '.join(skipped)}")

# ── Write flagged report ──────────────────────────────────────────────────────
flag_rows = [
    {"unresolved_name": k, "rejected_candidate": v[0], "reason": v[1]}
    for k, v in FLAGGED.items()
]
flag_df = pd.DataFrame(flag_rows)
flag_path = DATA / "tier1_aliases_flagged.csv"
flag_df.to_csv(flag_path, index=False)
print(f"\nFlagged {len(flag_df)} entries for expert review → {flag_path.name}")
for _, r in flag_df.iterrows():
    print(f"  ✗ {r['unresolved_name']:<30} (rejected: {r['rejected_candidate']})")
    print(f"      {r['reason']}")
