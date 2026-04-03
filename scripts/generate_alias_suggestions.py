"""
generate_alias_suggestions.py
==============================
Reads unresolved_accession_synonyms.csv and the passport to produce
data/alias_suggestions.csv — a tiered, manually-reviewable list of
suggested alias → canonical_vivc_name mappings to add to
marker_name_aliases.csv.

Confidence tiers
----------------
tier1  Exact prefix, single match: "TEMPRANILLO" starts exactly one passport
       name (e.g. "TEMPRANILLO TINTO"). High confidence — likely just the
       abbreviated name.
tier2  Exact prefix, multiple matches: "CHASSELAS" starts 65 passport names.
       Manual selection required.
tier3  Suffix / contains match: "MUSCAT D'ALEXANDRIE" → "MUSCAT OF ALEXANDRIA"
       via best token overlap. Needs human verification.
tier4  No passport candidate found. Genuinely new accession or source-specific
       local name.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import pandas as pd

BASE = Path(__file__).parent.parent
DATA = BASE / "data"

_NA = {"", "NA", "N/A", "n/a", "na", "N.A.", "NULL", "null", "None", "none", "nan"}


def _clean(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    return "" if s in _NA or s.lower() in _NA else s.upper()


# ── Load inputs ───────────────────────────────────────────────────────────────
unresolved = pd.read_csv(DATA / "unresolved_accession_synonyms.csv", dtype=str).fillna("")
master      = pd.read_csv(DATA / "master_accession_sheet.csv",       dtype=str).fillna("")
existing_aliases = pd.read_csv(DATA / "marker_name_aliases.csv",     dtype=str).fillna("")

passport_names: set[str] = set(
    master.loc[master["in_passport"] == "True", "vivc_prime_name"]
)

# Map canonical name → n_sources for ranking when multiple candidates exist
name_to_nsources: dict[str, int] = dict(
    zip(master["vivc_prime_name"],
        pd.to_numeric(master["n_sources"], errors="coerce").fillna(0).astype(int))
)

already_aliased: set[str] = set(_clean(v) for v in existing_aliases.get("alias", pd.Series()).tolist())

# ── Build passport token index ────────────────────────────────────────────────
token_idx: dict[str, list[str]] = defaultdict(list)
for pn in passport_names:
    for tok in pn.split():
        if len(tok) >= 4:
            token_idx[tok].append(pn)


def classify(name: str) -> tuple[str, str, list[str]]:
    """
    Returns (tier, best_candidate, all_candidates_list).
    """
    if not name or name in passport_names:
        return "already_matched", name, [name]

    # Tier 1/2: starts-with
    starts = sorted(
        [pn for pn in passport_names if pn.startswith(name)],
        key=lambda pn: (-name_to_nsources.get(pn, 0), len(pn), pn),
    )
    if len(starts) == 1:
        return "tier1", starts[0], starts
    if len(starts) > 1:
        return "tier2", starts[0], starts[:10]

    # Tier 3: token overlap
    tokens = [t for t in name.split() if len(t) >= 4]
    if not tokens:
        return "tier4", "", []

    candidates: dict[str, int] = defaultdict(int)
    for tok in tokens:
        for pn in token_idx.get(tok, []):
            candidates[pn] += 1

    min_hits = max(1, len(tokens) // 2)
    qualified = {pn: cnt for pn, cnt in candidates.items() if cnt >= min_hits}
    if not qualified:
        return "tier4", "", []

    ranked = sorted(
        qualified.keys(),
        key=lambda pn: (-qualified[pn], -name_to_nsources.get(pn, 0), len(pn), pn),
    )
    return "tier3", ranked[0], ranked[:10]


# ── Generate suggestions ──────────────────────────────────────────────────────
rows = []
for _, r in unresolved.iterrows():
    name = _clean(r["vivc_prime_name"])
    if not name or name in already_aliased:
        continue

    n_src = int(r.get("n_sources", 0) or 0)
    tier, best, all_cands = classify(name)

    # Which sources reference this name
    src_flags = {
        col.replace("in_", "").replace("_", " "): r.get(col, "False") == "True"
        for col in r.index if col.startswith("in_") and col != "in_passport"
    }
    src_list = "; ".join(k for k, v in src_flags.items() if v)

    # Sample synonym names from individual sources
    synonyms_seen = "; ".join(filter(None, [
        r.get("name_in_lacombe_2013", ""),
        r.get("name_in_laucou_2018", ""),
        r.get("name_in_lac_emanuelli_2020", ""),
        r.get("name_in_grin_pedigree", ""),
        r.get("name_in_grin_alignment", ""),
    ]))

    rows.append({
        "unresolved_name":      name,
        "n_sources":            n_src,
        "tier":                 tier,
        "suggested_canonical":  best,
        "all_candidates":       " | ".join(all_cands),
        "sources_referencing":  src_list,
        "synonyms_from_sources": synonyms_seen,
        "grin_accession_id":    r.get("grin_accession_id", ""),
        "action":               "",   # blank column for manual curation: add/skip/query
        "notes":                "",
    })

out = pd.DataFrame(rows).sort_values(
    ["tier", "n_sources", "unresolved_name"],
    ascending=[True, False, True],
).reset_index(drop=True)

out_path = DATA / "alias_suggestions.csv"
out.to_csv(out_path, index=False)

# Summary
print(f"Wrote {len(out):,} rows → {out_path}")
for tier in ["tier1", "tier2", "tier3", "tier4"]:
    n = (out["tier"] == tier).sum()
    print(f"  {tier}: {n:,}")

print("\nTier-1 high-confidence suggestions (sample):")
t1 = out[out["tier"] == "tier1"].head(30)
print(t1[["unresolved_name", "suggested_canonical", "n_sources", "sources_referencing"]].to_string(index=False))
