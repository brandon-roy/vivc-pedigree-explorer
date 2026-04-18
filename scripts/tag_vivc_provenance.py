"""
tag_vivc_provenance.py
======================
Annotate vivc_confirmed_marker_support.csv with provenance information.

For each VIVC pair, the `all_dois` field reveals which original paper(s) VIVC
is citing. When that DOI matches one of our loaded molecular sources, the VIVC
entry is NOT an independent confirmation — it's VIVC re-publishing data from
that paper. This script tags those entries so the app can display provenance
honestly and avoid inflating source counts.

Adds column: `vivc_derived_from`
  - Empty string  → VIVC is citing a source we DON'T have loaded (genuinely new info)
  - "Lacombe 2013" → VIVC is re-publishing Lacombe 2013 data (already loaded separately)
  - "Laucou 2018"  → same for Laucou 2018
  - etc.

Also updates `study_reference` for derived entries to say
"VIVC passport (via Lacombe 2013)" so tooltips are honest.

Outputs:
  data/vivc_confirmed_marker_support.csv  — updated in place
  data/vivc_provenance_report.csv         — audit report with all DOI mappings
"""

import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data"

# DOIs for sources we have loaded as separate molecular files
LOADED_DOI_TO_SOURCE = {
    "10.1007/s00122-012-1988-2":   "Lacombe 2013",
    "10.1371/journal.pone.0192540":"Laucou 2018",
    "10.3389/fpls.2020.605934":    "Lacombe & Emanuelli 2020",
    "10.1073/pnas.1009363108":     "Myles 2011",
    "10.1038/s41467-019-09135-8":  "Liang 2019",
    "10.3389/fpls.2024.1391679":   "Tello 2024",
    "10.5344/ajev.2019.18054":     "Migliaro 2019",
    "10.3389/fpls.2020.00127":     "Cunha 2020",
    "10.3390/biology10121279":     "Margaryan 2021",
    "10.1007/s00122-010-1411-9":   "Cipriani 2010",
}

# Human-readable labels for the major unloaded sources (for reference)
UNLOADED_DOI_LABELS = {
    "10.3389/fpls.2020.00127":              "Cunha et al. 2020 (Portuguese varieties, Front. Plant Sci.)",
    "10.1007/s00122-010-1411-9":            "Cipriani et al. 2010 (1,005 accessions SSR profile, TAG 121)",
    "10.5344/ajev.2019.18054":              "Am. J. Enol. Vitic. 2019 (rootstock hybrids parentage)",
    "10.3390/biology10121279":              "Margaryan et al. 2021 (Armenian varieties, Biology 10:1279)",
    "10.1038/s41598-020-71918-7":           "Maraš et al. 2020 (Croatian/Dalmatian varieties, Sci. Rep.)",
    "10.1016/j.scienta.2017.02.044":        "Schneider et al. 2017 (Italian varieties, Sci. Hortic.)",
    "10.1007/s00122-019-03320-5":           "TAG 2019 parentage (French/rootstock varieties)",
    "10.3390/genes11070737":                "Genes 2020 parentage (Hungarian/Croatian varieties)",
    "10.1007/s11295-014-0746-9":            "Tree Genetics & Genomes 2014 (SSR parentage study)",
    "10.1038/s41598-020-72799-6":           "Scientific Reports 2020 parentage study",
    "10.1371/journal.pone.0240665":         "PLOS ONE 2020 (Italian varieties, D'Onofrio?)",
    "10.5073/vitis.2015.54.special-issue.81-86": "Vitis 2015 SI (Iberian varieties, SSR panel)",
    "10.5073/vitis.2015.54.special-issue.59-65": "Vitis 2015 SI (Iberian varieties, continued)",
    "10.1111/ajgw.12282":                   "Aust. J. Grape Wine Res. 2018 (Australian/NZ varieties)",
    "10.1111/ajgw.12561":                   "Aust. J. Grape Wine Res. 2020 (Australian varieties)",
    "10.1126/science.285.5433.1562":        "Bowers et al. 1999 Science (Chardonnay/Gamay/Pinot parentage)",
    "10.5344/ajev.2016.16068":              "Am. J. Enol. Vitic. 2016 (SSR parentage)",
    "10.5344/ajev.2011.11052":              "Am. J. Enol. Vitic. 2011 (SSR parentage)",
    "10.20870/oeno-one.2021.55.3.4628":     "OENO One 2021 parentage study",
    "10.3103/S0095452718050031":            "Russian J. Genetics 2018 (Central Asian varieties)",
    "10.3390/plants10122696":               "Plants 2021 parentage study",
    "10.5073/vitis.2021.60.189-193":        "Vitis 2021 (SSR parentage)",
    "10.1134/S2635167622050135":            "Russian Genetics 2022 (grapevine varieties)",
    "10.21273/JASHS.133.4.559":             "Riaz et al. 2008 JASHS (Muscadine / Vitis rotundifolia, 67 cultivars SSR)",
}


def get_doi_list(row) -> list[str]:
    """Extract all DOIs from a VIVC row's all_dois or doi field."""
    combined = str(row.get("all_dois", "") or "")
    if not combined or combined == "nan":
        combined = str(row.get("doi", "") or "")
    return [d.strip() for d in combined.split(";") if d.strip() and d.strip() != "nan"]


def main() -> None:
    vivc = pd.read_csv(DATA / "vivc_confirmed_marker_support.csv", low_memory=False)
    print(f"Loaded {len(vivc)} VIVC pairs")

    derived_from_col = []
    primary_doi_col  = []
    unloaded_doi_col = []

    for _, row in vivc.iterrows():
        dois = get_doi_list(row)

        loaded_sources  = []
        unloaded_dois   = []
        for doi in dois:
            if doi in LOADED_DOI_TO_SOURCE:
                loaded_sources.append(LOADED_DOI_TO_SOURCE[doi])
            else:
                unloaded_dois.append(doi)

        derived_from_col.append("; ".join(sorted(set(loaded_sources))))
        primary_doi_col.append(dois[0] if dois else "")
        unloaded_doi_col.append("; ".join(unloaded_dois))

    # Use explicit "none" sentinel so independent rows survive CSV round-trip as non-NaN
    vivc["vivc_derived_from"]  = [x if x else "none" for x in derived_from_col]
    vivc["vivc_primary_doi"]   = primary_doi_col
    vivc["vivc_unloaded_dois"] = unloaded_doi_col

    # Update study_reference for derived entries
    def make_ref(row):
        derived = str(row.get("vivc_derived_from", "")).strip()
        if derived and derived != "none":
            return f"VIVC passport (via {derived})"
        else:
            return "VIVC passport (pedigree confirmed by markers)"

    vivc["study_reference"] = vivc.apply(make_ref, axis=1)

    vivc.to_csv(DATA / "vivc_confirmed_marker_support.csv", index=False)
    print(f"Updated vivc_confirmed_marker_support.csv with provenance tags")

    # ── Provenance report ─────────────────────────────────────────────────────
    derived     = vivc[vivc["vivc_derived_from"] != "none"]
    independent = vivc[vivc["vivc_derived_from"] == "none"]

    print(f"\n{'='*60}")
    print(f"VIVC PROVENANCE REPORT")
    print(f"{'='*60}")
    print(f"Total VIVC pairs:                  {len(vivc):>6}")
    print(f"Derived from loaded sources:       {len(derived):>6}  (double-counting risk)")
    print(f"Independent / new:                 {len(independent):>6}")
    print()

    print("Breakdown by derived source:")
    for src, cnt in vivc[vivc["vivc_derived_from"] != "none"]["vivc_derived_from"].value_counts().items():
        print(f"  {src:<40} {cnt:>5}")

    print()
    print("Independent VIVC pairs by unloaded source (top 15):")
    from collections import Counter
    unloaded_ctr = Counter()
    for _, r in independent.iterrows():
        for doi in str(r.get("vivc_unloaded_dois", "")).split(";"):
            doi = doi.strip()
            if doi:
                label = UNLOADED_DOI_LABELS.get(doi, doi)
                unloaded_ctr[label] += 1
    for label, cnt in unloaded_ctr.most_common(15):
        print(f"  {cnt:>4}x  {label}")

    no_doi = independent[independent["vivc_primary_doi"].str.strip().isin(["", "nan"])]
    print(f"\nVIVC pairs with no DOI whatsoever: {len(no_doi)}")

    # Save audit report
    report = vivc[["child_variety","parent_variety","vivc_derived_from",
                    "vivc_primary_doi","vivc_unloaded_dois","study_reference"]].copy()
    report.to_csv(DATA / "vivc_provenance_report.csv", index=False)
    print(f"\nSaved vivc_provenance_report.csv")


if __name__ == "__main__":
    main()
