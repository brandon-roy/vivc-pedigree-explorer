#!/usr/bin/env python3
"""
merge_scraped_dois.py
=====================
After the VIVC DOI scraper completes, this script:

1. Loads vivc_doi_scraped.csv (scraped bibliographies)
2. Loads vivc_confirmed_marker_support.csv (original marker data)
3. For each (child_variety, parent_variety) pair in the marker CSV,
   finds the best DOI from the scraped data
4. Writes an updated vivc_confirmed_marker_support.csv with real DOIs

"Best DOI" priority:
  - Prefer entries with a valid DOI (starts with "10.")
  - Among those, prefer the most-cited source (most appearances across varieties)
  - Fall back to any DOI found for the variety
"""

import os
import re
import pandas as pd
import numpy as np

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SCRAPED_CSV = os.path.join(DATA_DIR, "vivc_doi_scraped.csv")
SOURCE_CSV  = os.path.join(DATA_DIR, "vivc_confirmed_marker_support.csv")
OUTPUT_CSV  = os.path.join(DATA_DIR, "vivc_confirmed_marker_support.csv")  # overwrite in place
BACKUP_CSV  = os.path.join(DATA_DIR, "vivc_confirmed_marker_support_BACKUP.csv")


def valid_doi(d: str) -> bool:
    """Return True if d looks like a real DOI."""
    if not d or not isinstance(d, str):
        return False
    d = d.strip()
    return bool(d) and d.startswith("10.") and len(d) > 6


def main():
    # ── Load scraped data ───────────────────────────────────────────────────
    scraped = pd.read_csv(SCRAPED_CSV, dtype=str, keep_default_na=False)
    scraped["child_variety"] = scraped["child_variety"].str.strip().str.upper()
    scraped["doi"]           = scraped["doi"].str.strip()
    scraped["year"]          = scraped["year"].str.strip()

    print(f"Scraped rows:     {len(scraped)}")
    print(f"Scraped varieties: {scraped['child_variety'].nunique()}")

    # Filter to rows with valid DOIs
    doi_rows = scraped[scraped["doi"].apply(valid_doi)].copy()
    print(f"Rows with valid DOI: {len(doi_rows)}")

    # ── Build variety → best DOI mapping ───────────────────────────────────
    # Rank DOIs by frequency across all varieties (popular papers = more trustworthy)
    doi_popularity = (
        doi_rows.groupby("doi")["child_variety"]
        .nunique()
        .reset_index()
        .rename(columns={"child_variety": "variety_count"})
        .sort_values("variety_count", ascending=False)
    )
    print("\nTop DOIs by variety coverage:")
    print(doi_popularity.head(10).to_string(index=False))

    # For each variety: pick DOI from most-popular source first
    doi_rows = doi_rows.merge(doi_popularity, on="doi", how="left")
    doi_rows = doi_rows.sort_values(["child_variety", "variety_count", "year"],
                                     ascending=[True, False, False])

    # Best DOI per variety (highest-popularity source, then newest)
    best_doi = (
        doi_rows.drop_duplicates(subset="child_variety", keep="first")
        [["child_variety", "doi", "source_code", "author", "year", "title"]]
        .set_index("child_variety")
    )
    print(f"\nVarieties with a best DOI: {len(best_doi)}")

    # Build multi-doi string per variety (all unique DOIs, semicolon-separated)
    all_dois = (
        doi_rows.groupby("child_variety")["doi"]
        .apply(lambda s: "; ".join(sorted(s.unique())))
        .rename("all_dois")
    )

    # ── Load and update marker CSV ──────────────────────────────────────────
    markers = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False)
    markers["child_variety"] = markers["child_variety"].str.strip().str.upper()

    # Backup original
    markers.to_csv(BACKUP_CSV, index=False)
    print(f"\nBackup saved to {BACKUP_CSV}")

    # Merge best DOI
    markers = markers.merge(
        best_doi[["doi"]].rename(columns={"doi": "_scraped_doi"}),
        left_on="child_variety",
        right_index=True,
        how="left",
    )
    markers["_scraped_doi"] = markers["_scraped_doi"].fillna("")

    # Update doi column: replace NA/empty with scraped value
    _NA_VALS = {"NA", "N/A", "n/a", "na", "N.A.", "NULL", "null", "None", "none", ""}
    def pick_doi(row):
        existing = str(row["doi"]).strip()
        if existing in _NA_VALS or not valid_doi(existing):
            return row["_scraped_doi"] if row["_scraped_doi"] else "NA"
        return existing

    markers["doi"] = markers.apply(pick_doi, axis=1)
    markers = markers.drop(columns=["_scraped_doi"])

    # Also merge all_dois as a new column (optional enrichment)
    markers = markers.merge(
        all_dois.reset_index(),
        on="child_variety",
        how="left",
    )
    markers["all_dois"] = markers["all_dois"].fillna("")

    # ── Save ────────────────────────────────────────────────────────────────
    markers.to_csv(OUTPUT_CSV, index=False)
    print(f"\nUpdated marker CSV saved to {OUTPUT_CSV}")

    # ── Report ──────────────────────────────────────────────────────────────
    total_rows  = len(markers)
    has_doi     = markers["doi"].apply(valid_doi).sum()
    pct         = round(100 * has_doi / total_rows, 1)
    print(f"\nRows with valid DOI: {has_doi} / {total_rows} ({pct}%)")

    sample = markers[markers["doi"].apply(valid_doi)][["child_variety","doi"]].head(10)
    print("\nSample updated rows:")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
