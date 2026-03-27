#!/usr/bin/env python3
"""
VIVC DOI Scraper
================
For each unique child variety in vivc_confirmed_marker_support.csv,
fetches the VIVC microsatellite-parentage bibliography page and extracts:
  - Source code (lit_id)
  - Author / year / title
  - DOI (from doi.org links in the page)

Endpoint per variety:
  https://www.vivc.de/index.php
    ?var={vivc_no}
    &EvaAnalysisMikrosatellitenAbstammungenSearch[eva_analysis_pkValue]={vivc_no}
    &r=eva-analysis-mikrosatelliten-abstammungen/index

Output: data/vivc_doi_scraped.csv
Columns: child_variety, vivc_no, source_code, author, year, title, doi, n_loci

Progress is saved incrementally (flush every 25 varieties).
Supports resume — already-processed varieties are skipped.
"""

import csv
import os
import re
import sys
import time
import logging
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_URL     = "https://www.vivc.de/index.php"
DELAY_SEC    = 0.6    # polite crawl delay between requests
MAX_RETRIES  = 3
RETRY_DELAY  = 8      # seconds to wait after a failed request
DATA_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_CSV   = os.path.join(DATA_DIR, "vivc_doi_scraped.csv")
PROGRESS_LOG = os.path.join(DATA_DIR, "vivc_doi_scraper_progress.log")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_FIELDS = [
    "child_variety", "vivc_no", "source_code",
    "author", "year", "title", "doi", "n_loci",
]

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(PROGRESS_LOG, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── HTTP Session ───────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_confirmed_varieties() -> list[str]:
    """Unique child variety names from vivc_confirmed_marker_support.csv."""
    path = os.path.join(DATA_DIR, "vivc_confirmed_marker_support.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    varieties = sorted(df["child_variety"].str.strip().str.upper().unique())
    log.info(f"Confirmed varieties to process: {len(varieties)}")
    return varieties


def load_vivc_no_map() -> dict[str, str]:
    """{UPPER_PRIME_NAME → vivc_no} from vivc_supplementary.csv."""
    path = os.path.join(DATA_DIR, "vivc_supplementary.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["prime_name"] = df["prime_name"].str.strip().str.upper()
    df["vivc_no"]    = df["vivc_no"].str.strip()
    # Drop rows where vivc_no is empty/NA
    df = df[df["vivc_no"].str.match(r"^\d+$")]
    mapping = dict(zip(df["prime_name"], df["vivc_no"]))
    log.info(f"VIVC number map: {len(mapping)} entries")
    return mapping


def load_done_varieties() -> set[str]:
    """Return set of child_variety values already in the output CSV."""
    if not os.path.exists(OUTPUT_CSV):
        return set()
    try:
        df = pd.read_csv(OUTPUT_CSV, dtype=str, keep_default_na=False)
        return set(df["child_variety"].str.strip().str.upper().unique())
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
# Per-variety page fetcher + parser
# ─────────────────────────────────────────────────────────────────────────────
def fetch_variety_references(vivc_no: str) -> list[dict]:
    """
    Fetch the VIVC marker-parentage bibliography page for vivc_no.

    Returns a list of dicts with keys:
        source_code, author, year, title, doi, n_loci
    """
    params = {
        "var": vivc_no,
        "EvaAnalysisMikrosatellitenAbstammungenSearch[eva_analysis_pkValue]": vivc_no,
        "r": "eva-analysis-mikrosatelliten-abstammungen/index",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(BASE_URL, params=params, timeout=25)
            resp.raise_for_status()
            break
        except Exception as e:
            log.warning(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                log.error(f"    Giving up on vivc_no={vivc_no}")
                return []

    soup = BeautifulSoup(resp.text, "html.parser")

    # The data table has class "kv-grid-table"
    table = soup.find("table", {"class": re.compile(r"kv-grid-table")})
    if not table:
        # Fallback: any table in the content area
        content_div = soup.find("div", {"class": re.compile(r"eva-analysis")})
        if content_div:
            table = content_div.find("table")

    if not table:
        return []

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")

    records = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        source_code = cells[0].get_text(strip=True)
        bib_cell    = cells[1]
        n_loci      = cells[2].get_text(strip=True) if len(cells) > 2 else ""

        bib_text = bib_cell.get_text("\n", strip=True)
        lines    = [l.strip() for l in bib_text.split("\n") if l.strip()]

        # DOI: look for <a href="https://doi.org/..."> or https://dx.doi.org/
        doi = ""
        for a in bib_cell.find_all("a", href=True):
            href = a["href"]
            if "doi.org/" in href:
                doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", href).strip().rstrip(".,;)")
                break
        # Fallback: scan text for raw DOIs (must start with 10.)
        if not doi:
            m = re.search(r'\b(10\.\d{4,}/\S+)', bib_text)
            if m:
                doi = m.group(1).rstrip(".,;)")

        # Year: look for standalone 4-digit year (not embedded in DOI/URLs)
        # Strip DOI and URLs from text before searching for year
        year_text = re.sub(r'https?://\S+', '', bib_text)   # remove URLs
        year_text = re.sub(r'10\.\d{4,}/\S+', '', year_text)  # remove raw DOIs
        year = ""
        # Search lines in reverse (year usually at end of citation)
        clean_lines = [l.strip() for l in year_text.split("\n") if l.strip()]
        for line in reversed(clean_lines):
            m = re.search(r'\b(1[9][0-9]{2}|20[012][0-9])\b', line)
            if m:
                year = m.group(1)
                break

        # Author: first non-empty line (usually author list)
        author = lines[0][:200] if lines else ""

        # Title: second line if present
        title = lines[1][:250] if len(lines) > 1 else ""

        records.append({
            "source_code": source_code,
            "author":      author,
            "year":        year,
            "title":       title,
            "doi":         doi,
            "n_loci":      n_loci,
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main scraping loop
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 65)
    log.info(f"VIVC DOI Scraper started at {datetime.now().isoformat()}")
    log.info("=" * 65)

    varieties = load_confirmed_varieties()
    vivc_map  = load_vivc_no_map()
    done      = load_done_varieties()

    if done:
        log.info(f"Resuming — {len(done)} varieties already done, skipping.")

    # Open output file (append mode for resume support)
    write_header = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    out_f  = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_FIELDS)
    if write_header:
        writer.writeheader()

    total       = len(varieties)
    processed   = 0
    skipped     = 0
    no_vivc     = 0
    no_refs     = 0
    total_dois  = 0
    last_flush  = time.time()

    for i, variety in enumerate(varieties, 1):
        # Skip already processed
        if variety in done:
            skipped += 1
            continue

        vivc_no = vivc_map.get(variety)
        if not vivc_no:
            log.debug(f"  [{i}/{total}] '{variety}' — no vivc_no, skipping")
            no_vivc += 1
            # Write a placeholder so we don't retry
            writer.writerow({f: "" for f in OUTPUT_FIELDS} | {"child_variety": variety})
            done.add(variety)
            continue

        log.info(f"[{i}/{total}] {variety}  (vivc_no={vivc_no})")
        refs = fetch_variety_references(vivc_no)

        if refs:
            dois_here = sum(1 for r in refs if r["doi"])
            log.info(f"  → {len(refs)} reference(s), {dois_here} with DOI")
            total_dois += dois_here
            for ref in refs:
                writer.writerow({"child_variety": variety, "vivc_no": vivc_no, **ref})
        else:
            log.info(f"  → No references found")
            no_refs += 1
            writer.writerow({f: "" for f in OUTPUT_FIELDS} | {
                "child_variety": variety,
                "vivc_no": vivc_no,
            })

        done.add(variety)
        processed += 1

        # Flush & log checkpoint every 25 varieties
        if processed % 25 == 0:
            out_f.flush()
            elapsed = time.time() - last_flush
            rate = 25 / elapsed if elapsed > 0 else 0
            remaining = total - i
            eta_min = (remaining / rate / 60) if rate > 0 else 0
            log.info(
                f"\n── Checkpoint ─────────────────────────────────────────\n"
                f"   Processed : {processed}  /  {total}\n"
                f"   Skipped   : {skipped}\n"
                f"   No vivc_no: {no_vivc}\n"
                f"   No refs   : {no_refs}\n"
                f"   Total DOIs: {total_dois}\n"
                f"   Rate      : {rate:.1f} req/s\n"
                f"   ETA       : ~{eta_min:.0f} min\n"
                f"────────────────────────────────────────────────────────\n"
            )
            last_flush = time.time()

        time.sleep(DELAY_SEC)

    out_f.flush()
    out_f.close()

    log.info("=" * 65)
    log.info("SCRAPING COMPLETE")
    log.info(f"  Processed  : {processed}")
    log.info(f"  Skipped    : {skipped}  (already done)")
    log.info(f"  No vivc_no : {no_vivc}")
    log.info(f"  No refs    : {no_refs}")
    log.info(f"  Total DOIs : {total_dois}")
    log.info(f"  Output     : {OUTPUT_CSV}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
