"""
scrape_ssr_sources.py
─────────────────────
Pulls SSR microsatellite genotype data from three publicly accessible sources:

  1. VIVC "Microsatellite data of varieties" – POST search, then CSV export
  2. Lacombe et al. 2013 (TAG) supplementary PDFs – freely hosted on Springer
  3. GRIN-Global web service – SSR observations for Vitis accessions

Output
──────
  data/ssr_vivc.csv          – VIVC 9-locus profiles
  data/ssr_lacombe2013.csv   – Lacombe 2013 allele table
  data/ssr_grin.csv          – GRIN SSR observations
  data/ssr_crosssource_snippet.csv – side-by-side comparison for key varieties
"""

import io, re, time, textwrap
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
from pathlib import Path

OUT = Path(__file__).parent.parent / "data"
OUT.mkdir(exist_ok=True)

# Standard 9-locus marker set (GENRES081 / GrapeGen06 consensus)
NINE_LOCI = ["VVS2", "VVMD5", "VVMD7", "VVMD25", "VVMD27", "VVMD28", "VVMD32", "VrZAG62", "VrZAG79"]

# Reference varieties to anchor cross-source comparison
REFERENCE_VARIETIES = [
    "CABERNET SAUVIGNON", "CABERNET FRANC", "SAUVIGNON BLANC",
    "PINOT NOIR", "CHARDONNAY BLANC", "RIESLING WEISS",
    "MERLOT NOIR", "SYRAH", "MUSCAT A PETITS GRAINS BLANCS",
    "GEWUERZTRAMINER", "GRENACHE NOIR", "TEMPRANILLO",
    "NEBBIOLO", "SANGIOVESE", "GARNACHA TINTA",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 1 — VIVC Microsatellite data of varieties
# ─────────────────────────────────────────────────────────────────────────────

def scrape_vivc_ssr(varieties: list[str]) -> pd.DataFrame:
    """
    POST to VIVC SSR search, parse the result HTML table.
    Falls back to parsing the 'by profile' reference table if POST fails.
    """
    print("── SOURCE 1: VIVC SSR ──────────────────────────────────────────────")
    session = requests.Session()
    session.headers.update(HEADERS)

    base = "https://www.vivc.de"

    # Step 1: GET the search page to obtain CSRF token & session cookie
    try:
        r = session.get(f"{base}/index.php?r=eva-analysis-mikrosatelliten-vivc/index2",
                        timeout=20)
        soup = BeautifulSoup(r.text, "html.parser")
        csrf = soup.find("meta", {"name": "csrf-token"})
        csrf_token = csrf["content"] if csrf else ""
        print(f"  Session established; CSRF={'yes' if csrf_token else 'no'}")
    except Exception as e:
        print(f"  Could not reach VIVC: {e}")
        return pd.DataFrame()

    # Step 2: POST the search form with all reference variety names
    form_data = {"_csrf": csrf_token}
    for v in varieties:
        form_data.setdefault(
            "EvaAnalysisMikrosatellitenVivcSearch[leitname_list][]", []
        )
        if isinstance(form_data["EvaAnalysisMikrosatellitenVivcSearch[leitname_list][]"], list):
            form_data["EvaAnalysisMikrosatellitenVivcSearch[leitname_list][]"].append(v)

    try:
        r = session.post(
            f"{base}/index.php?r=eva-analysis-mikrosatelliten-vivc/resultmsatvar",
            data=form_data, timeout=30,
        )
        print(f"  POST status: {r.status_code}  len={len(r.text)}")
        rows = _parse_vivc_ssr_table(r.text, source="VIVC_direct")
        if rows:
            print(f"  Parsed {len(rows)} rows from POST result")
            return pd.DataFrame(rows)
    except Exception as e:
        print(f"  POST failed: {e}")

    # Fallback: parse the 'by profile' reference page (shows 4 fixed varieties)
    print("  Falling back to 'by profile' reference table…")
    try:
        r = session.get(f"{base}/index.php?r=eva-analysis-mikrosatelliten-vivc/index", timeout=20)
        rows = _parse_vivc_ssr_table(r.text, source="VIVC_reference")
        print(f"  Parsed {len(rows)} reference rows")
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"  Fallback also failed: {e}")
        return pd.DataFrame()


def _parse_vivc_ssr_table(html: str, source: str = "VIVC") -> list[dict]:
    """Parse SSR allele table from VIVC HTML response."""
    soup = BeautifulSoup(html, "html.parser")
    rows_out = []

    # Find main data table (>4 columns)
    for table in soup.find_all("table"):
        trows = table.find_all("tr")
        if len(trows) < 2:
            continue
        # Check if this looks like an SSR table
        header_text = " ".join(th.get_text() for th in trows[0].find_all(["th", "td"]))
        if not any(m in header_text for m in ("VVS2", "VVMD5", "VrZAG")):
            continue

        # Parse header: each locus spans two columns (A1, A2)
        headers_raw = [th.get_text(strip=True) for th in trows[0].find_all(["th", "td"])]
        # Build column map: variety name col + allele pairs
        col_names = []
        for h in headers_raw:
            col_names.append(h)

        for tr in trows[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
            if not cells or all(c == "" for c in cells):
                continue
            # Skip legend rows
            if any(kw in " ".join(cells) for kw in ("light blue", "light green", "rose")):
                continue
            variety = cells[0].strip()
            if not variety or variety in ("Reference variety", ""):
                continue
            # Alleles start at col 1, in pairs
            allele_vals = cells[1:]
            row = {"variety": variety, "source": source}
            for i, locus in enumerate(NINE_LOCI):
                idx_a1 = i * 2
                idx_a2 = i * 2 + 1
                a1 = allele_vals[idx_a1] if idx_a1 < len(allele_vals) else ""
                a2 = allele_vals[idx_a2] if idx_a2 < len(allele_vals) else ""
                row[f"{locus}_A1"] = a1
                row[f"{locus}_A2"] = a2
                row[locus] = f"{a1}/{a2}" if a1 and a2 else ""
            rows_out.append(row)
        break  # found the right table

    return rows_out


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 2 — Lacombe et al. 2013 supplementary PDFs
# ─────────────────────────────────────────────────────────────────────────────

LACOMBE_PDFS = {
    1: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM1_ESM.pdf",
    2: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM2_ESM.pdf",
    3: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM3_ESM.pdf",
    4: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM4_ESM.pdf",
    5: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM5_ESM.pdf",
    6: "https://static-content.springer.com/esm/art%3A10.1007%2Fs00122-012-1988-2/MediaObjects/122_2012_1988_MOESM6_ESM.pdf",
}

def scrape_lacombe_pdfs() -> pd.DataFrame:
    """Download Lacombe 2013 supplementary PDFs and extract SSR allele tables."""
    print("\n── SOURCE 2: Lacombe et al. 2013 (TAG) ────────────────────────────")
    session = requests.Session()
    session.headers.update({**HEADERS, "Referer": "https://link.springer.com/"})

    all_rows = []
    pdf_dir = OUT / "_lacombe_pdfs"
    pdf_dir.mkdir(exist_ok=True)

    for num, url in LACOMBE_PDFS.items():
        pdf_path = pdf_dir / f"MOESM{num}.pdf"
        # Download if not cached
        if not pdf_path.exists():
            try:
                r = session.get(url, timeout=30)
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
                    pdf_path.write_bytes(r.content)
                    print(f"  Downloaded MOESM{num} ({len(r.content)//1024} KB)")
                else:
                    print(f"  MOESM{num}: HTTP {r.status_code} — skipping")
                    continue
            except Exception as e:
                print(f"  MOESM{num}: {e} — skipping")
                continue
        else:
            print(f"  MOESM{num}: using cached file")

        # Parse PDF
        rows = _parse_lacombe_pdf(pdf_path, supplement_num=num)
        print(f"    → {len(rows)} genotype rows extracted")
        all_rows.extend(rows)
        time.sleep(0.5)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _parse_lacombe_pdf(pdf_path: Path, supplement_num: int) -> list[dict]:
    """
    Extract SSR genotype rows from a Lacombe 2013 supplementary PDF.
    The PDFs contain tables with variety name + 20 SSR loci (allele pairs).
    We capture the 9 GENRES081 loci if present.
    """
    rows_out = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # First pass: detect columns from page 1
            first_page_text = pdf.pages[0].extract_text() or ""
            print(f"    PDF {supplement_num} page 1 preview:")
            for line in first_page_text.split("\n")[:8]:
                print(f"      {line}")

            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    header = [str(c).strip() if c else "" for c in table[0]]
                    # Check if this has variety name + numeric allele data
                    if len(header) < 5:
                        continue
                    # Detect column positions for each locus
                    locus_cols = {}
                    for i, h in enumerate(header):
                        for locus in NINE_LOCI + ["VVS", "VVMD", "ZAG"]:
                            if locus in h.upper():
                                locus_cols[i] = h
                    # Parse data rows
                    for row in table[1:]:
                        if not row or not row[0]:
                            continue
                        variety = str(row[0]).strip().upper()
                        if not variety or len(variety) < 2:
                            continue
                        # Skip header-like rows
                        if any(kw in variety for kw in ("NAME", "VARIETY", "CULTIVAR", "LOCUS")):
                            continue
                        row_dict = {"variety": variety, "source": f"Lacombe2013_S{supplement_num}"}
                        for col_idx, col_name in locus_cols.items():
                            val = str(row[col_idx]).strip() if col_idx < len(row) and row[col_idx] else ""
                            row_dict[col_name] = val
                        rows_out.append(row_dict)

                # Also try text-based extraction for tables not captured structurally
                text = page.extract_text() or ""
                rows_out.extend(_parse_lacombe_text_table(text, supplement_num))

    except Exception as e:
        print(f"    PDF parse error: {e}")
    return rows_out


def _parse_lacombe_text_table(text: str, supplement_num: int) -> list[dict]:
    """
    Fallback: parse allele data from raw text lines.
    Lacombe PDFs often use format: VARIETY_NAME  allele1 allele2 allele1 allele2...
    """
    rows = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        # Heuristic: first token is variety name (letters), rest are numeric alleles
        name_parts = []
        allele_parts = []
        for p in parts:
            if re.match(r"^\d{2,4}$", p):
                allele_parts.append(p)
            elif re.match(r"^[A-Z][A-Z\s\-\'\.]+$", p) and not allele_parts:
                name_parts.append(p)
        if len(allele_parts) >= 10 and name_parts:
            variety = " ".join(name_parts)
            row = {"variety": variety, "source": f"Lacombe2013_S{supplement_num}_text"}
            # Assign alleles to loci in pairs
            for i, locus in enumerate(NINE_LOCI):
                idx = i * 2
                if idx + 1 < len(allele_parts):
                    a1, a2 = allele_parts[idx], allele_parts[idx + 1]
                    row[f"{locus}_A1"] = a1
                    row[f"{locus}_A2"] = a2
                    row[locus] = f"{a1}/{a2}"
            rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# SOURCE 3 — GRIN-Global SSR observations
# ─────────────────────────────────────────────────────────────────────────────

def scrape_grin_ssr() -> pd.DataFrame:
    """
    Query the GRIN-Global web service for SSR marker observations on Vitis.
    GRIN stores SSR data as characterization observations.
    """
    print("\n── SOURCE 3: GRIN-Global SSR ───────────────────────────────────────")

    # GRIN has a public REST-like SOAP web service
    # Try the GetData endpoint for characterization data (SSR markers)
    url = "https://npgsweb.ars-grin.gov/gringlobal/gui.asmx/GetData"
    payload = {
        "dataviewName": "accession_observation",
        "filterExpression": "crop_name like 'Vitis%' and obs_method_no in (select obs_method_no from obs_method where obs_method_name like '%SSR%' or obs_method_name like '%microsatellite%')",
        "sortExpression": "",
        "startRowIndex": "0",
        "maximumRows": "200",
    }
    try:
        r = requests.get(url, params=payload, timeout=20, headers=HEADERS)
        print(f"  GRIN GetData status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Response preview: {r.text[:500]}")
    except Exception as e:
        print(f"  GRIN web service error: {e}")

    # Also try the direct accession observation query
    url2 = "https://npgsweb.ars-grin.gov/gringlobal/query/accessionobservations"
    try:
        r2 = requests.get(url2, params={"taxon": "Vitis vinifera", "format": "json"}, timeout=20, headers=HEADERS)
        print(f"  GRIN observations query status: {r2.status_code}")
        if r2.status_code == 200:
            print(f"  Response preview: {r2.text[:300]}")
    except Exception as e:
        print(f"  GRIN observations error: {e}")

    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# BUILD CROSS-SOURCE COMPARISON SNIPPET
# ─────────────────────────────────────────────────────────────────────────────

def build_cross_source_snippet(vivc_df: pd.DataFrame, lacombe_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each reference variety present in ≥1 source, build a side-by-side
    comparison row showing allele calls at each of the 9 GENRES081 loci.
    """
    print("\n── CROSS-SOURCE COMPARISON ─────────────────────────────────────────")
    records = []

    def _get_profile(df: pd.DataFrame, variety: str) -> dict:
        if df.empty or "variety" not in df.columns:
            return {}
        match = df[df["variety"].str.upper() == variety.upper()]
        if match.empty:
            return {}
        row = match.iloc[0]
        profile = {}
        for locus in NINE_LOCI:
            if locus in row.index and pd.notna(row[locus]) and row[locus]:
                profile[locus] = str(row[locus])
            elif f"{locus}_A1" in row.index:
                a1 = str(row.get(f"{locus}_A1", "")).strip()
                a2 = str(row.get(f"{locus}_A2", "")).strip()
                if a1 and a2:
                    profile[locus] = f"{a1}/{a2}"
        return profile

    for var in REFERENCE_VARIETIES:
        vivc_profile  = _get_profile(vivc_df, var)
        lac_profile   = _get_profile(lacombe_df, var)

        if not vivc_profile and not lac_profile:
            continue

        for locus in NINE_LOCI:
            v_call = vivc_profile.get(locus, "—")
            l_call = lac_profile.get(locus, "—")
            match_flag = ""
            if v_call != "—" and l_call != "—":
                # Normalize allele order for comparison
                v_sorted = "/".join(sorted(v_call.split("/")))
                l_sorted = "/".join(sorted(l_call.split("/")))
                match_flag = "✓" if v_sorted == l_sorted else "⚠"

            records.append({
                "variety":        var,
                "locus":          locus,
                "VIVC":           v_call,
                "Lacombe_2013":   l_call,
                "match":          match_flag,
            })

    df_out = pd.DataFrame(records)
    print(f"  {len(df_out)} locus×variety comparison rows")
    if not df_out.empty:
        matches = df_out[df_out["match"] == "✓"]
        mismatches = df_out[df_out["match"] == "⚠"]
        print(f"  Concordant calls:   {len(matches)}")
        print(f"  Discordant calls:   {len(mismatches)}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Source 1: VIVC
    vivc_df = scrape_vivc_ssr(REFERENCE_VARIETIES)
    if not vivc_df.empty:
        vivc_df.to_csv(OUT / "ssr_vivc.csv", index=False)
        print(f"\n  Saved ssr_vivc.csv ({len(vivc_df)} rows)")
        print(vivc_df[["variety"] + [l for l in NINE_LOCI if l in vivc_df.columns]].to_string(index=False))
    else:
        print("  No VIVC SSR data retrieved.")

    # Source 2: Lacombe 2013
    lacombe_df = scrape_lacombe_pdfs()
    if not lacombe_df.empty:
        lacombe_df.to_csv(OUT / "ssr_lacombe2013.csv", index=False)
        print(f"\n  Saved ssr_lacombe2013.csv ({len(lacombe_df)} rows)")
        # Show sample for reference varieties
        ref_hits = lacombe_df[lacombe_df["variety"].isin([v.upper() for v in REFERENCE_VARIETIES])]
        print(f"  Reference variety hits in Lacombe: {len(ref_hits)}")
        if not ref_hits.empty:
            print(ref_hits.head(10).to_string(index=False))
    else:
        print("  No Lacombe SSR data retrieved.")

    # Source 3: GRIN SSR (exploratory)
    grin_ssr_df = scrape_grin_ssr()

    # Cross-source comparison
    snippet_df = build_cross_source_snippet(vivc_df, lacombe_df)
    if not snippet_df.empty:
        snippet_df.to_csv(OUT / "ssr_crosssource_snippet.csv", index=False)
        print(f"\n  Saved ssr_crosssource_snippet.csv")
        print("\n" + "="*70)
        print("CROSS-SOURCE GENOTYPE SNIPPET (first 5 varieties × all 9 loci)")
        print("="*70)
        first5 = snippet_df[snippet_df["variety"].isin(REFERENCE_VARIETIES[:5])]
        print(first5.to_string(index=False))
