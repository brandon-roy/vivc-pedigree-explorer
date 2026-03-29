"""
build_lacombe_marker_support.py
================================
Extract 828 confirmed parentage rows from Lacombe et al. 2013 (TAG)
Online Resource 2 (MOESM2_ESM.pdf) and emit two marker-support records
per row (one per parent role) in the standard 12-column schema.

Usage:
    python scripts/build_lacombe_marker_support.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
PDF_PATH   = (
    Path("/Users/brandonroy/Desktop/Informatique/Projects/VIVC App")
    / "Supplemental Literature Data"
    / "122_2012_1988_MOESM2_ESM.pdf"
)
ALIAS_PATH = BASE_DIR / "data" / "marker_name_aliases.csv"
OUT_PATH   = BASE_DIR / "data" / "lacombe_2013_marker_support.csv"

STUDY_REF  = "Lacombe et al. 2013"
DOI        = "10.1007/s00122-012-1988-2"
CONF_YEAR  = 2013
N_MARKERS  = 20


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_alias_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    if "alias" not in df.columns or "canonical_vivc_name" not in df.columns:
        return {}
    return {
        str(a).strip().upper(): str(c).strip().upper()
        for a, c in zip(df["alias"], df["canonical_vivc_name"])
    }


def norm(name: str, alias_map: dict[str, str]) -> str:
    """Strip, uppercase, apply alias map."""
    n = name.strip().upper()
    return alias_map.get(n, n)


def clean_variety_name(raw: str) -> str:
    """
    Remove trailing VIVC-id suffixes like '(#1234)', optional 'F' gender marker,
    and 'nd' (non-determined) parentage markers.
    """
    # Remove VIVC ID suffix: (#digits) with optional trailing junk
    s = re.sub(r"\s*\(#\d+\)\s*[a-z]?\s*$", "", raw, flags=re.IGNORECASE).strip()
    # Remove standalone 'F' or 'nd' at the end (female / non-determined flags)
    s = re.sub(r"\s+[Ff]$", "", s).strip()
    s = re.sub(r"\s+nd$", "", s, flags=re.IGNORECASE).strip()
    return s


# ── PDF parsing ────────────────────────────────────────────────────────────────

# We parse line-by-line. A "primary" data line looks like:
#   <Progeny text> <Parent1 text> <Parent2 text> <N/M> <LOD> [Comments...]
# where N/M is digits/digits and LOD is a decimal number.
# Continuation lines (starting with text that continues a multi-line cell)
# are detected by the absence of the N/M LOD pattern.

# Pattern to detect the "N/M LOD" pivot in a line
_PIVOT_RE = re.compile(
    r"^(.+?)\s+(\d{1,2}/\d{1,2})\s+([\d.]+)\s*(.*)$"
)

# Minimum token count to consider a leading portion parseable as 3 names
_MIN_NAMES_CHARS = 5


def split_three_names(text: str) -> tuple[str, str, str] | None:
    """
    Given text that is: '<Progeny> <Parent1> <Parent2>'
    try to split into three variety names.

    Heuristic: names contain uppercase letters, digits, spaces, hyphens,
    accented chars, '=', '.' and are separated by at least one space from
    the next name.  We look for VIVC id anchors (#digits) to split.
    """
    # Ideal case: all three names have (#id) tags
    ids = list(re.finditer(r"\(#\d+\)", text))
    if len(ids) >= 3:
        # Use positions of id tags to delimit names
        # name1 ends before first id end, name2 before second, name3 before third
        name1_raw = text[: ids[0].end()].strip()
        name2_raw = text[ids[0].end() : ids[1].end()].strip()
        name3_raw = text[ids[1].end() : ids[2].end()].strip()
        return (clean_variety_name(name1_raw),
                clean_variety_name(name2_raw),
                clean_variety_name(name3_raw))

    if len(ids) == 2:
        # Two ids: progeny has id, one parent has id
        # name1: up to ids[0].end, name2: ids[0].end to ids[1].end, name3: rest
        name1_raw = text[: ids[0].end()].strip()
        name2_raw = text[ids[0].end() : ids[1].end()].strip()
        name3_raw = text[ids[1].end():].strip()
        if name1_raw and name3_raw:
            return (clean_variety_name(name1_raw),
                    clean_variety_name(name2_raw),
                    clean_variety_name(name3_raw))

    if len(ids) == 1:
        # Only one id — progeny probably has id
        name1_raw = text[: ids[0].end()].strip()
        rest = text[ids[0].end():].strip()
        # Try to split rest by uppercase tokens
        # Simple heuristic: split at space sequences of 2+ spaces or caps boundary
        parts = re.split(r"\s{2,}", rest)
        if len(parts) >= 2:
            return (clean_variety_name(name1_raw),
                    clean_variety_name(parts[0]),
                    clean_variety_name(" ".join(parts[1:])))

    return None


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract parentage rows from the PDF using pdfplumber page-by-page text.

    Strategy
    --------
    The PDF text is extracted page-by-page. The critical structure is:

      Primary line:
        <Progeny text> [<Parent1 text>] [<Parent2 text>] N/M  LOD  [Comments...]

      Continuation lines (next physical line) can contain:
        - Remaining (#ID) suffixes for parent names split across lines
          e.g.  "(#766) (#908)"  or  "(#1544)"
        - Text continuation of the Comments column

    Detection heuristic:
      A continuation line that consists *only* of (#ID) tokens (possibly with
      small surrounding text like variety name fragments) is an ID-continuation
      and should be appended to the current record's raw_before for re-parsing.

      A continuation line that contains regular words (reference text, etc.) is
      a comment continuation and is appended to the comments field.

    Returns list of raw dicts with keys:
      progeny, parent1, parent2, n_contrib, n_mismatch, lod_score, comments
    """
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber not installed. Run: pip install pdfplumber", file=sys.stderr)
        sys.exit(1)

    # ID-continuation: short line that contains at least one (#id) token.
    # This catches both pure-id lines "(#766) (#908)" and lines that have
    # some trailing comment text after the ids, e.g. "(#634) Vargas 2009".
    # We treat any line < 100 chars containing (#id) as an id-continuation
    # *unless* it already matches the main pivot pattern (would be a new row).
    def _is_id_cont(line: str) -> bool:
        return bool(re.search(r"\(#\d+\)", line)) and len(line) < 100

    header_skip_re = re.compile(
        r"^(?:Online Resource|List of|Vitis vinifera|INRA Domaine"
        r"|Progeny\s|nSSR|Contrib|Mismatch|LOD score"
        r"|\d+\s*/\s*\d+\s*$"  # page numbers like "1 / 37"
        r"|Online Resource 2, Lacombe)",
        re.IGNORECASE,
    )
    page_num_re = re.compile(r"^\d+\s*$")

    # Collect all body lines first (strip headers/footers)
    all_lines: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if page_num_re.match(line):
                    continue
                if header_skip_re.match(line):
                    continue
                all_lines.append(line)

    rows: list[dict] = []

    # Pass 1: merge continuation lines into their primary lines.
    # A primary line contains the N/M LOD pattern.
    # A continuation line immediately after a primary line that matches
    # _ID_CONT_RE is merged into the primary line's pre-pivot portion.
    # Otherwise it is merged into the comments.

    merged_lines: list[tuple[str, str]] = []  # (pre_pivot, post_pivot/comments)

    i = 0
    while i < len(all_lines):
        line = all_lines[i]
        m = _PIVOT_RE.match(line)
        if m:
            pre_pivot = m.group(1).strip()
            nssr_str  = m.group(2)
            lod_str   = m.group(3)
            comments  = m.group(4).strip()

            # Consume following continuation lines
            i += 1
            while i < len(all_lines):
                next_line = all_lines[i]
                # Does next line match as a new primary?
                if _PIVOT_RE.match(next_line):
                    break
                # Is it an ID-continuation?
                if _is_id_cont(next_line):
                    # Append to pre_pivot so name splitting can use the IDs
                    pre_pivot = (pre_pivot + " " + next_line).strip()
                    i += 1
                else:
                    # Comment continuation
                    comments = (comments + " " + next_line).strip()
                    i += 1

            # Parse N/M
            nssr_parts = nssr_str.split("/")
            try:
                n_contrib  = int(nssr_parts[0])
                n_mismatch = int(nssr_parts[1])
            except (IndexError, ValueError):
                n_contrib  = 20
                n_mismatch = 0

            try:
                lod = float(lod_str)
            except ValueError:
                lod = float("nan")

            names = split_three_names(pre_pivot)
            if names is None:
                # Last-ditch: just try to split on wide whitespace
                parts = re.split(r"\s{3,}", pre_pivot)
                if len(parts) >= 3:
                    names = (
                        clean_variety_name(parts[0]),
                        clean_variety_name(parts[1]),
                        clean_variety_name(" ".join(parts[2:])),
                    )
                elif len(parts) == 2:
                    names = (
                        clean_variety_name(parts[0]),
                        clean_variety_name(parts[1]),
                        "",
                    )

            if names is None:
                continue

            progeny, p1, p2 = names
            rows.append({
                "progeny":    progeny,
                "parent1":    p1,
                "parent2":    p2,
                "n_contrib":  n_contrib,
                "n_mismatch": n_mismatch,
                "lod_score":  lod,
                "comments":   comments,
            })
        else:
            i += 1

    return rows


# ── Confidence assignment ──────────────────────────────────────────────────────

def confidence(n_mismatch: int) -> str:
    if n_mismatch == 0:
        return "confirmed"
    elif n_mismatch <= 2:
        return "probable"
    else:
        return "disputed"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    alias_map = load_alias_map(ALIAS_PATH)

    print(f"Parsing PDF: {PDF_PATH}")
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", file=sys.stderr)
        sys.exit(1)

    raw_rows = parse_pdf(PDF_PATH)
    print(f"  Raw rows parsed: {len(raw_rows)}")

    output_rows: list[dict] = []

    for r in raw_rows:
        progeny = norm(r["progeny"], alias_map)
        p1      = norm(r["parent1"], alias_map)
        p2      = norm(r["parent2"], alias_map)

        if not progeny or not p1:
            continue

        n_contrib  = r["n_contrib"]
        n_mismatch = r["n_mismatch"]
        lod        = r["lod_score"]
        comments   = r["comments"].strip() if r["comments"] else ""
        conf       = confidence(n_mismatch)
        nssr_note  = f"nSSR: {n_contrib}/{n_mismatch}"
        notes_str  = f"{comments} | {nssr_note}".strip(" |") if comments else nssr_note

        base = {
            "evidence_type":   "SSR",
            "marker_type":     "SSR",
            "n_markers":       N_MARKERS,
            "lod_score":       lod,
            "study_reference": STUDY_REF,
            "doi":             DOI,
            "confirmed_year":  CONF_YEAR,
            "confidence_level": conf,
            "notes":           notes_str,
        }

        # Parent 1 record
        output_rows.append({
            "child_variety":  progeny,
            "parent_variety": p1,
            "parent_role":    "Parent 1",
            **base,
        })

        # Parent 2 record (skip if same as parent 1 — self-crosses are valid)
        if p2:
            output_rows.append({
                "child_variety":  progeny,
                "parent_variety": p2,
                "parent_role":    "Parent 2",
                **base,
            })

    # Build DataFrame with standard column order
    COLS = [
        "child_variety", "parent_variety", "parent_role",
        "evidence_type", "marker_type", "n_markers",
        "lod_score", "study_reference", "doi",
        "confirmed_year", "confidence_level", "notes",
    ]
    out_df = pd.DataFrame(output_rows, columns=COLS)

    # Write output
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"  Output rows: {len(out_df)}  →  {OUT_PATH}")

    # Summary
    conf_counts = out_df["confidence_level"].value_counts()
    print("\nConfidence breakdown:")
    for lvl in ("confirmed", "probable", "disputed"):
        print(f"  {lvl}: {conf_counts.get(lvl, 0)}")

    # Sanity — expect roughly 828*2 rows (minus empty p2 in rare cases)
    raw_trios = len(raw_rows)
    print(f"\nTrios extracted from PDF: {raw_trios}")
    if raw_trios < 750:
        print("WARNING: expected ~828 trios — parsing may be incomplete.", file=sys.stderr)


if __name__ == "__main__":
    main()
