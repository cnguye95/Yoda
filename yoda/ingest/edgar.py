"""Fetches the most recent SEC 10-Q (or 10-K fallback) for a given ticker.

The one public function, fetch_latest_filing(), returns a dict with the
filing metadata and the cleaned text ready for Phase 2 chunking. Raw HTML
and filing metadata are cached to data/filings/{ticker}/ so that repeated
calls during development skip all network round-trips.

SEC rules:
  - Every request must include a User-Agent header identifying the caller.
    The value comes from SEC_USER_AGENT in .env (format: "Name email").
  - Stay under 10 requests per second — we sleep 0.15s between calls.
"""

import json
import time
import pathlib
from datetime import date, timedelta

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC filings are XHTML/XML but lxml's HTML parser handles them correctly.
# Suppress the warning so it doesn't surface in the Streamlit log.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from yoda import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Root directory for cached HTML filings. Created on first use.
_CACHE_DIR = pathlib.Path("data/filings")

# SEC endpoint that maps every ticker to its CIK (company identifier) number.
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC endpoint that returns a company's full filing history given its CIK.
# The CIK must be zero-padded to 10 digits.
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Filing index HTML — lists every document in a specific accession.
# The primary document (the 10-Q or 10-K itself) is always Seq 1.
_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{acc}-index.htm"

# Base URL for downloading actual filing documents from the EDGAR archive.
_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodashes}/{doc}"

# If the most recent 10-Q is older than this many days, fall back to 10-K.
_TEN_Q_FRESHNESS_DAYS = 180

# ---------------------------------------------------------------------------
# Module-level cache for the ticker→CIK mapping so we only fetch it once
# per Python process (the file is ~1 MB of JSON).
# ---------------------------------------------------------------------------
_ticker_cik_cache: dict | None = None


def _headers() -> dict:
    # Build the HTTP headers dict required by SEC EDGAR.
    # Without a proper User-Agent the SEC returns HTTP 403.
    return {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _get(url: str) -> requests.Response:
    # Wrapper around requests.get that adds our required headers and enforces
    # a short sleep so we don't exceed SEC's 10 req/sec rate limit.
    time.sleep(0.15)
    resp = requests.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()  # Blow up loudly on 4xx/5xx — no silent failures.
    return resp


def _load_ticker_cik_map() -> dict:
    # Return a dict mapping uppercase ticker strings to integer CIK values.
    # The result is kept in a module-level variable so subsequent calls within
    # the same process skip the network round-trip.
    global _ticker_cik_cache
    if _ticker_cik_cache is not None:
        return _ticker_cik_cache

    resp = _get(_TICKERS_URL)
    # The JSON structure is { "0": {cik_str, ticker, title}, "1": {...}, ... }
    raw = resp.json()
    _ticker_cik_cache = {
        entry["ticker"].upper(): int(entry["cik_str"])
        for entry in raw.values()
    }
    return _ticker_cik_cache


def _get_filings_metadata(cik: int) -> list[dict]:
    # Fetch the filing history for a company and return it as a list of dicts.
    # Each dict has: form, filing_date (str YYYY-MM-DD), accession_number.
    # The submissions JSON nests filings under data["filings"]["recent"].
    resp = _get(_SUBMISSIONS_URL.format(cik=cik))
    recent = resp.json()["filings"]["recent"]

    forms   = recent.get("form", [])
    dates   = recent.get("filingDate", [])
    accnums = recent.get("accessionNumber", [])

    filings = []
    for form, date_str, acc in zip(forms, dates, accnums):
        filings.append({
            "form":             form,
            "filing_date":      date_str,
            "accession_number": acc,   # e.g. "0000320193-25-000123"
        })
    return filings


def _pick_latest_filing(filings: list[dict], today: date) -> dict:
    # Choose the most recent 10-Q filed within the freshness window.
    # If none qualifies, fall back to the most recent 10-K.
    # Raise RuntimeError loudly if neither exists — don't silently return None.
    cutoff = today - timedelta(days=_TEN_Q_FRESHNESS_DAYS)

    best_10q = None
    best_10k = None

    for f in filings:
        filed = date.fromisoformat(f["filing_date"])
        if f["form"] == "10-Q" and filed >= cutoff:
            if best_10q is None or filed > date.fromisoformat(best_10q["filing_date"]):
                best_10q = f
        if f["form"] == "10-K":
            if best_10k is None or filed > date.fromisoformat(best_10k["filing_date"]):
                best_10k = f

    if best_10q is not None:
        return best_10q
    if best_10k is not None:
        return best_10k
    raise RuntimeError(
        "No 10-Q (within 6 months) or 10-K found for this ticker. "
        "The company may not be an SEC filer, or EDGAR has no recent filings."
    )


def _get_primary_document(cik: int, accession_number: str) -> str:
    # Fetch the filing index page and return the filename of the primary
    # document (the actual 10-Q or 10-K HTML, always listed as Seq 1 in the
    # index table). This is one extra HTTP call but avoids guessing file names.
    acc_nodashes = accession_number.replace("-", "")
    idx_url = _INDEX_URL.format(cik=cik, acc_nodashes=acc_nodashes, acc=accession_number)
    resp = _get(idx_url)

    soup = BeautifulSoup(resp.text, "lxml")
    # The index table has rows: Seq | Description | Document | Type | Size.
    # Seq 1 is always the primary document (the filing itself, not exhibits).
    # The Document cell has an <a> tag; use its href to get the clean filename
    # because iXBRL labels like "iXBRL" can be appended to the visible text.
    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 3 and cells[0].get_text(strip=True) == "1":
            link = cells[2].find("a")
            if link and link.get("href"):
                # href may be an iXBRL viewer path like
                # "/ix?doc=/Archives/edgar/data/.../aapl-20260328.htm"
                # rsplit("/", 1)[-1] extracts just the filename.
                doc_name = link["href"].rsplit("/", 1)[-1]
                if doc_name.lower().endswith((".htm", ".html")):
                    return doc_name

    raise RuntimeError(
        f"Could not find primary .htm document for accession {accession_number}. "
        "The filing index page may have an unexpected structure."
    )


def _download_filing_html(cik: int, accession_number: str, primary_doc: str) -> tuple[str, str]:
    # Download the primary HTML document for a filing from the EDGAR archive.
    # Returns (raw_html_text, url).
    acc_nodashes = accession_number.replace("-", "")
    url = _ARCHIVES_URL.format(cik=cik, acc_nodashes=acc_nodashes, doc=primary_doc)
    resp = _get(url)
    return resp.text, url


def _clean_html(raw_html: str) -> str:
    # Strip away everything that isn't readable text, while keeping heading
    # tags (h1-h6) as text so the Phase 2 chunker can use them as section
    # boundary markers (e.g. "Item 1A. Risk Factors").
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script, style, nav, header, and footer blocks entirely.
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Extract text with newline separators so paragraphs stay separated.
    text = soup.get_text(separator="\n")

    # Collapse runs of blank lines to a single blank line so the output is
    # readable and chunk boundaries are obvious.
    lines = text.splitlines()
    cleaned_lines = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        if stripped:
            cleaned_lines.append(stripped)
            prev_blank = False
        elif not prev_blank:
            cleaned_lines.append("")
            prev_blank = True

    return "\n".join(cleaned_lines)


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------

def _read_disk_cache(ticker: str) -> dict | None:
    # If both the metadata JSON and the HTML file exist for this ticker,
    # return a dict with all stored fields plus raw_html. Otherwise None.
    # This allows a second fetch_latest_filing() call to skip all HTTP calls.
    meta_path = _CACHE_DIR / ticker / "latest.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    html_path = _CACHE_DIR / ticker / f"{meta['accession_number']}.html"
    if not html_path.exists():
        return None

    meta["raw_html"] = html_path.read_text(encoding="utf-8")
    return meta


def _write_disk_cache(ticker: str, meta: dict, raw_html: str) -> None:
    # Write the raw HTML and a small JSON metadata file to disk so future
    # calls can skip network round-trips entirely.
    cache_dir = _CACHE_DIR / ticker
    cache_dir.mkdir(parents=True, exist_ok=True)

    acc = meta["accession_number"]
    (cache_dir / f"{acc}.html").write_text(raw_html, encoding="utf-8")

    # Only persist the lightweight fields, not raw_html itself (stored separately).
    saveable = {k: v for k, v in meta.items() if k != "raw_html"}
    (cache_dir / "latest.json").write_text(json.dumps(saveable), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_latest_filing(ticker: str, force_refresh: bool = False) -> dict:
    """Fetch the most recent 10-Q (or 10-K fallback) for the given ticker.

    Returns a dict with keys:
        ticker, cik, filing_type, filing_date, accession_number, url,
        raw_html, clean_text

    Both the raw HTML and filing metadata are cached to disk so a second call
    for the same ticker skips all HTTP calls and returns from disk.

    Pass force_refresh=True to bypass the disk cache and re-fetch from SEC.
    Use this when a new filing may have been published since the last cache
    write. The new filing overwrites the cached one on disk.
    """
    ticker = ticker.upper().strip()

    # Check disk cache first — if we have everything on disk, skip all HTTP.
    # force_refresh=True bypasses the cache so callers can pick up new filings.
    if not force_refresh:
        cached = _read_disk_cache(ticker)
        if cached:
            cached["clean_text"] = _clean_html(cached["raw_html"])
            return cached

    # Step 1: resolve ticker to CIK using the SEC's public mapping file.
    cik_map = _load_ticker_cik_map()
    if ticker not in cik_map:
        raise RuntimeError(
            f"Ticker '{ticker}' not found in SEC EDGAR. "
            "Check the symbol or try the company's CIK directly."
        )
    cik = cik_map[ticker]

    # Step 2: get the company's filing history and pick the right filing.
    filings = _get_filings_metadata(cik)
    chosen = _pick_latest_filing(filings, date.today())

    accession_number = chosen["accession_number"]
    filing_type      = chosen["form"]
    filing_date      = chosen["filing_date"]

    # Step 3: find the primary document filename from the filing index page.
    primary_doc = _get_primary_document(cik, accession_number)

    # Step 4: download the primary document from the EDGAR archive.
    raw_html, url = _download_filing_html(cik, accession_number, primary_doc)

    # Step 5: persist to disk so the next call is fully cache-served.
    meta = {
        "ticker":           ticker,
        "cik":              cik,
        "filing_type":      filing_type,
        "filing_date":      filing_date,
        "accession_number": accession_number,
        "url":              url,
    }
    _write_disk_cache(ticker, meta, raw_html)

    # Step 6: clean the HTML to plain text for the Phase 2 chunker.
    clean_text = _clean_html(raw_html)

    return {**meta, "raw_html": raw_html, "clean_text": clean_text}


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.ingest.edgar [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(f"Fetching latest filing for {ticker_arg}...")
    result = fetch_latest_filing(ticker_arg)

    # Print metadata
    print(f"\nTicker:           {result['ticker']}")
    print(f"CIK:              {result['cik']}")
    print(f"Filing type:      {result['filing_type']}")
    print(f"Filing date:      {result['filing_date']}")
    print(f"Accession number: {result['accession_number']}")
    print(f"URL:              {result['url']}")
    print(f"Clean text length:{len(result['clean_text'])} chars")

    # Print first 500 chars of clean text
    print(f"\n--- First 500 chars of clean text ---")
    print(result["clean_text"][:500])
