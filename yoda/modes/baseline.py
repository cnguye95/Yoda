"""Prompt-only baseline mode for Yoda.

run_baseline() takes a ticker and a manually supplied filing excerpt, makes
one consensus call and one news search, then asks gpt-4o to produce a fully
structured EarningsReport in a single call using OpenAI structured outputs.

This baseline intentionally does no RAG or agent loop — it exists as the
lower bound to compare Modes 1 and 2 against in Phase 9 evaluation.
"""

import json
import pathlib
from datetime import datetime, timezone

from openai import OpenAI

from yoda import config
from yoda.schema import EarningsReport
from yoda.tools.news import search_news


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# gpt-4o is locked in by PLAN.md for all final generation calls.
GPT4O_MODEL = "gpt-4o"

# System prompt that enforces the 'cite or skip' rule.
_SYSTEM_PROMPT = """You are a financial analyst assistant that produces structured
pre-earnings research reports in JSON format.

Rules you must follow without exception:
1. Every entry in key_metrics, revenue_segments, key_risks, and the
   forward_guidance block MUST have a non-empty source_citation field. The
   citation must name the specific section or sentence in the filing excerpt
   that supports the fact (e.g. "MD&A, paragraph 3" or "Item 1A, risk #2").
2. If a fact is not directly supported by the filing excerpt, the consensus
   block, or the news items provided, do NOT include it in the main fields.
   Instead, add a plain-English description of what is missing to data_gaps.
3. Never fabricate financial figures. If a number is not in the provided
   sources, it goes in data_gaps.
4. For recent_news, populate from the news items provided. Use the url field
   exactly as given; do not invent URLs.
5. Set report_generated_at to the ISO-8601 UTC timestamp provided in the
   user message.
6. Populate bull_case, bear_case, and what_to_watch from evidence in the
   excerpt and news — not from general knowledge about the company."""


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused across calls)
# ---------------------------------------------------------------------------

# Matches the pattern in yoda/retrieval/embeddings.py: single instantiation
# at module load avoids re-reading the API key on every call.
_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------

def _validate_citations(report: EarningsReport) -> None:
    # Walk every source_citation field in the report and raise ValueError if
    # any are empty or whitespace. The system prompt instructs the LLM to use
    # data_gaps instead of leaving citations blank, so a hit here is a model
    # compliance failure — we fail loud rather than accept a bad report.
    errors = []

    for i, m in enumerate(report.key_metrics):
        if not m.source_citation.strip():
            errors.append(f"key_metrics[{i}].source_citation")

    for i, s in enumerate(report.revenue_segments):
        if not s.source_citation.strip():
            errors.append(f"revenue_segments[{i}].source_citation")

    for i, r in enumerate(report.key_risks):
        if not r.source_citation.strip():
            errors.append(f"key_risks[{i}].source_citation")

    if not report.forward_guidance.source_citation.strip():
        errors.append("forward_guidance.source_citation")

    if errors:
        raise ValueError(
            f"Empty source_citation in: {', '.join(errors)}. "
            "The LLM must put uncitable facts in data_gaps, not leave citations blank."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(ticker: str, manual_excerpt: str) -> EarningsReport:
    """Produce an EarningsReport from a filing excerpt using a single LLM call.

    No RAG, no agent loop — this is the prompt-only baseline. The caller
    supplies a manually extracted (or auto-sliced) excerpt from the most
    recent 10-Q or 10-K. The function adds one consensus call and one news
    search, then asks gpt-4o to fill the full schema in one shot.

    Raises ValueError if the model returns any empty source_citation fields.
    Raises RuntimeError if any external tool call fails.
    """
    ticker = ticker.upper().strip()
    now_utc = datetime.now(timezone.utc).isoformat()

    # Fetch recent news relevant to the ticker's upcoming earnings.
    news_results = search_news(f"{ticker} earnings", max_results=5)

    # Build the user message combining all data sources the LLM may cite.
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Report timestamp (use for report_generated_at): {now_utc}\n\n"
        f"--- FILING EXCERPT ---\n{manual_excerpt}\n\n"
        f"--- RECENT NEWS (JSON) ---\n{json.dumps(news_results, default=str)}\n\n"
        "Produce the structured EarningsReport now."
    )

    # Call gpt-4o with structured outputs. response_format=EarningsReport tells
    # the OpenAI client to generate a strict JSON schema and return a parsed
    # Pydantic instance directly — no manual JSON parsing needed.
    completion = _client.beta.chat.completions.parse(
        model=GPT4O_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=EarningsReport,
        temperature=0,
    )

    report = completion.choices[0].message.parsed

    # Validate that every citable field has a non-empty citation.
    _validate_citations(report)

    return report


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.baseline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from yoda.ingest.edgar import fetch_latest_filing
    from yoda.ingest.chunker import chunk_filing

    ticker = "NFLX"

    # Load the cached NFLX filing from Phase 1 (avoids a network round-trip).
    print(f"Loading cached filing for {ticker}...")
    filing = fetch_latest_filing(ticker)

    # Build a high-quality ~5000-char excerpt using Phase 2's section-aware
    # chunker. We prefer MD&A (substantive management commentary) and fall back
    # to Financial Statements if MD&A chunks are too short.
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    preferred = ["MD&A", "Financial Statements"]
    excerpt_parts: list[str] = []
    total_chars = 0

    for section_name in preferred:
        for chunk in chunks:
            if chunk.section == section_name and total_chars < 5000:
                excerpt_parts.append(chunk.text)
                total_chars += len(chunk.text)
        if total_chars >= 5000:
            break

    # Trim to exactly 5000 chars so the context window stays predictable.
    excerpt = " ".join(excerpt_parts)[:5000]
    print(f"Excerpt: {len(excerpt)} chars (sections: {preferred})")

    # Run the baseline.
    print(f"Running baseline for {ticker} via {GPT4O_MODEL}...")
    report = run_baseline(ticker, excerpt)

    # Save to data/eval/ for Phase 9 evaluation comparison.
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"baseline_{ticker}.json"
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    # Print a summary so the output is scannable without opening the JSON.
    print(f"\nSaved to {out_file}")
    print(f"Company:          {report.company_name}")
    print(f"Filing:           {report.filing_type} — {report.filing_date}")
    print(f"Key metrics:      {len(report.key_metrics)}")
    print(f"Revenue segments: {len(report.revenue_segments)}")
    print(f"Key risks:        {len(report.key_risks)}")
    print(f"Recent news:      {len(report.recent_news)}")
    print(f"Data gaps:        {len(report.data_gaps)}")
    if report.data_gaps:
        for gap in report.data_gaps:
            print(f"  - {gap}")
