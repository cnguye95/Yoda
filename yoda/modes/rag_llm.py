"""Mode 1 (RAG-LLM) for Yoda.

run_rag_llm() is the full RAG pipeline: it fetches and caches the most recent
10-Q/10-K for a ticker, chunks and embeds the filing, runs five fixed retrieval
queries against the vector store, gathers analyst consensus and recent news, then
asks gpt-4o to produce a fully structured EarningsReport in a single call.

Unlike the prompt-only baseline, the LLM receives targeted filing excerpts
tagged with chunk IDs so every source_citation can name the exact chunk.

Verification gate (Phase 5): python -m yoda.modes.rag_llm [TICKER]
"""

import json
import pathlib
import time
from datetime import datetime, timezone

from openai import OpenAI

from yoda import config
from yoda.ingest.edgar import fetch_latest_filing
from yoda.ingest.chunker import chunk_filing
from yoda.modes.baseline import _validate_citations
from yoda.retrieval.embeddings import embed_texts
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import EarningsReport
from yoda.tools.news import search_news


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# gpt-4o is locked in by PLAN.md for all final generation calls.
GPT4O_MODEL = "gpt-4o"

# gpt-4o pricing (USD per 1K tokens) as of 2026-05-10 — used for cost logging.
GPT4O_INPUT_COST_PER_1K  = 0.0025
GPT4O_OUTPUT_COST_PER_1K = 0.01

# text-embedding-3-small pricing (USD per 1K tokens).
EMBEDDING_COST_PER_1K = 0.00002

# Fixed retrieval queries from PLAN.md — do not parameterize or reorder.
RETRIEVAL_QUERIES = [
    "revenue segment breakdown",
    "forward guidance language",
    "key risk factors",
    "capital expenditure and margins",
    "notable changes from prior filing",
]
TOP_K = 5

# System prompt for the RAG generation call. Same "cite or skip" contract as
# the baseline, but citations must name the section heading label provided
# in the context (e.g. "MD&A — Revenue Recognition").
_SYSTEM_PROMPT = """You are a financial analyst assistant that produces structured
pre-earnings research reports in JSON format.

Rules you must follow without exception:
1. Every entry in key_metrics, revenue_segments, key_risks, and the
   forward_guidance block MUST have a non-empty source_citation field.
   For facts from the filing, cite the exact label provided in the
   context (e.g. "MD&A — Revenue Recognition" or "Risk Factors — Cybersecurity").
   For facts from news items, cite the article URL exactly as given.
2. If a fact is not directly supported by the retrieved chunks, the consensus
   block, or the news items provided, do NOT include it in the main fields.
   Instead, add a plain-English description of what is missing to data_gaps.
3. Never fabricate financial figures. If a number is not in the provided
   sources, it goes in data_gaps.
4. For recent_news, populate from the news items provided. Use the url field
   exactly as given; do not invent URLs.
5. Set report_generated_at to the ISO-8601 UTC timestamp provided in the
   user message.
6. Populate bull_case, bear_case, and what_to_watch from evidence in the
   retrieved chunks and news — not from general knowledge about the company."""


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused across calls)
# ---------------------------------------------------------------------------

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def _chunk_heading(chunk) -> str:
    # Extract the first short non-empty line from the chunk text to use as a
    # human-readable heading in citations (e.g. "Revenue Recognition").
    # Falls back to the section label if no line is short enough.
    for line in chunk.text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) <= 80:
            return stripped
    return chunk.section


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rag_llm(ticker: str) -> EarningsReport:
    """Produce an EarningsReport using the full RAG pipeline.

    Fetches (or reads from cache) the most recent 10-Q/10-K for ticker,
    embeds it into ChromaDB, retrieves the most relevant chunks for five
    fixed queries, gathers analyst consensus and recent news, then produces
    a structured report in one gpt-4o call.

    Raises ValueError if the model returns any empty source_citation fields.
    Raises RuntimeError if any external call (EDGAR, OpenAI, Finnhub, Tavily) fails.
    """
    ticker = ticker.upper().strip()
    wall_start = time.perf_counter()

    # Running tally of embedding tokens (approximated) across all embed_texts
    # calls in this pipeline. Used for cost logging at the end.
    embedding_tokens = 0

    # ------------------------------------------------------------------
    # Step 1: Fetch the filing (cached after first run)
    # ------------------------------------------------------------------
    print(f"[rag_llm] Fetching filing for {ticker}...")
    filing = fetch_latest_filing(ticker)
    print(f"[rag_llm] Filing: {filing['filing_type']} filed {filing['filing_date']}")

    # ------------------------------------------------------------------
    # Step 2: Chunk, embed, and upsert into ChromaDB
    # ------------------------------------------------------------------
    print(f"[rag_llm] Chunking filing...")
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    print(f"[rag_llm] Chunked into {len(chunks)} chunks; embedding...")

    chunk_texts = [c.text for c in chunks]

    t0 = time.perf_counter()
    embeddings = embed_texts(chunk_texts)
    embed_elapsed = time.perf_counter() - t0

    # Approximate token count: 4 chars per token is a common rule of thumb.
    embedding_tokens += sum(len(t) for t in chunk_texts) // 4
    print(f"[rag_llm] Embedded {len(chunks)} chunks in {embed_elapsed:.2f}s; upserting...")

    store = ChromaStore()
    store.upsert(filing["accession_number"], chunks, embeddings)

    # ------------------------------------------------------------------
    # Step 3: Run the five fixed retrieval queries; deduplicate results
    # ------------------------------------------------------------------

    # collected_chunks maps chunk_index -> Chunk to eliminate duplicates while
    # preserving the insertion order of first discovery (most relevant first).
    collected_chunks: dict[int, object] = {}

    for query in RETRIEVAL_QUERIES:
        t0 = time.perf_counter()
        results = store.query(filing["accession_number"], query, k=TOP_K)
        elapsed = time.perf_counter() - t0

        # Each store.query() call embeds the query string — track those tokens.
        embedding_tokens += len(query) // 4

        # Log: query, latency, top result summary (first 80 chars of top chunk).
        if results:
            top = results[0]
            snippet = top.text.replace("\n", " ")[:80]
            print(f'[rag_llm] Query "{query}" (k={TOP_K}) -> {elapsed:.2f}s; '
                  f'top: [{top.section} chunk {top.chunk_index}] "{snippet}"')
        else:
            print(f'[rag_llm] Query "{query}" (k={TOP_K}) -> {elapsed:.2f}s; no results')

        # Insert into the dict only if not already present (first occurrence wins).
        for chunk in results:
            if chunk.chunk_index not in collected_chunks:
                collected_chunks[chunk.chunk_index] = chunk

    unique_chunks = list(collected_chunks.values())
    print(f"[rag_llm] Unique chunks after deduplication: {len(unique_chunks)}")

    # ------------------------------------------------------------------
    # Step 4: Fetch recent news
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    news_results = search_news(f"{ticker} earnings", max_results=5)
    print(f"[rag_llm] search_news -> {time.perf_counter() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Step 5: Build the user prompt combining chunks and tool outputs
    # ------------------------------------------------------------------
    now_utc = datetime.now(timezone.utc).isoformat()

    # Render each retrieved chunk with a descriptive label so the LLM can cite
    # it by section and heading rather than an opaque chunk number.
    chunks_section = "\n\n".join(
        f"[{c.section} — {_chunk_heading(c)}]\n{c.text}"
        for c in unique_chunks
    )

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Report timestamp (use for report_generated_at): {now_utc}\n\n"
        f"--- RETRIEVED FILING CHUNKS ---\n{chunks_section}\n\n"
        f"--- RECENT NEWS (JSON) ---\n{json.dumps(news_results, default=str)}\n\n"
        "Produce the structured EarningsReport now."
    )

    # ------------------------------------------------------------------
    # Step 6: Single gpt-4o structured output call
    # ------------------------------------------------------------------
    print(f"[rag_llm] Calling {GPT4O_MODEL}...")
    t0 = time.perf_counter()

    completion = _client.beta.chat.completions.parse(
        model=GPT4O_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=EarningsReport,
        temperature=0,
    )

    llm_elapsed = time.perf_counter() - t0
    llm_in  = completion.usage.prompt_tokens
    llm_out = completion.usage.completion_tokens
    print(f"[rag_llm] LLM call -> {llm_elapsed:.2f}s ({llm_in} in / {llm_out} out tokens)")

    report = completion.choices[0].message.parsed

    # ------------------------------------------------------------------
    # Step 7: Validate that all source_citation fields are non-empty
    # ------------------------------------------------------------------
    _validate_citations(report)

    # ------------------------------------------------------------------
    # Step 8: Log cost and wall time totals
    # ------------------------------------------------------------------
    wall_elapsed = time.perf_counter() - wall_start

    embed_cost = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K
    llm_cost   = (llm_in  / 1000) * GPT4O_INPUT_COST_PER_1K \
               + (llm_out / 1000) * GPT4O_OUTPUT_COST_PER_1K
    total_cost = embed_cost + llm_cost

    print(f"[rag_llm] === TOTALS ===")
    print(f"[rag_llm] Wall time:        {wall_elapsed:.2f}s")
    print(f"[rag_llm] Embedding tokens: ~{embedding_tokens} (~${embed_cost:.5f})")
    print(f"[rag_llm] LLM tokens:       {llm_in} in + {llm_out} out (${llm_cost:.5f})")
    print(f"[rag_llm] Total cost:       ${total_cost:.5f}")

    return report


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.rag_llm [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NFLX").upper()

    print(f"Running RAG-LLM mode for {ticker}...")
    report = run_rag_llm(ticker)

    # Save to data/eval/ for Phase 9 evaluation comparison.
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rag_llm_{ticker}.json"
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
