"""Mode 2 (Agent-Reasoning) for Yoda.

run_agent() is a ReAct-style loop that starts with the same initial retrieval
as Mode 1 (so it can never do worse), then iterates: gpt-4o-mini decides the
next action each step; the action is executed and logged in a ReasoningTrace.
When the agent chooses FINISH (or the iteration cap is hit), gpt-4o produces
the final structured EarningsReport from all accumulated context.

Key invariant: NO report content is generated during the loop. The LLM only
chooses actions and writes a one-sentence thought per step. This prevents
accumulated hallucination across iterations.

Verification gate (Phase 6): python -m yoda.modes.agent [TICKER]
"""

import json
import pathlib
import time
from datetime import datetime, timezone

from openai import OpenAI
from pydantic import BaseModel

from yoda import config
from yoda.ingest.edgar import fetch_latest_filing
from yoda.ingest.chunker import chunk_filing, Chunk
from yoda.modes.baseline import _validate_citations
from yoda.modes.rag_llm import RETRIEVAL_QUERIES, TOP_K
from yoda.retrieval.embeddings import embed_texts
from yoda.retrieval.vector_store import ChromaStore
from yoda.schema import EarningsReport
from yoda.tools.consensus import get_consensus
from yoda.tools.news import search_news


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# gpt-4o for final structured generation (PLAN.md locks this in).
GPT4O_MODEL = "gpt-4o"

# gpt-4o-mini for cheap iterative action selection (PLAN.md).
GPT4O_MINI_MODEL = "gpt-4o-mini"

# Pricing (USD per 1K tokens) as of 2026-05-10.
GPT4O_INPUT_COST_PER_1K       = 0.0025
GPT4O_OUTPUT_COST_PER_1K      = 0.01
GPT4O_MINI_INPUT_COST_PER_1K  = 0.00015
GPT4O_MINI_OUTPUT_COST_PER_1K = 0.0006
EMBEDDING_COST_PER_1K         = 0.00002

# The four valid action names the agent may choose. Any other value is treated
# as FINISH to prevent an infinite loop if the model hallucinates an action.
VALID_ACTIONS = {"RETRIEVE_FROM_FILING", "SEARCH_NEWS", "LOOKUP_RELATED_TICKER", "FINISH"}

# System prompt for the final gpt-4o generation call.
_GENERATION_SYSTEM_PROMPT = """You are a financial analyst assistant that produces structured
pre-earnings research reports in JSON format.

Rules you must follow without exception:
1. Every entry in key_metrics, revenue_segments, key_risks, and the
   forward_guidance block MUST have a non-empty source_citation field.
   For facts from the primary filing, cite the exact chunk ID as given
   (e.g. "MD&A chunk 12" or "Risk Factors chunk 7").
   For facts from a related company's filing, include the ticker prefix
   exactly as given (e.g. "MSFT Financial Statements chunk 5").
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

# System prompt for gpt-4o-mini action-selection calls.
_ACTION_SYSTEM_PROMPT = """You are a research agent building a pre-earnings financial report.
You have already collected some context via retrieval and tool calls. Your job is to
decide what to do next: gather more specific information, or declare FINISH if you
have enough to write a high-quality report.

Available actions (choose exactly one):
  RETRIEVE_FROM_FILING(query) — semantic search over the primary ticker's 10-Q/10-K chunks
  SEARCH_NEWS(query)          — Tavily web search for recent news on any topic
  LOOKUP_RELATED_TICKER(ticker) — fetch and search a related company's latest SEC filing
  FINISH                      — you have enough context to write the report

Important rules:
  - If a RETRIEVE_FROM_FILING query already returned "+0 new chunks" in the trace,
    do NOT repeat it. Try a different query, a different action, or FINISH.
  - If you have no clear evidence that another retrieval would add value, choose FINISH.

Respond with:
  - thought: one sentence explaining what's missing and why you chose this action
  - action: exactly one of the four action names above (no other values allowed)
  - argument: the query string, ticker symbol, or empty string for FINISH"""


# ---------------------------------------------------------------------------
# Agent-specific data models
# ---------------------------------------------------------------------------

class TraceStep(BaseModel):
    # One step in the agent's reasoning loop.
    iteration:   int   # 0 = initial retrieval steps; 1+ = agent-loop decisions
    thought:     str   # LLM's reasoning, or a description for synthetic steps
    action:      str   # one of VALID_ACTIONS
    argument:    str   # the input to the action (query, ticker, or "")
    observation: str   # what was returned (short summary, not full text)
    timestamp:   str   # ISO-8601 UTC
    tokens_used: int   # gpt-4o-mini tokens for this step (0 for synthetic steps)


class ReasoningTrace(BaseModel):
    # Complete reasoning trace for one run_agent() call.
    ticker:           str
    total_iterations: int
    steps:            list[TraceStep]


class _AgentDecision(BaseModel):
    # Internal structured output model for gpt-4o-mini action selection.
    # Not exported — only used within this module.
    thought:  str   # one sentence explaining the reasoning
    action:   str   # must be in VALID_ACTIONS (validated post-parse)
    argument: str   # depends on action; "" for FINISH


# ---------------------------------------------------------------------------
# Module-level OpenAI client (created once, reused across calls)
# ---------------------------------------------------------------------------

_client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _now_utc() -> str:
    # Return the current UTC time as an ISO-8601 string.
    return datetime.now(timezone.utc).isoformat()


def _render_chunks_for_prompt(
    all_chunks: dict[str, Chunk],
    primary_ticker: str,
) -> str:
    # Render the deduplicated chunk dict as labeled context for the final call.
    # Primary-ticker chunks are labeled "[SECTION chunk N]".
    # Related-ticker chunks are labeled "[RELATED_TICKER SECTION chunk N]".
    parts = []
    for key, chunk in all_chunks.items():
        # Key format is "{ticker}_{chunk_index}"; split on first "_" only.
        chunk_ticker = key.split("_")[0]
        if chunk_ticker == primary_ticker:
            label = f"[{chunk.section} chunk {chunk.chunk_index}]"
        else:
            label = f"[{chunk_ticker} {chunk.section} chunk {chunk.chunk_index}]"
        parts.append(f"{label}\n{chunk.text}")
    return "\n\n".join(parts)


def _render_trace_for_prompt(steps: list[TraceStep]) -> str:
    # Format the trace as compact text for the action-selection prompt.
    # Excludes full chunk text to keep gpt-4o-mini's context small.
    lines = []
    for s in steps:
        lines.append(
            f"Iteration {s.iteration} | {s.action}({s.argument})\n"
            f"  Thought: {s.thought}\n"
            f"  Observation: {s.observation}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent(ticker: str, max_iterations: int = 8) -> tuple[EarningsReport, ReasoningTrace]:
    """Produce an EarningsReport and ReasoningTrace via the agent-reasoning loop.

    Starts with the same initial retrieval as Mode 1 (5 fixed queries + consensus
    + news). Then iterates: gpt-4o-mini picks one action per step; the action is
    executed and logged. When the agent chooses FINISH (or max_iterations is hit),
    gpt-4o produces the final structured report from all accumulated context.

    Raises ValueError if the model returns any empty source_citation fields.
    Raises RuntimeError if any external call (EDGAR, OpenAI, Finnhub, Tavily) fails.
    """
    ticker = ticker.upper().strip()
    wall_start = time.perf_counter()

    # Running counters updated throughout the pipeline.
    embed_tokens = 0
    mini_in = mini_out = 0
    gpt4o_in = gpt4o_out = 0

    trace_steps: list[TraceStep] = []

    # all_chunks is keyed by "{ticker}_{chunk_index}" so chunks from different
    # companies can coexist without collision.
    all_chunks: dict[str, Chunk] = {}
    all_news: list[dict] = []
    capped = False

    print(f"[agent] Starting agent-reasoning mode for {ticker}...")

    # ------------------------------------------------------------------
    # Step 1: Initial retrieval — mirrors Mode 1 exactly
    # ------------------------------------------------------------------

    # Fetch, chunk, embed, and upsert the primary ticker's filing.
    print(f"[agent] Fetching and indexing {ticker} filing...")
    store = ChromaStore()
    filing = fetch_latest_filing(ticker)
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    chunk_texts = [c.text for c in chunks]
    t0 = time.perf_counter()
    embeddings = embed_texts(chunk_texts)
    embed_elapsed = time.perf_counter() - t0
    embed_tokens += sum(len(t) for t in chunk_texts) // 4
    store.upsert(filing["accession_number"], chunks, embeddings)
    print(f"[agent] Indexed {len(chunks)} chunks in {embed_elapsed:.2f}s ({filing['filing_type']} {filing['filing_date']})")

    # Run all 5 fixed retrieval queries and collect unique chunks.
    for query in RETRIEVAL_QUERIES:
        results = store.query(filing["accession_number"], query, k=TOP_K)
        embed_tokens += len(query) // 4
        for chunk in results:
            key = f"{ticker}_{chunk.chunk_index}"
            if key not in all_chunks:
                all_chunks[key] = chunk

    # Log the 5-query sweep as a single synthetic trace step.
    init_obs = f"Retrieved {len(all_chunks)} unique chunks from {filing['filing_type']}."
    trace_steps.append(TraceStep(
        iteration=0,
        thought="Initial retrieval (no LLM decision).",
        action="RETRIEVE_FROM_FILING",
        argument="5 fixed queries",
        observation=init_obs,
        timestamp=_now_utc(),
        tokens_used=0,
    ))
    print(f"[agent] init retrieval: {init_obs}")

    # Fetch consensus and news; log as a single synthetic trace step.
    t0 = time.perf_counter()
    consensus_data = get_consensus(ticker)
    news_results   = search_news(f"{ticker} earnings", max_results=5)
    all_news.extend(news_results)
    tools_elapsed = time.perf_counter() - t0

    tools_obs = (
        f"consensus source={consensus_data['source']}; "
        f"{len(news_results)} news items fetched."
    )
    trace_steps.append(TraceStep(
        iteration=0,
        thought="Initial tool calls (no LLM decision).",
        action="SEARCH_NEWS",
        argument=f"{ticker} earnings",
        observation=tools_obs,
        timestamp=_now_utc(),
        tokens_used=0,
    ))
    print(f"[agent] init tools ({tools_elapsed:.2f}s): {tools_obs}")

    # ------------------------------------------------------------------
    # Step 2: Agent loop — gpt-4o-mini decides the next action each iteration
    # ------------------------------------------------------------------

    for iteration in range(1, max_iterations + 1):

        # Build a compact trace summary for the action-selection prompt.
        trace_text = _render_trace_for_prompt(trace_steps)

        user_msg = (
            f"Primary ticker: {ticker}\n"
            f"Filing: {filing['filing_type']} filed {filing['filing_date']}\n\n"
            f"--- AGENT TRACE SO FAR ---\n{trace_text}\n\n"
            "Identify what report sections (key_metrics, revenue_segments, "
            "forward_guidance, key_risks) still lack supporting evidence, "
            "then choose the most valuable next action or FINISH."
        )

        # Call gpt-4o-mini for cheap action selection.
        t0 = time.perf_counter()
        completion = _client.beta.chat.completions.parse(
            model=GPT4O_MINI_MODEL,
            messages=[
                {"role": "system", "content": _ACTION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            response_format=_AgentDecision,
            temperature=0,
        )
        step_elapsed = time.perf_counter() - t0
        decision: _AgentDecision = completion.choices[0].message.parsed
        step_tokens = completion.usage.prompt_tokens + completion.usage.completion_tokens
        mini_in  += completion.usage.prompt_tokens
        mini_out += completion.usage.completion_tokens

        # Validate the action name. If the model hallucinated an unknown action,
        # treat it as FINISH to avoid an infinite or broken loop.
        if decision.action not in VALID_ACTIONS:
            print(f"[agent] iter {iteration}: unknown action '{decision.action}' — treating as FINISH")
            decision.action   = "FINISH"
            decision.argument = ""
            decision.thought  = (
                f"[auto-corrected from unknown action] {decision.thought}"
            )

        print(f"[agent] iter {iteration} ({step_elapsed:.2f}s): {decision.action}({decision.argument!r})")
        print(f"[agent]   thought: {decision.thought}")

        # Execute the chosen action.
        observation = ""

        if decision.action == "FINISH":
            trace_steps.append(TraceStep(
                iteration=iteration,
                thought=decision.thought,
                action="FINISH",
                argument="",
                observation="Agent decided it has enough context.",
                timestamp=_now_utc(),
                tokens_used=step_tokens,
            ))
            break

        elif decision.action == "RETRIEVE_FROM_FILING":
            query = decision.argument or "key financial data"
            results = store.query(filing["accession_number"], query, k=TOP_K)
            embed_tokens += len(query) // 4

            added = 0
            top_obs = ""
            for chunk in results:
                key = f"{ticker}_{chunk.chunk_index}"
                if key not in all_chunks:
                    all_chunks[key] = chunk
                    added += 1
                if not top_obs:
                    snippet = chunk.text.replace("\n", " ")[:120]
                    top_obs = f"[{chunk.section} chunk {chunk.chunk_index}] \"{snippet}\""

            observation = f"+{added} new chunks. Top: {top_obs}" if top_obs else "No new results."

        elif decision.action == "SEARCH_NEWS":
            query = decision.argument or f"{ticker} earnings"
            new_items = search_news(query, max_results=3)
            all_news.extend(new_items)
            first_headline = new_items[0]["title"] if new_items else "none"
            observation = f"+{len(new_items)} news items. First: \"{first_headline}\""

        elif decision.action == "LOOKUP_RELATED_TICKER":
            related = decision.argument.upper().strip() if decision.argument else ""
            if not related:
                observation = "No ticker argument provided; skipping."
            else:
                try:
                    print(f"[agent]   fetching and indexing {related}...")
                    rel_filing = fetch_latest_filing(related)
                    rel_chunks = chunk_filing(rel_filing["clean_text"], rel_filing["raw_html"])
                    rel_texts  = [c.text for c in rel_chunks]
                    rel_embeddings = embed_texts(rel_texts)
                    embed_tokens += sum(len(t) for t in rel_texts) // 4
                    store.upsert(rel_filing["accession_number"], rel_chunks, rel_embeddings)

                    # Run all 5 fixed queries against the related ticker's filing.
                    before_count = len(all_chunks)
                    for query in RETRIEVAL_QUERIES:
                        results = store.query(rel_filing["accession_number"], query, k=TOP_K)
                        embed_tokens += len(query) // 4
                        for chunk in results:
                            key = f"{related}_{chunk.chunk_index}"
                            if key not in all_chunks:
                                all_chunks[key] = chunk

                    added = len(all_chunks) - before_count
                    observation = (
                        f"Retrieved {added} chunks from {related}'s "
                        f"{rel_filing['filing_type']} ({rel_filing['filing_date']})."
                    )
                except RuntimeError as exc:
                    observation = f"LOOKUP failed for {related}: {exc}"

        print(f"[agent]   observation: {observation[:120]}")

        trace_steps.append(TraceStep(
            iteration=iteration,
            thought=decision.thought,
            action=decision.action,
            argument=decision.argument,
            observation=observation,
            timestamp=_now_utc(),
            tokens_used=step_tokens,
        ))

    else:
        # The for-loop completed all iterations without a break (no FINISH).
        capped = True
        print(f"[agent] Hit max_iterations ({max_iterations}); producing report from current context.")

    # ------------------------------------------------------------------
    # Step 3: Final structured generation call (gpt-4o)
    # ------------------------------------------------------------------

    # If the loop was capped, add a warning so the model puts under-supported
    # fields in data_gaps rather than fabricating.
    gen_system = _GENERATION_SYSTEM_PROMPT
    if capped:
        gen_system += (
            "\n\nNote: the agent loop hit its iteration cap before choosing FINISH. "
            "Some report fields may be under-supported — put them in data_gaps."
        )

    chunks_section = _render_chunks_for_prompt(all_chunks, ticker)
    now_utc = _now_utc()
    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Report timestamp (use for report_generated_at): {now_utc}\n\n"
        f"--- RETRIEVED FILING CHUNKS ---\n{chunks_section}\n\n"
        f"--- CONSENSUS DATA (JSON) ---\n{json.dumps(consensus_data, default=str)}\n\n"
        f"--- RECENT NEWS (JSON) ---\n{json.dumps(all_news, default=str)}\n\n"
        "Produce the structured EarningsReport now."
    )

    print(
        f"[agent] Final generation ({GPT4O_MODEL}, "
        f"{len(all_chunks)} chunks, {len(all_news)} news items)..."
    )
    t0 = time.perf_counter()
    completion = _client.beta.chat.completions.parse(
        model=GPT4O_MODEL,
        messages=[
            {"role": "system", "content": gen_system},
            {"role": "user",   "content": user_prompt},
        ],
        response_format=EarningsReport,
        temperature=0,
    )
    gen_elapsed = time.perf_counter() - t0
    gpt4o_in  = completion.usage.prompt_tokens
    gpt4o_out = completion.usage.completion_tokens
    print(f"[agent] Generation -> {gen_elapsed:.2f}s ({gpt4o_in} in / {gpt4o_out} out tokens)")

    report = completion.choices[0].message.parsed
    _validate_citations(report)

    # ------------------------------------------------------------------
    # Step 4: Log cost and wall time totals
    # ------------------------------------------------------------------
    wall_elapsed = time.perf_counter() - wall_start

    embed_cost = (embed_tokens / 1000) * EMBEDDING_COST_PER_1K
    mini_cost  = (mini_in  / 1000) * GPT4O_MINI_INPUT_COST_PER_1K \
               + (mini_out / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K
    gen_cost   = (gpt4o_in  / 1000) * GPT4O_INPUT_COST_PER_1K \
               + (gpt4o_out / 1000) * GPT4O_OUTPUT_COST_PER_1K
    total_cost = embed_cost + mini_cost + gen_cost

    agent_loop_steps = sum(1 for s in trace_steps if s.iteration > 0)

    print(f"[agent] === TOTALS ===")
    print(f"[agent] Wall time:         {wall_elapsed:.2f}s")
    print(f"[agent] Embed tokens:      ~{embed_tokens} (~${embed_cost:.5f})")
    print(f"[agent] gpt-4o-mini:       {mini_in} in + {mini_out} out (${mini_cost:.5f})")
    print(f"[agent] gpt-4o generation: {gpt4o_in} in + {gpt4o_out} out (${gen_cost:.5f})")
    print(f"[agent] Total cost:        ${total_cost:.5f}")
    print(f"[agent] Agent loop steps:  {agent_loop_steps} (beyond initial retrieval)")

    return report, ReasoningTrace(
        ticker=ticker,
        total_iterations=len(trace_steps),
        steps=trace_steps,
    )


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.modes.agent [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ticker = (sys.argv[1] if len(sys.argv) > 1 else "PANW").upper()

    print(f"Running agent-reasoning mode for {ticker}...")
    report, trace = run_agent(ticker)

    # Save report and trace to data/eval/ for Phase 9 evaluation comparison.
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file   = out_dir / f"agent_{ticker}.json"
    trace_file = out_dir / f"agent_{ticker}_trace.json"
    out_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    trace_file.write_text(trace.model_dump_json(indent=2), encoding="utf-8")

    print(f"\nSaved to {out_file}")
    print(f"Trace saved to {trace_file}")
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

    # Print a human-readable trace summary.
    agent_steps = [s for s in trace.steps if s.iteration > 0]
    print(f"\nTrace steps:      {trace.total_iterations} total ({len(agent_steps)} agent-loop)")
    for s in agent_steps:
        print(f"  iter {s.iteration}: {s.action}({s.argument!r})")
        print(f"    thought: {s.thought}")
        print(f"    obs:     {s.observation[:100]}")
