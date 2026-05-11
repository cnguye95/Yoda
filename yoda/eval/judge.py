"""Model-as-judge scoring for Phase 9 evaluation.

judge_report() calls Anthropic claude-sonnet-4-6 to score an EarningsReport
against the source filing text using a five-dimension rubric. Results are
cached on disk so re-running the same report does not re-spend API budget.

Cache key: first 16 hex chars of SHA256(report.model_dump_json()).
Cache location: data/eval/judge_cache/{hash}.json

Cross-family choice: Anthropic judges OpenAI-generated reports to prevent
same-model bias, which would make the baseline look better than it is.

Smoke test: python -m yoda.eval.judge
"""

import hashlib
import json
import pathlib

from anthropic import Anthropic

from yoda import config
from yoda.eval.rubric import JudgeScores
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# PLAN.md requires the cross-family judge to be this exact model.
_JUDGE_MODEL = "claude-sonnet-4-6"

# Where to persist cached judge results. Created on first use.
_CACHE_DIR = pathlib.Path("data/eval/judge_cache")

# System prompt that instructs the judge how to grade.
_SYSTEM_PROMPT = (
    "You are an expert financial analyst grading a pre-earnings research "
    "report against the source 10-Q/10-K filing. Score the report on the "
    "five rubric dimensions using the submit_rubric_scores tool. "
    "Base your scores only on the provided report JSON and filing text."
)

# Human-readable rubric appended to the user message so the judge understands
# each dimension before scoring.
_RUBRIC_TEXT = """
RUBRIC (score each dimension 1-5):

1. extraction_completeness — Did the report extract the key facts that were
   actually available in the filing? 1 = almost nothing extracted,
   5 = all major facts captured.

2. accuracy — Are the figures and claims in the report correct relative to
   the source filing? 1 = multiple errors, 5 = fully accurate.

3. source_traceability — Do the source_citation fields in the report resolve
   to real sections or chunks in the filing? 1 = citations fabricated or
   missing, 5 = every citation pinpoints a real location.

4. relevance — Is the content focused on what matters for pre-earnings
   analysis (guidance, risks, segment performance)? 1 = mostly off-topic,
   5 = tightly relevant.

5. usefulness — Would a sell-side analyst find this report actionable before
   an earnings call? 1 = not useful, 5 = immediately useful.
"""


# ---------------------------------------------------------------------------
# Module-level Anthropic client (created once, reused across calls)
# ---------------------------------------------------------------------------

_client = Anthropic(api_key=config.ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_report(report: EarningsReport, filing_text: str) -> JudgeScores:
    """Score a report using Anthropic claude-sonnet-4-6 as the judge.

    Caches results to data/eval/judge_cache/{hash}.json keyed by the report's
    JSON content so re-running the same report doesn't re-spend.

    Args:
        report: the EarningsReport to evaluate.
        filing_text: the source 10-Q/10-K text the report was generated from.
                     Truncate to ~30K chars before passing.

    Returns:
        JudgeScores with five dimension scores and an overall comment.
    """
    # Compute a stable cache key from the report's serialized JSON.
    report_json = report.model_dump_json()
    cache_key = hashlib.sha256(report_json.encode()).hexdigest()[:16]
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Return cached result if it exists, avoiding redundant API spend.
    if cache_file.exists():
        print(f"  [judge] cache hit: {cache_key}")
        return JudgeScores.model_validate_json(cache_file.read_text(encoding="utf-8"))

    print(f"  [judge] scoring via {_JUDGE_MODEL} (key={cache_key})...")

    # Build the user message: rubric + report JSON + truncated filing.
    user_message = (
        f"{_RUBRIC_TEXT}\n\n"
        f"--- REPORT JSON ---\n{report_json}\n\n"
        f"--- FILING TEXT (truncated) ---\n{filing_text}"
    )

    # Define the single tool that forces the judge to return structured scores.
    # input_schema is derived from JudgeScores so the shape is authoritative.
    tool_def = {
        "name": "submit_rubric_scores",
        "description": "Submit rubric scores for the evaluated report.",
        "input_schema": JudgeScores.model_json_schema(),
    }

    # Call the Anthropic API with tool_choice set to force the tool call.
    response = _client.messages.create(
        model=_JUDGE_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[tool_def],
        tool_choice={"type": "tool", "name": "submit_rubric_scores"},
        messages=[{"role": "user", "content": user_message}],
    )

    # Verify we hit the intended model — catches wrong-key misconfiguration.
    assert response.model == _JUDGE_MODEL, (
        f"Model mismatch: expected {_JUDGE_MODEL}, got {response.model}"
    )

    # Extract the tool call input dict from the first content block.
    scores_dict = response.content[0].input

    # The model occasionally returns DimensionScore fields as JSON strings
    # rather than nested dicts. Only attempt parse on strings that look like
    # JSON objects (start with '{') to leave plain strings like overall_comment
    # untouched.
    coerced = {
        k: (json.loads(v) if isinstance(v, str) and v.startswith("{") else v)
        for k, v in scores_dict.items()
    }
    scores = JudgeScores.model_validate(coerced)

    # Persist to cache before returning.
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(scores.model_dump_json(indent=2), encoding="utf-8")
    print(f"  [judge] cached to {cache_file}")

    return scores


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.eval.judge
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    from yoda.ingest.edgar import fetch_latest_filing

    # Load the existing NFLX RAG-LLM report JSON from Phase 4.
    report_path = pathlib.Path("data/eval/rag_llm_NFLX.json")
    report = EarningsReport.model_validate_json(report_path.read_text(encoding="utf-8"))

    # Fetch the cached NFLX filing for source text (no network hit if cached).
    print("Loading NFLX filing text...")
    filing = fetch_latest_filing("NFLX")
    filing_text = filing["clean_text"][:30000]

    # Call the judge (first run hits API; second run should be instant cache hit).
    print("Calling judge...")
    scores = judge_report(report, filing_text)

    # Print all five dimension scores plus the overall comment.
    print(f"\nExtraction completeness: {scores.extraction_completeness.score}/5 — {scores.extraction_completeness.justification}")
    print(f"Accuracy:                {scores.accuracy.score}/5 — {scores.accuracy.justification}")
    print(f"Source traceability:     {scores.source_traceability.score}/5 — {scores.source_traceability.justification}")
    print(f"Relevance:               {scores.relevance.score}/5 — {scores.relevance.justification}")
    print(f"Usefulness:              {scores.usefulness.score}/5 — {scores.usefulness.justification}")
    print(f"\nOverall: {scores.overall_comment}")
