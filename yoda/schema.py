"""Pydantic v2 models for the Yoda structured earnings report.

Every downstream phase — baseline, RAG-LLM, agent, PDF, and eval — imports
from here. The schema enforces the project's core 'cite or skip' rule: any
field that carries a financial fact must also carry a source_citation. The
baseline and both modes are responsible for populating data_gaps when a fact
cannot be cited.

These models are also passed directly to OpenAI structured outputs
(client.beta.chat.completions.parse), so all types stay within the subset
that OpenAI's strict JSON schema mode supports (str, int, float, bool,
list, nested BaseModel, T | None).
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Sub-models — one per distinct fact type in the report
# ---------------------------------------------------------------------------

class Metric(BaseModel):
    # A single quantitative or qualitative metric extracted from the filing.
    # value is a string (e.g. "$10.2B", "23.5%") for display flexibility.
    name: str
    value: str
    unit: str
    source_citation: str   # which section / sentence this came from


class Segment(BaseModel):
    # One revenue or business segment reported in the filing.
    name: str
    revenue: str           # formatted string, e.g. "$4.5B"
    yoy_change: str        # e.g. "+12%" or "flat"
    commentary: str        # one-sentence description of drivers
    source_citation: str


class ForwardGuidance(BaseModel):
    # Management's stated outlook or guidance, quoted or closely paraphrased.
    text: str
    source_citation: str


class Risk(BaseModel):
    # A single risk factor from Item 1A (10-K) or equivalent (10-Q).
    description: str
    is_new: bool           # True if this risk wasn't present in the prior filing
    source_citation: str


class NewsItem(BaseModel):
    # One recent news article relevant to the ticker's upcoming earnings.
    headline: str
    date: str              # ISO date string, e.g. "2026-04-15"
    url: str
    relevance_note: str    # one sentence on why this matters for earnings


class ConsensusBlock(BaseModel):
    # Analyst consensus estimates, populated from Phase 3 get_consensus().
    # Fields are nullable because Finnhub free tier may not cover all tickers.
    eps_estimate: float | None
    revenue_estimate: float | None
    next_earnings_date: str | None
    source: str            # "finnhub" | "yfinance_backup" | "finnhub_empty"


# ---------------------------------------------------------------------------
# Top-level report model
# ---------------------------------------------------------------------------

class EarningsReport(BaseModel):
    # Header fields — who, what, when
    ticker: str
    company_name: str
    filing_type: str           # "10-Q" or "10-K"
    filing_date: str           # ISO date of the most recent filing
    report_generated_at: str   # ISO-8601 UTC timestamp

    # Core analysis — all source_citation fields are required (not optional)
    key_metrics:      list[Metric]
    revenue_segments: list[Segment]
    forward_guidance: ForwardGuidance
    key_risks:        list[Risk]

    # External data — news from Tavily, consensus from Finnhub
    recent_news: list[NewsItem]
    consensus:   ConsensusBlock

    # Synthesis — LLM-generated, no citation required for these lists
    bull_case:    list[str]
    bear_case:    list[str]
    what_to_watch: list[str]

    # Transparency — REQUIRED to list anything the system could not cite.
    # An empty list is fine when everything is covered; it must never be omitted.
    data_gaps: list[str]
