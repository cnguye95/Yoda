# Yoda — Pre-Earnings Research Assistant

> **Status:** All 9 phases complete. Evaluation results across NFLX, COIN, and PANW are in [`data/eval/summary.md`](data/eval/summary.md).

---
## 1. Context, User, and Problem

**Who:** A buy-side or sell-side analyst who covers 10–50 tickers and prepares for earnings calls individually, often the night before.

**The workflow being improved:** Pre-earnings research. Before every earnings call, an analyst reads the most recent 10-Q or 10-K, pulls consensus estimates, scans recent news, and assembles a one-page dossier: key metrics, segment trends, forward guidance, risk flags, and what to watch. This typically takes 60–90 minutes per ticker and is done manually, tab by tab.

**Why it matters:** Earnings surprise is one of the highest-signal events in equity analysis, and analyst preparation directly affects how quickly they can react. The bottleneck is not judgment — it is assembly time: fetching the right SEC filing section, locating a specific guidance quote, cross-checking segment revenue against prior quarters. A system that automates citation-backed assembly frees the analyst to focus on the 10% that is genuinely interpretive.

**The failure mode we are solving for:** Generic LLM summaries of earnings filings hallucinate figures and drop citations. An analyst cannot use a report that might be fabricated. Every claim in Yoda must carry a source citation (section name + chunk ID from the filing, or API name + timestamp for external data); anything that cannot be cited goes into a visible `data_gaps` list rather than being silently invented.

---

## 2. Solution and Design

Yoda fetches the most recent 10-Q (or 10-K fallback) from SEC EDGAR for a given ticker, chunks it by section, embeds and indexes it in a local ChromaDB vector store, enriches with analyst consensus and news, and generates a structured `EarningsReport` with two different retrieval strategies. The report downloads as a PDF.

### Two Modes

Both modes share the same ingestion and presentation layers; they differ in how they retrieve evidence before generation.

- **RAG-LLM mode** (`yoda/modes/rag_llm.py`) — fixed pipeline. Runs 5 predefined retrieval queries (revenue segments, forward guidance, risk factors, capex/margins, changes from prior filing), pulls consensus estimates and news, and generates the report in a single structured LLM call. Fast and consistent; intended for wide ticker coverage.

- **Agent-Reasoning mode** (`yoda/modes/agent.py`) — ReAct loop. Starts with the same initial retrieval, then iterates: observe → identify gaps → call tools (extra retrieval, news lookup, related-company lookup) → observe → repeat. Terminates at FINISH or a hard iteration cap. The full reasoning trace is streamed to the UI for auditability.

### Key Design Choices

| Choice | Rationale |
|---|---|
| `gpt-4o` for generation, `gpt-4o-mini` for agent steps | Accuracy where it counts; cost control in the loop |
| ChromaDB persistent client | No native deps on Windows; persists across runs without a server |
| `reportlab` for PDF (not weasyprint) | weasyprint requires GTK3/Pango/Cairo native libs on Windows; reportlab is pure Python |
| Hand-rolled ReAct loop | More transparent than LangGraph; every step is loggable and inspectable |
| Cite-or-skip rule | Any uncitable fact goes to `data_gaps` instead of the report; the LLM is instructed to never fill gaps with general knowledge |
| `claude-sonnet-4-6` as eval judge | Different model family from the OpenAI-based generation system; prevents same-model bias in scoring |

### Architecture

```
app.py (Streamlit)
    ├── yoda/ingest/edgar.py       # SEC EDGAR fetch + disk cache
    ├── yoda/ingest/chunker.py     # section-aware HTML → text chunks
    ├── yoda/retrieval/            # text-embedding-3-small + ChromaDB
    ├── yoda/tools/consensus.py    # Finnhub + FMP fallback
    ├── yoda/tools/news.py         # Tavily search
    ├── yoda/modes/rag_llm.py      # Mode 1: fixed pipeline
    ├── yoda/modes/agent.py        # Mode 2: ReAct loop
    ├── yoda/schema.py             # EarningsReport Pydantic model
    ├── yoda/report/pdf.py         # EarningsReport → PDF (reportlab)
    └── yoda/eval/                 # Phase 9 evaluation harness
```

---

## 3. Evaluation and Results

### Baseline

A **prompt-only baseline** (`yoda/modes/baseline.py`) makes a single `gpt-4o` call with a manually sliced ~5000-character excerpt from the filing (MD&A preferred, Financial Statements fallback). No RAG, no agent loop. This represents the "just give the LLM a chunk of the filing" approach and sets the lower bound.

### Rubric

Reports were scored by `claude-sonnet-4-6` (cross-family to prevent self-grading bias) on five dimensions, each rated 1–5:

| Dimension | What it measures |
|---|---|
| **Extraction completeness** | Did the report extract the key facts actually available in the filing? |
| **Accuracy** | Are the figures correct relative to the source filing? |
| **Source traceability** | Do citations resolve to real sections or chunks? |
| **Relevance** | Is the content focused on pre-earnings analysis? |
| **Usefulness** | Would a sell-side analyst find this actionable before an earnings call? |

### Test Set

Three tickers across different sectors: **NFLX** (media/streaming), **COIN** (crypto exchange), **PANW** (cybersecurity).

### Results

Mean scores across 3 tickers (higher is better; max 5):

| Mode | Extract | Accuracy | Traceability | Relevance | Usefulness | Latency | Cost/report |
|---|---|---|---|---|---|---|---|
| **Agent** | **2.67** | **2.67** | **2.33** | **3.0** | **2.33** | 25.8s | ~$0.05 |
| **RAG-LLM** | **2.67** | 2.33 | 2.0 | **3.0** | 2.0 | 13.2s | ~$0.03 |
| **Baseline** | 1.67 | 2.33 | 1.67 | 2.33 | 1.33 | 8.1s | ~$0.00 |

**Key findings:**

- **Agent vs RAG-LLM:** Agent scores higher on source traceability (2.33 vs 2.0) and usefulness (2.33 vs 2.0) at the cost of ~2× latency and ~65% more spend. The iterative retrieval loop fills evidence gaps that the fixed query set misses.
- **RAG-LLM vs Baseline:** Both RAG modes score significantly higher than the baseline on extraction completeness (2.67 vs 1.67) and usefulness (≥2.0 vs 1.33). RAG access to the full indexed filing vs a static 5000-char excerpt explains the gap.
- **Source traceability is the weakest dimension across all modes** (max 2.33). Citations use internal chunk IDs that are not directly visible in the filed document, limiting verifiability. This is the clearest area for improvement.
- **COIN outperforms PANW across all modes**, likely because Coinbase's 10-K has cleaner tabular financial data that chunks and retrieves more predictably than PANW's narrative-heavy risk sections.

Full per-ticker results are in [`data/eval/results.csv`](data/eval/results.csv) and [`data/eval/summary.md`](data/eval/summary.md).

---

## 4. Artifact Snapshot

### UI Flow

```
┌─────────────────────────────────────────────────┐
│  Yoda — pre-earnings research assistant          │
│                                                  │
│  Ticker  [  NFLX         ]                       │
│  Mode    ○ RAG-LLM (fast)                        │
│          ● Agent-Reasoning (deep)                │
│                                                  │
│  [  Generate Report  ]                           │
└─────────────────────────────────────────────────┘
           ↓ (on click)
┌─────────────────────────────────────────────────┐
│  ▼ Generating Agent-Reasoning report for NFLX   │
│  ┌───────────────────────────────────────────┐  │
│  │ Fetching NFLX filing from EDGAR...        │  │
│  │ Chunking 847 sections...                  │  │
│  │ Embedding 312 chunks...                   │  │
│  │ [agent] iter 1: RETRIEVE_FROM_FILING(     │  │
│  │   "revenue by segment Q1 2026")           │  │
│  │ [agent] iter 2: RETRIEVE_FROM_FILING(     │  │
│  │   "forward guidance operating margin")    │  │
│  │ [agent] iter 3: FINISH                    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
           ↓ (complete)
┌─────────────────────────────────────────────────┐
│  NFLX — Netflix, Inc.           10-Q 2026-04-18  │
│                                                  │
│  Key Metrics  Segments  Risks   Data Gaps        │
│      8            4       6         2            │
│                                                  │
│  [  Download PDF Report  ]                       │
│                                                  │
│  ▶ Bull / Bear / What to Watch                   │
│  ▶ Data gaps (2)                                 │
│  ▶ Reasoning trace (3 agent steps)               │
└─────────────────────────────────────────────────┘
```

### Sample Report Fields (NFLX, RAG-LLM mode)

```json
{
  "ticker": "NFLX",
  "company_name": "Netflix, Inc.",
  "filing_type": "10-Q",
  "filing_date": "2026-04-18",
  "key_metrics": [
    {
      "name": "Total Revenue",
      "value": "$12,249,757K",
      "unit": "",
      "source_citation": "Financial Statements chunk 57"
    },
    {
      "name": "Operating Margin",
      "value": "32.3",
      "unit": "%",
      "source_citation": "MD&A chunk 65"
    }
  ],
  "forward_guidance": {
    "text": "Management expects continued margin expansion driven by advertising tier growth...",
    "source_citation": "MD&A chunk 72"
  },
  "data_gaps": [
    "EPS figures not cited — not present in provided filing sections",
    "Paid net additions not extracted — subscriber table not in retrieved chunks"
  ]
}
```

### PDF Output Structure

The downloaded PDF contains: cover page (ticker, company, filing date), key metrics table, revenue segments table, forward guidance blockquote, key risks (new risks flagged in red), analyst consensus, recent news with hyperlinks, bull/bear/watch bullets, and data gaps in amber.

---

## 5. Setup and Usage

### Prerequisites

- Python 3.11+
- Conda (recommended) or virtualenv
- API keys for OpenAI, Anthropic, Finnhub, and Tavily (all have free tiers sufficient for testing)

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd Yoda

# Create and activate a conda environment (or use virtualenv)
conda create -n yoda python=3.11
conda activate yoda

# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
cp .env.example .env
```

Edit `.env` with your keys:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FINNHUB_API_KEY=...
TAVILY_API_KEY=tvly-...
SEC_USER_AGENT="Your Name your@email.com"
```

### Run the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), enter a ticker (e.g. `NFLX`), choose a mode, and click **Generate Report**.

The first run for a ticker fetches and indexes the SEC filing (~30s). Subsequent runs for the same ticker are fast because the filing and ChromaDB index are cached to disk.

### Run the Evaluation Harness

```bash
# Single ticker (cheap verification, ~$0.30, 3–5 min)
python -m yoda.eval.runner NFLX

# Three tickers used in the paper
python -m yoda.eval.runner NFLX COIN PANW
```

Outputs `data/eval/results.csv` and `data/eval/summary.md`. Judge results are cached in `data/eval/judge_cache/` so repeated runs are free.

### Run Individual Mode Smoke Tests

```bash
# Baseline (prompt-only, no RAG)
python -m yoda.modes.baseline

# RAG-LLM mode
python -m yoda.modes.rag_llm

# Agent-Reasoning mode
python -m yoda.modes.agent

# PDF generation
python -m yoda.report.pdf NFLX
```

---

## Required API Keys

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM generation (`gpt-4o`) and cheap iterative steps (`gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Yes | Model-as-judge in the evaluation harness (`claude-sonnet-4-6`) |
| `FINNHUB_API_KEY` | Yes | Analyst consensus estimates |
| `TAVILY_API_KEY` | Yes | News and web search |
| `SEC_USER_AGENT` | Yes | SEC EDGAR requires a contact string, e.g. `"Name email@example.com"` |

---

## Repo Structure

```
yoda/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── .env.example
├── yoda/
│   ├── config.py                   # env vars, model names, constants
│   ├── ingest/
│   │   ├── edgar.py                # ticker -> SEC EDGAR fetch
│   │   └── chunker.py              # section-aware chunking
│   ├── retrieval/
│   │   ├── embeddings.py           # embedding wrapper
│   │   └── vector_store.py         # ChromaDB wrapper
│   ├── tools/
│   │   ├── consensus.py            # Finnhub + yfinance wrapper
│   │   └── news.py                 # Tavily wrapper
│   ├── modes/
│   │   ├── baseline.py             # prompt-only baseline
│   │   ├── rag_llm.py              # Mode 1: RAG-LLM
│   │   └── agent.py                # Mode 2: ReAct agent loop
│   ├── schema.py                   # Pydantic report schema
│   ├── report/
│   │   └── pdf.py                  # EarningsReport -> PDF
│   └── eval/
│       ├── rubric.py               # rubric Pydantic models
│       ├── judge.py                # Anthropic model-as-judge
│       └── runner.py               # batch eval over test tickers
├── data/
│   ├── filings/                    # cached EDGAR downloads (gitignored)
│   ├── chroma/                     # ChromaDB persistence (gitignored)
│   └── eval/                       # eval outputs and judge cache
└── tests/
    └── test_smoke.py
```

---

## APIs and Keys

API keys are stored as environment variables (`.env`) and excluded from the repo via `.gitignore`. See `.env.example` for the full template.
