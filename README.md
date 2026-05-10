**Status:** Phase 0 complete (project setup). See [PLAN.md](PLAN.md) for upcoming phases.

## Project Overview

Yoda is a GenAI-powered pre-earnings research assistant for financial analysts. Given a ticker symbol, it fetches SEC filings from EDGAR, runs RAG over the filing text, pulls consensus estimates and recent news via external APIs, and generates a structured research dossier downloadable as PDF.

## Two Modes

Both modes share the same ingestion and presentation layers; they differ in retrieval behavior.

- **RAG-LLM mode** (`yoda/modes/rag_llm.py`) — fixed pipeline. Runs 5 predefined retrieval queries (revenue segments, forward guidance, risk factors, capex/margins, changes from prior filing), pulls consensus estimates and news via external tools, and generates the report in a single structured LLM call. Fast and consistent; intended for wide ticker coverage.

- **Agent-Reasoning mode** (`yoda/modes/agent.py`) — ReAct loop. Starts with the same initial retrieval, then iterates: observe → identify gaps → call tools (extra retrieval, news lookup, related-company lookup) → observe → repeat. Terminates when all report sections have sufficient context or a hard iteration cap is hit. The full reasoning trace is streamed to the UI for auditability.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env template and fill in your keys
cp .env.example .env

# Run the Streamlit app
streamlit run app.py
```

## Required API Keys

Add these to your `.env` file (see `.env.example` for the template):

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM generation (`gpt-4o`) and cheap iterative steps (`gpt-4o-mini`) |
| `ANTHROPIC_API_KEY` | Yes | Model-as-judge in the evaluation harness (`claude-sonnet-4-6`) |
| `FINNHUB_API_KEY` | Yes | Analyst consensus estimates |
| `TAVILY_API_KEY` | Yes | News and web search |
| `SEC_USER_AGENT` | Yes | SEC EDGAR requires a contact string, e.g. `"Name email@example.com"` |

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
│       ├── rubric.py               # rubric definition
│       ├── judge.py                # Anthropic model-as-judge
│       └── runner.py               # batch eval over test tickers
├── data/
│   ├── filings/                    # cached EDGAR downloads (gitignored)
│   ├── chroma/                     # ChromaDB persistence (gitignored)
│   └── eval/                       # eval outputs and judge cache
└── tests/
    └── test_smoke.py
```

## Architecture

### Entry Point
`app.py` — Streamlit UI. Ticker text input, mode toggle (RAG-LLM / Agent-Reasoning), "Generate Report" button. RAG-LLM shows a progress bar with pipeline stages; Agent-Reasoning streams the reasoning trace live. Calls into `yoda/modes/rag_llm.py` or `yoda/modes/agent.py` depending on selection.

### Shared Ingestion Pipeline
1. Fetch the most recent 10-Q (or 10-K fallback) from SEC EDGAR for the given ticker.
2. Clean and chunk text by section header (MD&A, Risk Factors, Financial Statements, Quantitative Disclosures). Tag each chunk with its section for citation traceability.
3. Embed chunks and store in a persistent ChromaDB collection keyed by accession number.

### `rag_llm.py` — RAG-LLM Mode
- Runs a fixed set of retrieval queries against the filing index.
- Calls Finnhub + yfinance for consensus estimates and Tavily for recent news.
- Passes all context to `gpt-4o` in a single structured generation call (OpenAI structured outputs, enforced by the `EarningsReport` Pydantic schema).

### `agent.py` — Agent-Reasoning Mode
- Starts with the same initial retrieval as RAG-LLM mode.
- Enters a ReAct loop: observe → reason → choose action (RETRIEVE_FROM_FILING, SEARCH_NEWS, LOOKUP_RELATED_TICKER, FINISH) → observe → repeat.
- Loop terminates at FINISH or the iteration cap. Unanswered queries go into `data_gaps`; the report is still produced.
- Every iteration's thought, action, and observation is logged in a `ReasoningTrace` and streamed to the UI.

### Output
Both modes produce a structured `EarningsReport` with: key metrics, revenue segments, forward guidance, key risks, recent news, consensus estimates, bull/bear cases, and what to watch for. Every claim includes a source citation (filing section + chunk ID, or API + timestamp). Reports render to PDF via `yoda/report/pdf.py`.

### Evaluation Harness
`yoda/eval/` runs all three approaches (baseline, Mode 1, Mode 2) over a test ticker set and scores each report using `claude-sonnet-4-6` as the model-as-judge (different model family from the OpenAI-based system being graded). Outputs `data/eval/results.csv` and `data/eval/summary.md`.

### APIs and Keys
API keys are stored as environment variables (`.env`) and excluded from the repo via `.gitignore`.
