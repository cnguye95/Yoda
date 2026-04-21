## Project Overview

Yoda is a GenAI-powered pre-earnings research assistant for financial analysts. Given a ticker symbol, it fetches SEC filings from EDGAR, runs RAG over the filing text, pulls consensus estimates and recent news via external APIs, and generates a structured research dossier downloadable as PDF.

## Two Modes

Both modes share the same ingestion and presentation layers; they differ in retrieval behavior.

- **RAG-LLM mode** (`rag_mode.py`) — fixed pipeline. Runs 5 predefined retrieval queries (revenue segments, forward guidance, risk factors, capex/margins, changes from prior filing), pulls consensus estimates and news via external tools, and generates the report in a single structured LLM call. Fast and consistent; intended for wide ticker coverage.

- **Agent-Reasoning mode** (`agent_mode.py`) — ReAct loop. Starts with the same initial retrieval, then iterates: observe → identify gaps → call tools (extra retrieval, news lookup, related-company lookup) → observe → repeat. Terminates when all report sections have sufficient context or a hard iteration cap is hit. The full reasoning trace is streamed to the UI for auditability.


## Key Commands

```bash
# Run the Streamlit app
streamlit run streamlit_app.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Entry Point
`streamlit_app.py` — Streamlit UI. Ticker text input, mode toggle (RAG-LLM / Agent-Reasoning), "Generate Report" button. RAG-LLM shows a progress bar; Agent-Reasoning streams reasoning text. Calls into `rag_mode.py` or `agent_mode.py` depending on selection.

### Shared Ingestion Pipeline (implemented in both mode files)
1. Fetch most recent 10-Q or 10-K from SEC EDGAR for the given ticker.
2. Clean and chunk text by section header (MD&A, Risk Factors, Financial Statements, Quantitative Disclosures). Tag each chunk with its section for citation traceability.
3. Embed chunks (OpenAI `text-embedding-3-small` or `sentence-transformers`) and store in an in-memory vector index (FAISS or ChromaDB).

### `rag_mode.py` — RAG-LLM Mode
- Runs a fixed set of retrieval queries: `"revenue segment breakdown"`, `"forward guidance language"`, `"key risk factors"`, `"capital expenditure and margins"`, `"notable changes from prior filing"`.
- Retrieves top-k chunks per query, calls financial data API for consensus estimates and web search for recent news.
- Passes all context to the LLM in a single structured generation call.

### `agent_mode.py` — Agent-Reasoning Mode
- Starts with the same initial retrieval as RAG-LLM mode.
- Enters a ReAct loop: observe → reason → decide tool call (additional EDGAR search, news lookup, related company lookup) → observe → repeat.
- Loop terminates when all required report sections are covered or the iteration cap is hit.
- Full reasoning trace is logged and streamed to the UI for auditability.

### Output
Both modes produce a structured report with: key metrics from filing(s), consensus estimates, recent material news, bull/bear cases, and "what to watch for." Every claim includes a source citation (filing section + page, or API + timestamp). Final PDFs are written to `reports/`.

### APIs and Keys
API keys are stored as environment variables and excluded from the repo via `.gitignore`.

### External tools (called from both modes)
- **Financial data API** — consensus estimates.
- **Web search API** — recent material news.

### `reports/`
Directory where generated PDF dossiers are saved.
