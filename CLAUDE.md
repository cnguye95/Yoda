# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yoda is a GenAI-powered pre-earnings research assistant for financial analysts. Given a ticker symbol, it fetches SEC filings from EDGAR, runs RAG over the filing text, pulls consensus estimates and recent news via external APIs, and generates a structured research dossier.

Two modes share the same ingestion and presentation layers but differ in retrieval behavior:
- **RAG-LLM mode** — fixed pipeline, 4–5 predefined retrieval queries, single LLM generation call, fast throughput.
- **Agent-Reasoning mode** — ReAct-style agent loop that observes retrieval results, identifies gaps, issues follow-up tool calls, and iterates until all report sections are covered. Has a hard iteration cap.

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
Both modes produce a structured report with: key metrics from filing(s), consensus estimates, recent material news, bull/bear cases, and "what to watch for." Every claim includes a source citation (filing section + page, or API + timestamp). Report is downloadable as PDF.

### APIs and Keys
API keys are stored as environment variables and excluded from the repo via `.gitignore`. The two external tool integrations are:
- **Financial data API** — consensus estimates (key: to be added to env)
- **Web search API** — recent news (key: to be added to env)

### `reports/`
Directory where generated PDF dossiers are saved.

## Rules

- Add `#` comment descriptions above every logical code block explaining what it does.
