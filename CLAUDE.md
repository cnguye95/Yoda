## Coding Philosophy
- Before writing any solution, propose 2-3 approaches ranked by simplicity
- Wait for approval before implementing
- Prefer built-ins and stdlib over custom logic
- No classes unless state genuinely needs to be managed
- Complexity must be explicitly justified — default to the simplest working solution
- Apply YAGNI: don't build for hypothetical future requirements
- Prefer flat over nested (Zen of Python)

## Code Review Habit
- After any implementation, self-critique for unnecessary complexity
- Ask: "What could be removed without losing functionality?"

## General Rules
- No scaffolding, boilerplate, or placeholder code unless explicitly asked
- If a built-in or 5-line solution exists, use it before reaching for a library
- One thing at a time — do it well before expanding scope
- Add `#` comment descriptions above every logical code block explaining what it does.
- Comments are designed for explanation to user with only basic knowledge of Python and no knowledge of Yoda

## Project Overview

Yoda is a GenAI-powered pre-earnings research assistant for financial analysts. Given a ticker symbol, it fetches SEC filings from EDGAR, runs RAG over the filing text, pulls consensus estimates and recent news via external APIs, and generates a structured research dossier downloadable as PDF.

**Current status:** scaffolding only. `rag_mode.py`, `agent_mode.py`, `streamlit_app.py`, and `requirements.txt` exist but are empty. Implementation has not started.

## Two Modes

Both modes share the same ingestion and presentation layers; they differ in retrieval behavior.

- **RAG-LLM mode** (`rag_mode.py`) — fixed pipeline. Runs 5 predefined retrieval queries (revenue segments, forward guidance, risk factors, capex/margins, changes from prior filing), pulls consensus estimates and news via external tools, and generates the report in a single structured LLM call. Fast and consistent; intended for wide ticker coverage.
- **Agent-Reasoning mode** (`agent_mode.py`) — ReAct loop. Starts with the same initial retrieval, then iterates: observe → identify gaps → call tools (extra retrieval, news lookup, related-company lookup) → observe → repeat. Terminates when all report sections have sufficient context or a hard iteration cap is hit. The full reasoning trace is streamed to the UI for auditability.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Architecture

### Entry point: `streamlit_app.py`
Streamlit UI. Ticker input, mode toggle (RAG-LLM / Agent-Reasoning), "Generate Report" button. Progress bar for RAG-LLM, streaming reasoning text for Agent-Reasoning. Dispatches to `rag_mode` or `agent_mode`.

### Shared ingestion pipeline (implemented in both mode files)
1. Fetch the most recent 10-Q or 10-K from SEC EDGAR for the given ticker.
2. Clean and chunk by section header (MD&A, Risk Factors, Financial Statements, Quantitative Disclosures). Each chunk is tagged with its section of origin for citation traceability.
3. Embed and store in an in-memory vector index.
   - Embedding candidates: OpenAI `text-embedding-3-small`, or a `sentence-transformers` model.
   - Index candidates: FAISS, ChromaDB.

### Output contract
Every report contains: key metrics from the filing(s), consensus estimates, recent material news, bull/bear cases, and "what to watch for." Every claim must carry a source citation (filing section + page, or API + timestamp). Final PDFs are written to `reports/`.

### External tools (called from both modes)
- **Financial data API** — consensus estimates.
- **Web search API** — recent material news.

API keys live in environment variables and must never be committed. 

## Rules

- In every Python file, precede each logical block (imports, function, class, loop, branch, etc.) with a `#` comment explaining what it does.
- Comments are designed for explanation to user with only basic knowledge of Python and no knowledge of Yoda
