"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Wraps Phases 1–7 in a browser UI: ticker input, mode toggle, streaming
progress, summary card, and PDF download. Both modes use print() internally
for progress logging; we redirect stdout to a Streamlit placeholder so the
log streams live in the browser without any mode-side refactoring.
"""

import contextlib
import io
import pathlib
import tempfile

import streamlit as st

from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.agent import run_agent
from yoda.modes.rag_llm import run_rag_llm
from yoda.report.pdf import report_to_pdf


# ---------------------------------------------------------------------------
# Page configuration — must come before any other st.* call.
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Yoda", layout="centered")


# ---------------------------------------------------------------------------
# Streaming stdout capture
# ---------------------------------------------------------------------------

class _StreamlitLogStream:
    # File-like object that mirrors writes into both an in-memory buffer and
    # a Streamlit placeholder. Each write() refreshes the placeholder so the
    # browser shows print() output line-by-line in real time.
    def __init__(self, placeholder):
        self.buf = io.StringIO()
        self.placeholder = placeholder

    def write(self, text):
        self.buf.write(text)
        # .code() renders monospace which suits log output.
        self.placeholder.code(self.buf.getvalue())
        return len(text)

    def flush(self):
        # Required by the file-like protocol; nothing to do since write()
        # already pushes updates to the placeholder.
        pass


# ---------------------------------------------------------------------------
# PDF generation helper (cached per-ticker via session state)
# ---------------------------------------------------------------------------

def _ensure_pdf(report) -> bytes:
    # Return cached PDF bytes if we already generated one for this ticker.
    if (
        "pdf_bytes" in st.session_state
        and st.session_state.get("pdf_ticker") == report.ticker
    ):
        return st.session_state["pdf_bytes"]

    # reportlab writes to a path; use a temp file then read back the bytes.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        pdf_path = f.name
    try:
        # fetch_latest_filing is cached on disk (Phase 1) so this is fast.
        filing = fetch_latest_filing(report.ticker)
        report_to_pdf(report, pdf_path, filing_url=filing["url"])
        pdf_bytes = pathlib.Path(pdf_path).read_bytes()
    finally:
        pathlib.Path(pdf_path).unlink(missing_ok=True)

    st.session_state["pdf_bytes"]  = pdf_bytes
    st.session_state["pdf_ticker"] = report.ticker
    return pdf_bytes


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Yoda — pre-earnings research assistant")
st.caption(
    "Generates a structured, citation-backed pre-earnings report from the most "
    "recent 10-Q / 10-K filing, analyst consensus, and recent news."
)


# ---------------------------------------------------------------------------
# Input controls
# ---------------------------------------------------------------------------

# Ticker input: normalize to uppercase and strip whitespace.
ticker_raw = st.text_input("Ticker", placeholder="e.g. NFLX")
ticker = ticker_raw.strip().upper()

# Mode toggle — fast RAG vs deep agent loop.
mode_label = st.radio(
    "Mode",
    ["RAG-LLM (fast)", "Agent-Reasoning (deep)"],
    horizontal=True,
)

# Generate button is disabled until the user enters a ticker.
generate_clicked = st.button("Generate Report", disabled=not ticker, type="primary")


# ---------------------------------------------------------------------------
# Generation handler — runs when the button is clicked.
# ---------------------------------------------------------------------------

if generate_clicked:
    # Show a collapsible status box with the streaming log inside.
    with st.status(
        f"Generating {mode_label} report for {ticker}...",
        expanded=True,
    ) as status:
        log_box = st.empty()
        stream = _StreamlitLogStream(log_box)
        try:
            # Redirect stdout so every print() in the mode functions gets
            # routed through our placeholder for live display.
            with contextlib.redirect_stdout(stream):
                if mode_label.startswith("RAG-LLM"):
                    report = run_rag_llm(ticker)
                    trace = None
                else:
                    report, trace = run_agent(ticker)

            status.update(label=f"{ticker} report ready", state="complete")

            # Persist results in session state so they survive reruns.
            st.session_state["report"] = report
            st.session_state["trace"]  = trace
            st.session_state["ticker"] = ticker
            # Invalidate any previously cached PDF.
            st.session_state.pop("pdf_bytes",  None)
            st.session_state.pop("pdf_ticker", None)

        except Exception as exc:
            status.update(label="Generation failed", state="error")
            st.error(f"Could not generate report for {ticker}: {exc}")


# ---------------------------------------------------------------------------
# Results section — shown only after a successful generation.
# ---------------------------------------------------------------------------

if "report" in st.session_state:
    report = st.session_state["report"]

    st.divider()
    st.subheader(f"{report.ticker} — {report.company_name}")
    st.caption(f"{report.filing_type} filed {report.filing_date}")

    # Four-column metric strip summarising the report.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Key Metrics",     len(report.key_metrics))
    col2.metric("Segments",        len(report.revenue_segments))
    col3.metric("Risks",           len(report.key_risks))
    col4.metric("Data Gaps",       len(report.data_gaps))

    # PDF download button — generated once and cached.
    pdf_bytes = _ensure_pdf(report)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"report_{report.ticker}.pdf",
        mime="application/pdf",
    )

    # Bull / Bear / Watch summary in an expander.
    with st.expander("Bull / Bear / What to Watch"):
        st.markdown("**Bull case**")
        for point in report.bull_case:
            st.markdown(f"- {point}")
        st.markdown("**Bear case**")
        for point in report.bear_case:
            st.markdown(f"- {point}")
        st.markdown("**What to watch**")
        for point in report.what_to_watch:
            st.markdown(f"- {point}")

    # Data gaps expander — visible only when there are gaps to show.
    if report.data_gaps:
        with st.expander(f"Data gaps ({len(report.data_gaps)})"):
            for gap in report.data_gaps:
                st.markdown(f"- {gap}")

    # Reasoning trace expander — agent mode only.
    if st.session_state.get("trace"):
        trace = st.session_state["trace"]
        agent_steps = [s for s in trace.steps if s.iteration > 0]
        with st.expander(f"Reasoning trace ({len(agent_steps)} agent steps)"):
            for s in agent_steps:
                st.markdown(f"**Iter {s.iteration}** — `{s.action}({s.argument})`")
                st.write(f"Thought: {s.thought}")
                st.write(f"Observation: {s.observation}")
                st.markdown("---")
