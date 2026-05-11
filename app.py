"""Streamlit entry point for Yoda. Run with: `streamlit run app.py`.

Phase 10 UI: single-mode personality panel (Fast/Deep toggle) plus a ticker
queue for batch overnight runs. Streaming progress, summary card, and PDF
download are unchanged from prior phases. The queue panel adds: live status
table, ZIP download of all PDFs when batch completes, and a completion
notification banner with auto-scroll.

Both run paths use print() internally for progress logging; we redirect
stdout to a Streamlit placeholder so the log streams live in the browser
without any mode-side refactoring.
"""

import contextlib
import io
import pathlib
import tempfile
import threading

import streamlit as st
import streamlit.components.v1 as components

from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.personality_panel import run_personality_panel
from yoda.queue.processor import process_queue
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
        self._lock = threading.Lock()

    def write(self, text):
        # Lock so concurrent personality threads can't interleave the two steps.
        with self._lock:
            self.buf.write(text)
            try:
                self.placeholder.code(self.buf.getvalue())
            except Exception:
                # Streamlit raises NoSessionContext when called from a background
                # thread. Suppress it — the buffer still accumulates and is
                # rendered from the main thread after the run completes.
                pass
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
        # fetch_latest_filing is cached on disk so this is fast.
        filing = fetch_latest_filing(report.ticker)
        report_to_pdf(report, pdf_path, filing_url=filing["url"])
        pdf_bytes = pathlib.Path(pdf_path).read_bytes()
    finally:
        pathlib.Path(pdf_path).unlink(missing_ok=True)

    st.session_state["pdf_bytes"]  = pdf_bytes
    st.session_state["pdf_ticker"] = report.ticker
    return pdf_bytes


# ---------------------------------------------------------------------------
# Session-state defaults — populate keys we use so we don't need .get() guards
# everywhere downstream.
# ---------------------------------------------------------------------------

st.session_state.setdefault("queue", [])
st.session_state.setdefault("queue_results", None)
st.session_state.setdefault("queue_zip_bytes", None)
st.session_state.setdefault("queue_zip_name",  None)
st.session_state.setdefault("queue_status", "idle")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Yoda — pre-earnings research assistant")
st.caption(
    "Multi-agent personality panel: six analysts (Optimist, Pessimist, "
    "Conservative, Dreamer, Contrarian, Quant) investigate the most recent "
    "filing in parallel, then cross-critique before synthesis."
)


# ---------------------------------------------------------------------------
# Completion notification — pinned near top so it's immediately visible when
# the user returns to the tab after a long queue run.
# ---------------------------------------------------------------------------

if st.session_state["queue_status"] == "complete":
    results = st.session_state.get("queue_results") or []
    n_ok   = sum(1 for r in results if r["status"] == "complete")
    n_fail = sum(1 for r in results if r["status"] == "failed")
    if n_fail:
        st.success(
            f"Queue complete — {n_ok} reports generated, {n_fail} failed. "
            f"Download ZIP below."
        )
    else:
        st.success(
            f"Queue complete — {n_ok} reports generated. Download ZIP below."
        )

    if st.session_state.get("queue_zip_bytes"):
        st.download_button(
            label=f"Download All Reports (ZIP, {len(st.session_state['queue_zip_bytes']) // 1024} KB)",
            data=st.session_state["queue_zip_bytes"],
            file_name=st.session_state["queue_zip_name"] or "yoda_queue.zip",
            mime="application/zip",
            type="primary",
        )

    # Auto-scroll the page to the top so the notification is visible. We do
    # this once per "complete" transition by tying it to the queue_status.
    components.html(
        "<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
        height=0,
    )


# ---------------------------------------------------------------------------
# Tabs — Single ticker (live exploration) and Queue (overnight batch)
# ---------------------------------------------------------------------------

tab_single, tab_queue = st.tabs(["Single ticker", "Queue (batch)"])


# ---------------------------------------------------------------------------
# Tab 1 — single ticker
# ---------------------------------------------------------------------------

with tab_single:
    # Ticker input: normalize to uppercase and strip whitespace.
    ticker_raw = st.text_input(
        "Ticker", placeholder="e.g. NFLX", key="single_ticker_input",
    )
    ticker = ticker_raw.strip().upper()

    # Mode toggle — Fast (~35-50s) vs Deep (~75-95s with cross-critique).
    mode_label = st.radio(
        "Mode",
        ["Fast (no debate, ~40s)", "Deep (with cross-critique, ~90s)"],
        horizontal=True,
        key="single_mode_radio",
    )
    deep = mode_label.startswith("Deep")

    # Generate button is disabled until the user enters a ticker.
    generate_clicked = st.button(
        "Generate Report",
        disabled=not ticker,
        type="primary",
        key="single_generate",
    )

    # Generation handler — runs when the button is clicked.
    if generate_clicked:
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
                    report, personality_results, critique_messages = run_personality_panel(
                        ticker, deep=deep,
                    )

                status.update(label=f"{ticker} report ready", state="complete")
                # Render accumulated log from the main thread — background
                # threads couldn't call placeholder.code() directly.
                log_box.code(stream.buf.getvalue())

                # Persist results in session state so they survive reruns.
                st.session_state["report"] = report
                st.session_state["personality_results"] = personality_results
                st.session_state["critique_messages"]   = critique_messages
                st.session_state["ticker"] = ticker
                # Invalidate any previously cached PDF.
                st.session_state.pop("pdf_bytes",  None)
                st.session_state.pop("pdf_ticker", None)

            except Exception as exc:
                status.update(label="Generation failed", state="error")
                # Show whatever was logged before the failure.
                log_box.code(stream.buf.getvalue())
                st.error(f"Could not generate report for {ticker}: {exc}")

    # Results section — shown only after a successful generation.
    if "report" in st.session_state:
        report = st.session_state["report"]

        st.divider()
        st.subheader(f"{report.ticker} — {report.company_name}")
        st.caption(f"{report.filing_type} filed {report.filing_date}")

        # Four-column metric strip summarising the report.
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Key Metrics", len(report.key_metrics))
        col2.metric("Segments",    len(report.revenue_segments))
        col3.metric("Hypotheses",  len(report.hypotheses_explored))
        col4.metric("Data Gaps",   len(report.data_gaps))

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

        # Panel trace expander — replaces the old reasoning-trace expander.
        # Renders each personality's hypotheses + the critique message graph.
        if st.session_state.get("personality_results"):
            results = st.session_state["personality_results"]
            messages = st.session_state.get("critique_messages") or []

            with st.expander(
                f"Panel trace ({len(results)} personalities, {len(messages)} critique messages)"
            ):
                for pr in results:
                    clean_marker = "✓" if pr.finished_cleanly else "✗"
                    st.markdown(
                        f"**{clean_marker} {pr.personality}** — "
                        f"{pr.tool_calls_used} tools, {pr.wall_seconds:.1f}s, "
                        f"${pr.cost_usd:.4f}"
                    )
                    for h in pr.hypotheses:
                        st.markdown(
                            f"- `{h.id}` (confidence {h.confidence}/5) — "
                            f"*{h.question}*"
                        )
                        st.markdown(f"  > {h.summary}")
                    st.markdown("---")

                if messages:
                    st.markdown("**Critique messages**")
                    for m in messages:
                        color = {
                            "SUPPORTS":   "🟢",
                            "CHALLENGES": "🔴",
                            "EXTENDS":    "🔵",
                        }.get(m.message_type, "⚪")
                        st.markdown(
                            f"{color} **{m.from_personality} → "
                            f"`{m.target_hypothesis_id}`** "
                            f"[{m.message_type}]: {m.content}"
                        )


# ---------------------------------------------------------------------------
# Tab 2 — queue / batch
# ---------------------------------------------------------------------------

with tab_queue:
    st.caption(
        "Queue multiple tickers and run them sequentially. PDFs save to "
        "`data/reports/` as they complete; all are bundled into a ZIP at "
        "the end. Failures don't abort the batch."
    )

    # Two-column input: paste-list on the left, queue control on the right.
    col_input, col_actions = st.columns([2, 1])

    with col_input:
        queue_text = st.text_area(
            "Tickers (one per line or comma-separated)",
            placeholder="NFLX\nCOIN\nPANW",
            height=120,
            key="queue_text",
        )

    with col_actions:
        queue_mode = st.radio(
            "Mode",
            ["Fast", "Deep"],
            horizontal=False,
            key="queue_mode",
        )
        deep_queue = queue_mode == "Deep"

        # Parse the textarea into a normalized ticker list whenever the user
        # types — this lets us update the count in the button label.
        # Accept both newlines and commas as separators.
        raw_tickers = []
        for line in (queue_text or "").splitlines():
            for token in line.split(","):
                token = token.strip().upper()
                if token:
                    raw_tickers.append(token)

        add_btn = st.button(
            f"Add to Queue ({len(raw_tickers)} ticker{'s' if len(raw_tickers) != 1 else ''})",
            disabled=not raw_tickers,
            key="queue_add",
        )
        if add_btn:
            # Extend the queue with new entries, deduplicated against current.
            existing = set(st.session_state["queue"])
            for t in raw_tickers:
                if t not in existing:
                    st.session_state["queue"].append(t)
                    existing.add(t)

    # Display the current queue with a clear-all button.
    if st.session_state["queue"]:
        st.markdown("**Queue:**")
        st.write(", ".join(st.session_state["queue"]))
        clear_clicked = st.button("Clear queue", key="queue_clear")
        if clear_clicked:
            st.session_state["queue"] = []
            st.session_state["queue_results"] = None
            st.session_state["queue_status"] = "idle"
            st.rerun()
    else:
        st.info("Queue is empty. Add tickers above.")

    # Run button.
    run_clicked = st.button(
        "Run Queue",
        disabled=not st.session_state["queue"]
                  or st.session_state["queue_status"] == "running",
        type="primary",
        key="queue_run",
    )

    if run_clicked:
        st.session_state["queue_status"] = "running"

        # Per-ticker live status table. We render it as an empty placeholder
        # and update it from the on_progress callback below.
        status_placeholder = st.empty()

        # Pre-populate the table with "pending" rows so the user sees the
        # full list immediately, not just the one currently running.
        live_status: dict[str, str] = {t: "pending" for t in st.session_state["queue"]}

        def _render_status_table():
            # Build a simple markdown table from live_status. Streamlit
            # re-renders the placeholder each call so updates are immediate.
            rows = ["| Ticker | Status |", "|---|---|"]
            for t, s in live_status.items():
                # Color the status with an emoji so progress is scannable.
                emoji = (
                    "⏳" if s == "running"
                    else "✅" if s == "complete"
                    else "❌" if s.startswith("failed")
                    else "·"
                )
                rows.append(f"| {t} | {emoji} {s} |")
            status_placeholder.markdown("\n".join(rows))

        _render_status_table()

        def on_progress(ticker: str, status: str) -> None:
            # Streamlit callback. Updates the live table after each ticker
            # transition so the user sees progress in real time.
            live_status[ticker] = status
            _render_status_table()

        # Show streaming log too so the user can watch the panel internals.
        log_box = st.empty()
        stream = _StreamlitLogStream(log_box)

        try:
            with contextlib.redirect_stdout(stream):
                zip_path, results = process_queue(
                    st.session_state["queue"],
                    deep=deep_queue,
                    on_progress=on_progress,
                )

            # Read the ZIP into memory so the download_button can serve it
            # even if the file is later cleaned up.
            zip_bytes = pathlib.Path(zip_path).read_bytes()

            st.session_state["queue_results"]   = results
            st.session_state["queue_zip_bytes"] = zip_bytes
            st.session_state["queue_zip_name"]  = zip_path.name
            st.session_state["queue_status"]    = "complete"
            st.rerun()

        except Exception as exc:
            st.session_state["queue_status"] = "idle"
            log_box.code(stream.buf.getvalue())
            st.error(f"Queue run failed: {exc}")

    # If a queue has completed in this session, show the per-ticker results
    # table with individual download links. This is in addition to the ZIP
    # button at the top.
    if st.session_state.get("queue_results"):
        st.divider()
        st.subheader("Results")
        for r in st.session_state["queue_results"]:
            if r["status"] == "complete":
                pdf = pathlib.Path(r["pdf_path"])
                # Show each PDF as a download button so users can grab one
                # at a time without unpacking the ZIP.
                st.download_button(
                    label=f"✅ {r['ticker']} — {pdf.name} ({r['seconds']:.0f}s)",
                    data=pdf.read_bytes(),
                    file_name=pdf.name,
                    mime="application/pdf",
                    key=f"qdl_{r['ticker']}_{pdf.stem}",
                )
            else:
                st.error(f"❌ {r['ticker']} — {r['error']}")
