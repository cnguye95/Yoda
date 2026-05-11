"""Phase 9 evaluation runner.

run_eval() orchestrates all three modes (baseline, rag_llm, agent) against a
list of tickers, judges each report with judge_report(), and returns a
long-format DataFrame with per-row scores, latency, and cost.

Outputs written to disk:
  data/eval/results.csv   — full long-format DataFrame
  data/eval/summary.md    — mean scores and costs per mode, rendered as markdown

Cost/latency are extracted from each mode's existing stdout log lines:
  "Wall time: X.XXs"  and  "Total cost: $X.XXXXX"
All three modes already print these consistently, so we capture stdout rather
than refactoring the mode APIs.

CLI: python -m yoda.eval.runner [TICKER [TICKER ...]]
Default ticker: NFLX (cheap verification run, ~$0.30, 3-5 min).
Full set: AAPL AMZN JPM PANW NFLX COIN
"""

import contextlib
import io
import re
import time

import pandas as pd

from yoda.eval.judge import judge_report
from yoda.eval.rubric import JudgeScores
from yoda.ingest.chunker import chunk_filing
from yoda.ingest.edgar import fetch_latest_filing
from yoda.modes.agent import run_agent
from yoda.modes.baseline import run_baseline
from yoda.modes.rag_llm import run_rag_llm
from yoda.schema import EarningsReport


# ---------------------------------------------------------------------------
# Regex patterns for parsing cost and latency from captured stdout
# ---------------------------------------------------------------------------

_RE_WALL_TIME = re.compile(r"Wall time:\s+([\d.]+)s")
_RE_COST      = re.compile(r"Total cost:\s+\$([\d.]+)")


# ---------------------------------------------------------------------------
# Baseline excerpt builder
# (Duplicated from baseline.py __main__ because that block is not a function.
# Keeping it local avoids importing __main__-level code from another module.)
# ---------------------------------------------------------------------------

def _build_baseline_excerpt(filing: dict) -> str:
    # Build a ~5000-char excerpt preferring MD&A then Financial Statements,
    # matching the same logic used in baseline.py's __main__ smoke test.
    chunks = chunk_filing(filing["clean_text"], filing["raw_html"])
    preferred = ["MD&A", "Financial Statements"]
    parts: list[str] = []
    total = 0

    for section_name in preferred:
        for chunk in chunks:
            if chunk.section == section_name and total < 5000:
                parts.append(chunk.text)
                total += len(chunk.text)
        if total >= 5000:
            break

    return " ".join(parts)[:5000]


# ---------------------------------------------------------------------------
# Stdout-capture helpers
# ---------------------------------------------------------------------------

def _capture(fn, *args) -> tuple[EarningsReport, float, float, str]:
    # Run fn(*args) with stdout captured. Returns (report, latency_s, cost_usd, log).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report = fn(*args)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


def _capture_agent(ticker: str) -> tuple[EarningsReport, float, float, str]:
    # Thin variant of _capture for run_agent which returns (report, trace).
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        report, _trace = run_agent(ticker)
    elapsed = time.perf_counter() - t0
    log = buf.getvalue()

    wall_match = _RE_WALL_TIME.search(log)
    cost_match  = _RE_COST.search(log)
    latency = float(wall_match.group(1)) if wall_match else elapsed
    cost    = float(cost_match.group(1))  if cost_match  else 0.0

    return report, latency, cost, log


# ---------------------------------------------------------------------------
# Per-mode dispatch
# ---------------------------------------------------------------------------

def _run_mode(
    mode: str, ticker: str, filing: dict
) -> tuple[EarningsReport, float, float, str]:
    """Run one mode for one ticker. Returns (report, latency_s, cost_usd, log)."""
    if mode == "baseline":
        excerpt = _build_baseline_excerpt(filing)
        return _capture(run_baseline, ticker, excerpt)
    elif mode == "rag_llm":
        return _capture(run_rag_llm, ticker)
    elif mode == "agent":
        return _capture_agent(ticker)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ---------------------------------------------------------------------------
# Score extraction helper
# ---------------------------------------------------------------------------

def _scores_to_dict(scores: JudgeScores) -> dict:
    # Flatten the five DimensionScore instances into plain int columns.
    return {
        "extraction_completeness": scores.extraction_completeness.score,
        "accuracy":                scores.accuracy.score,
        "source_traceability":     scores.source_traceability.score,
        "relevance":               scores.relevance.score,
        "usefulness":              scores.usefulness.score,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_eval(
    tickers: list[str],
    modes: list[str] = ("baseline", "rag_llm", "agent"),
) -> pd.DataFrame:
    """Run each (ticker, mode) pair, judge it, return a long-format DataFrame.

    DataFrame columns:
        ticker, mode, extraction_completeness, accuracy, source_traceability,
        relevance, usefulness, latency_seconds, cost_usd

    Also writes data/eval/results.csv and data/eval/summary.md.
    """
    rows = []

    for ticker in tickers:
        ticker = ticker.upper()
        print(f"\n=== {ticker} ===")

        # Fetch the filing once per ticker; all three modes share it.
        print(f"  fetching filing for {ticker}...")
        filing = fetch_latest_filing(ticker)
        filing_text = filing["clean_text"][:30000]

        for mode in modes:
            print(f"  running mode={mode}...")
            try:
                report, latency, cost, log = _run_mode(mode, ticker, filing)
            except Exception as exc:
                print(f"  ERROR in mode={mode}: {exc}")
                continue

            # Judge the report against the filing text.
            scores = judge_report(report, filing_text)

            row = {
                "ticker": ticker,
                "mode": mode,
                "latency_seconds": round(latency, 2),
                "cost_usd": round(cost, 5),
                **_scores_to_dict(scores),
            }
            rows.append(row)
            print(f"  mode={mode} done — latency={latency:.1f}s cost=${cost:.4f}")

    df = pd.DataFrame(rows)

    # Write the full long-format results.
    import pathlib
    out_dir = pathlib.Path("data/eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'results.csv'}")

    # Build and write the summary markdown.
    _write_summary(df, out_dir / "summary.md")
    print(f"Summary saved to {out_dir / 'summary.md'}")

    return df


# ---------------------------------------------------------------------------
# Summary markdown builder
# ---------------------------------------------------------------------------

def _df_to_markdown(df: pd.DataFrame) -> str:
    # Render a DataFrame as a GitHub-flavored markdown table without tabulate.
    header = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    sep    = "| " + " | ".join(["---"] * (len(df.columns) + 1)) + " |"
    rows   = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(df.index, df.values)
    ]
    return "\n".join([header, sep] + rows)


def _write_summary(df: pd.DataFrame, path) -> None:
    # Compute mean scores and operational metrics grouped by mode.
    score_cols = [
        "extraction_completeness", "accuracy",
        "source_traceability", "relevance", "usefulness",
    ]
    ops_cols = ["latency_seconds", "cost_usd"]

    mode_means = df.groupby("mode")[score_cols + ops_cols].mean().round(2)

    lines = ["# Phase 9 Evaluation Summary", ""]
    lines.append("## Mean scores by mode")
    lines.append("")
    lines.append(_df_to_markdown(mode_means))
    lines.append("")

    # Per-ticker mean if more than one ticker was evaluated.
    if df["ticker"].nunique() > 1:
        ticker_means = df.groupby("ticker")[score_cols].mean().round(2)
        lines.append("## Mean scores by ticker")
        lines.append("")
        lines.append(_df_to_markdown(ticker_means))
        lines.append("")

    import pathlib
    pathlib.Path(path).write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point — run with: python -m yoda.eval.runner [TICKER ...]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Default to NFLX-only for the verification gate (~$0.30, 3-5 min).
    # Full run: python -m yoda.eval.runner AAPL AMZN JPM PANW NFLX COIN
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["NFLX"]
    df = run_eval(tickers)
    print(df.to_string(index=False))
