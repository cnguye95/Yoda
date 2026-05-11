"""Analyst consensus estimates and basic market metrics for a given ticker.

Two public functions:
  get_consensus(ticker)     — analyst EPS / revenue / earnings-date from
                              Finnhub, with a yfinance backup when free-tier
                              coverage is absent.
  get_basic_metrics(ticker) — current price, market cap, trailing EPS from
                              yfinance; always called, not a fallback.
"""

import sys
from datetime import datetime, timezone, date, timedelta

import finnhub
from finnhub.exceptions import FinnhubAPIException
import yfinance as yf

from yoda import config


# ---------------------------------------------------------------------------
# Internal Finnhub call wrapper
# ---------------------------------------------------------------------------

def _finnhub_call(fn):
    # Run a Finnhub API call and distinguish auth/network failures (which
    # must raise) from premium-gated or empty responses (which return None
    # so the yfinance backup can fill the gap).
    try:
        return fn()
    except FinnhubAPIException as exc:
        if exc.status_code == 401:
            raise RuntimeError(
                "Finnhub API key rejected (HTTP 401). "
                "Check FINNHUB_API_KEY in .env."
            ) from exc
        # HTTP 403 = premium endpoint not available on free tier.
        # HTTP 422 or other non-auth errors — treat as empty; yfinance fills in.
        return None
    except Exception as exc:
        raise RuntimeError(f"Finnhub network error: {exc}") from exc


# ---------------------------------------------------------------------------
# Field-level Finnhub extractors
# ---------------------------------------------------------------------------

def _next_earnings_finnhub(client: finnhub.Client, ticker: str) -> str | None:
    # Query the earnings calendar for the next scheduled date within 180 days.
    today = date.today()
    future = today + timedelta(days=180)
    data = _finnhub_call(lambda: client.earnings_calendar(
        _from=today.isoformat(),
        to=future.isoformat(),
        symbol=ticker,
        international=False,
    ))
    events = (data or {}).get("earningsCalendar") or []
    return events[0].get("date") if events else None


def _eps_estimate_finnhub(client: finnhub.Client, ticker: str) -> float | None:
    # Pull the most recent quarterly EPS estimate (premium on Finnhub free tier).
    data = _finnhub_call(lambda: client.company_eps_estimates(symbol=ticker, freq="quarterly"))
    estimates = (data or {}).get("data") or []
    val = estimates[0].get("epsAvg") if estimates else None
    return float(val) if val is not None else None


def _revenue_estimate_finnhub(client: finnhub.Client, ticker: str) -> float | None:
    # Pull the most recent quarterly revenue estimate in USD (premium on free tier).
    data = _finnhub_call(lambda: client.company_revenue_estimates(symbol=ticker, freq="quarterly"))
    estimates = (data or {}).get("data") or []
    val = estimates[0].get("revenueAvg") if estimates else None
    return float(val) if val is not None else None


def _analyst_count_finnhub(client: finnhub.Client, ticker: str) -> int | None:
    # Sum buy/hold/sell counts from the most recent recommendation trend entry.
    data = _finnhub_call(lambda: client.recommendation_trends(symbol=ticker))
    trends = data or []
    if not trends:
        return None
    latest = trends[0]
    total = sum(
        latest.get(k, 0) or 0
        for k in ("strongBuy", "buy", "hold", "sell", "strongSell")
    )
    return total if total > 0 else None


# ---------------------------------------------------------------------------
# yfinance backup — fills fields Finnhub free tier leaves empty
# ---------------------------------------------------------------------------

def _consensus_yfinance_backup(ticker: str) -> dict:
    # Attempt to populate the four consensus fields from yfinance when
    # Finnhub returns nothing. Best-effort: any field yfinance also lacks
    # stays None. Exceptions are swallowed so they never crash the main call.
    result = {
        "next_earnings_date": None,
        "eps_estimate":       None,
        "revenue_estimate":   None,
        "analyst_count":      None,
    }
    try:
        cal = yf.Ticker(ticker).calendar
        if not cal:
            return result

        # calendar is a dict; "Earnings Date" may be a list of Timestamps.
        earnings_dates = cal.get("Earnings Date")
        if earnings_dates:
            first = earnings_dates[0] if isinstance(earnings_dates, list) else earnings_dates
            if hasattr(first, "strftime"):
                result["next_earnings_date"] = first.strftime("%Y-%m-%d")
            else:
                result["next_earnings_date"] = str(first)

        eps = cal.get("EPS Estimate")
        if eps is not None:
            result["eps_estimate"] = float(eps)

        rev = cal.get("Revenue Estimate")
        if rev is not None:
            result["revenue_estimate"] = float(rev)

        opinions = cal.get("Number Of Analyst Opinions")
        if opinions is not None:
            result["analyst_count"] = int(opinions)

    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_consensus(ticker: str) -> dict:
    """Analyst consensus estimates for the given ticker.

    Tries Finnhub first. Falls back to yfinance for any field Finnhub leaves
    empty (free-tier limitation). Source is 'finnhub' when Finnhub covered at
    least one field, 'yfinance_backup' when only yfinance helped, or
    'finnhub_empty' when both returned nothing. Never fabricates data.

    Returns a dict with:
        ticker, next_earnings_date, eps_estimate, revenue_estimate,
        analyst_count, source, fetched_at (ISO-8601 UTC)
    """
    ticker = ticker.upper().strip()
    fetched_at = datetime.now(timezone.utc).isoformat()

    # Finnhub pass — each field is fetched independently so a premium-gated
    # endpoint failing doesn't block the others.
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)
    next_earnings_date = _next_earnings_finnhub(client, ticker)
    eps_estimate       = _eps_estimate_finnhub(client, ticker)
    revenue_estimate   = _revenue_estimate_finnhub(client, ticker)
    analyst_count      = _analyst_count_finnhub(client, ticker)

    # Track whether Finnhub provided anything before we merge the backup.
    finnhub_covered = any(
        v is not None
        for v in (next_earnings_date, eps_estimate, revenue_estimate, analyst_count)
    )

    # yfinance backup — merge any field still None after Finnhub.
    if not all(
        v is not None
        for v in (next_earnings_date, eps_estimate, revenue_estimate, analyst_count)
    ):
        backup = _consensus_yfinance_backup(ticker)
        next_earnings_date = next_earnings_date or backup["next_earnings_date"]
        eps_estimate       = eps_estimate       or backup["eps_estimate"]
        revenue_estimate   = revenue_estimate   or backup["revenue_estimate"]
        analyst_count      = analyst_count      or backup["analyst_count"]

    # Label the source so downstream code knows where the data came from.
    final_any = any(
        v is not None
        for v in (next_earnings_date, eps_estimate, revenue_estimate, analyst_count)
    )
    if finnhub_covered:
        source = "finnhub"
    elif final_any:
        source = "yfinance_backup"
    else:
        source = "finnhub_empty"

    return {
        "ticker":             ticker,
        "next_earnings_date": next_earnings_date,
        "eps_estimate":       eps_estimate,
        "revenue_estimate":   revenue_estimate,
        "analyst_count":      analyst_count,
        "source":             source,
        "fetched_at":         fetched_at,
    }


def get_basic_metrics(ticker: str) -> dict:
    """Current price, market cap, and trailing EPS from Finnhub.

    Always called regardless of analyst consensus coverage. Raises RuntimeError
    if Finnhub returns no data (bad ticker, auth failure, or network error).

    Returns a dict with:
        ticker, current_price, market_cap (USD), recent_eps, source, fetched_at
    """
    ticker = ticker.upper().strip()
    fetched_at = datetime.now(timezone.utc).isoformat()
    client = finnhub.Client(api_key=config.FINNHUB_API_KEY)

    # quote() returns current price ('c') and previous close ('pc').
    # An unrecognised ticker returns all zeros, so we check for that below.
    quote = _finnhub_call(lambda: client.quote(ticker))
    current_price = (quote or {}).get("c") or None

    # company_profile2() includes market cap in millions; convert to USD.
    profile = _finnhub_call(lambda: client.company_profile2(symbol=ticker))
    market_cap_m = (profile or {}).get("marketCapitalization")
    market_cap = market_cap_m * 1_000_000 if market_cap_m else None

    # company_basic_financials() annual EPS series: most recent entry is [0].
    bf = _finnhub_call(lambda: client.company_basic_financials(ticker, "all"))
    eps_series = ((bf or {}).get("series") or {}).get("annual", {}).get("eps", [])
    recent_eps = eps_series[0]["v"] if eps_series else None

    # A zero or missing price means Finnhub doesn't recognise the ticker.
    if current_price is None and market_cap is None:
        raise RuntimeError(
            f"Finnhub returned no data for '{ticker}'. "
            "Check the ticker symbol or confirm the stock is listed on a US exchange."
        )

    return {
        "ticker":        ticker,
        "current_price": current_price,
        "market_cap":    market_cap,
        "recent_eps":    recent_eps,
        "source":        "finnhub",
        "fetched_at":    fetched_at,
    }


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.tools.consensus [TICKER]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t = sys.argv[1].upper() if len(sys.argv) > 1 else "NFLX"

    print(f"Fetching consensus estimates for {t}...")
    c = get_consensus(t)
    print(f"\n--- get_consensus ---")
    print(f"Ticker:             {c['ticker']}")
    print(f"Next earnings date: {c['next_earnings_date']}")
    print(f"EPS estimate:       {c['eps_estimate']}")
    print(f"Revenue estimate:   {c['revenue_estimate']}")
    print(f"Analyst count:      {c['analyst_count']}")
    print(f"Source:             {c['source']}")
    print(f"Fetched at:         {c['fetched_at']}")

    print(f"\nFetching basic metrics for {t}...")
    m = get_basic_metrics(t)
    print(f"\n--- get_basic_metrics ---")
    print(f"Ticker:             {m['ticker']}")
    print(f"Current price:      {m['current_price']}")
    print(f"Market cap:         {m['market_cap']}")
    print(f"Trailing EPS:       {m['recent_eps']}")
    print(f"Source:             {m['source']}")
    print(f"Fetched at:         {m['fetched_at']}")
