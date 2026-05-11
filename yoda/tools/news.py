"""News search for a given query using the Tavily API.

The one public function, search_news(), returns a list of citation-ready
result dicts. Each result includes the article title, URL, publication date,
a short snippet, and the domain name (for citations).
"""

import sys
from urllib.parse import urlparse

from tavily import TavilyClient

from yoda import config


def search_news(query: str, max_results: int = 5) -> list[dict]:
    """Search for recent news articles matching the given query.

    Uses the Tavily API (LLM-friendly search). Raises RuntimeError on
    network or auth failures. An empty results list is a valid return value
    when no articles match the query.

    Returns a list of dicts, each with:
        title, url, published_date (ISO date string or None),
        snippet (up to 500 chars), source (domain name)
    """
    # Build the Tavily client from the key stored in config.
    client = TavilyClient(api_key=config.TAVILY_API_KEY)

    # Issue the search. search_depth="basic" avoids crawling full page content,
    # keeping latency low and staying within Tavily's free-tier rate limits.
    try:
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_raw_content=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Tavily search failed: {exc}") from exc

    results = response.get("results") or []

    # Map each Tavily result to our standard citation-ready dict shape.
    output = []
    for item in results:
        url = item.get("url") or ""
        # Extract the bare domain from the URL (strip www. for clean citations).
        domain = urlparse(url).netloc.removeprefix("www.")
        # Tavily's "content" field is the article snippet; cap at 500 chars.
        snippet = (item.get("content") or "")[:500]

        output.append({
            "title":          item.get("title") or "",
            "url":            url,
            "published_date": item.get("published_date"),  # ISO string or None
            "snippet":        snippet,
            "source":         domain,
        })

    return output


# ---------------------------------------------------------------------------
# Smoke test — run with: python -m yoda.tools.news
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    query = "Netflix subscriber growth Q1 2026"
    print(f'Searching: "{query}"\n')

    results = search_news(query)
    if not results:
        print("No results returned.")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['title']} — {r['source']} — {r['published_date']}")
        print(f"    {r['url']}")
        print(f"    {r['snippet'][:200]}")
        print()
