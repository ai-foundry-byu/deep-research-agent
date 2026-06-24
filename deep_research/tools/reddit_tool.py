"""Reddit search tool.

Reddit shut down unauthenticated access to its JSON endpoints — the old
`https://www.reddit.com/r/<sub>/search.json` now returns HTTP 403 for every
User-Agent (verified 2026-06-07). So this tool has two paths:

1. OAuth (preferred, true recency): if REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET
   are set, do app-only (client_credentials) OAuth and hit oauth.reddit.com
   with sort=new + a time filter. Gives genuinely current posts.
   Create a free "script" app at https://www.reddit.com/prefs/apps to get creds.

2. Brave fallback (zero-cred, works today): search `site:reddit.com <query>`
   via Brave with a recency window. Lower fidelity (Brave's Reddit index lags
   and is sparser) but needs no Reddit credentials.

Both return the same formatted-string shape so the research loop is agnostic.
"""
import os
import httpx

# Reddit requires a descriptive User-Agent. Set REDDIT_USER_AGENT to your own
# (Reddit recommends "<platform>:<app id>:<version> (by /u/<your-username>)").
_UA = os.environ.get(
    "REDDIT_USER_AGENT", "deep-research-agent/1.0 (by /u/your_username)"
)
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH_BASE = "https://oauth.reddit.com"


async def reddit_search(
    query: str,
    subreddit: str = "all",
    limit: int = 10,
    freshness: str = "pw",
    time_filter: str = "week",
    days: int | None = None,
) -> str:
    """Search Reddit for recent posts.

    Args:
        query: search keywords
        subreddit: subreddit to scope to ("all" = sitewide)
        limit: max results
        freshness: Brave recency window for the fallback path (pd/pw/pm/py/"")
        time_filter: Reddit OAuth 't' param for the OAuth path
            (hour/day/week/month/year/all). Ignored when `days` is set.
        days: trailing-window in days (e.g. 90 = past ~3 months). When set,
            the OAuth path picks the smallest 't' that covers the window,
            fetches newest-first, and client-side trims to exactly this window
            so the result is "most-recent items within the last N days."
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")

    if client_id and client_secret:
        try:
            return await _reddit_via_oauth(
                query, subreddit, limit, time_filter, client_id, client_secret, days
            )
        except Exception as e:
            # Fall through to Brave on any OAuth failure rather than hard-failing.
            brave = await _reddit_via_brave(query, subreddit, limit, freshness)
            return f"(Reddit OAuth failed: {str(e)[:120]} — fell back to Brave)\n\n{brave}"

    return await _reddit_via_brave(query, subreddit, limit, freshness)


def _t_for_days(days: int | None) -> str:
    """Smallest Reddit 't' bucket that fully covers a day-window."""
    if not days:
        return ""
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    if days <= 366:
        return "year"
    return "all"


async def _reddit_via_oauth(
    query: str,
    subreddit: str,
    limit: int,
    time_filter: str,
    client_id: str,
    client_secret: str,
    days: int | None = None,
) -> str:
    """True-recency path via Reddit app-only OAuth (sort=new + time filter).

    When `days` is set, widen 't' to cover the window and over-fetch so the
    client-side window-trim still leaves ~`limit` of the newest posts.
    """
    import time

    if days:
        t_param = _t_for_days(days)
        fetch_limit = min(100, max(limit * 3, 50))  # over-fetch, then trim to window
        cutoff = time.time() - days * 86400
    else:
        t_param = time_filter
        fetch_limit = limit
        cutoff = None

    async with httpx.AsyncClient(timeout=30.0) as client:
        tok_resp = await client.post(
            _TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(client_id, client_secret),
            headers={"User-Agent": _UA},
        )
        tok_resp.raise_for_status()
        token = tok_resp.json()["access_token"]

        sub = subreddit or "all"
        params = {
            "q": query,
            "limit": fetch_limit,
            # sort=relevance, NOT new: Reddit's search ignores `q` under sort=new
            # (returns the global new-feed) — verified 2026-06-08. relevance honors
            # the query; we re-sort by recency client-side for "most-recent items".
            "sort": "relevance",
            "t": t_param,
            "raw_json": 1,
        }
        if sub != "all":
            params["restrict_sr"] = "true"  # only meaningful when scoped to a subreddit
        resp = await client.get(
            f"{_OAUTH_BASE}/r/{sub}/search",
            params=params,
            headers={"Authorization": f"bearer {token}", "User-Agent": _UA},
        )
        resp.raise_for_status()
        data = resp.json()

        children = [c.get("data", {}) for c in data.get("data", {}).get("children", [])]
        if cutoff is not None:
            children = [p for p in children if (p.get("created_utc") or 0) >= cutoff]
        # sort=relevance doesn't order by date — sort newest-first within the matches
        children.sort(key=lambda p: p.get("created_utc") or 0, reverse=True)
        children = children[:limit]
        if not children:
            return "No results found"
        return "\n---\n".join(_format_oauth_post(p) for p in children)


def _format_oauth_post(p: dict) -> str:
    import datetime
    created = p.get("created_utc", 0)
    date = (
        datetime.datetime.utcfromtimestamp(created).strftime("%Y-%m-%d")
        if created else "unknown date"
    )
    body = (p.get("selftext", "") or "")[:200]
    body = f"{body}..." if body else ""
    return (
        f"**{p.get('title', 'No title')}** ({p.get('score', 0)} upvotes, {p.get('num_comments', 0)} comments) [{date}]\n"
        f"r/{p.get('subreddit', 'unknown')} - u/{p.get('author', 'unknown')}\n"
        f"https://reddit.com{p.get('permalink', '')}\n"
        f"{body}"
    )


async def _reddit_via_brave(query: str, subreddit: str, limit: int, freshness: str) -> str:
    """Zero-cred fallback: Reddit results via Brave site filter + recency window."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return (
            "Error: Reddit unavailable — set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET "
            "(reddit.com/prefs/apps) for OAuth, or BRAVE_API_KEY for the fallback path."
        )

    if subreddit and subreddit not in ("all", ""):
        site = f"site:reddit.com/r/{subreddit}"
    else:
        site = "site:reddit.com"
    site_query = f"{site} {query}"

    try:
        params = {"q": site_query, "count": limit}
        if freshness:
            params["freshness"] = freshness
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params=params,
            )
            resp.raise_for_status()
            results = resp.json()

        web_results = results.get("web", {}).get("results", [])
        if not web_results:
            return "No Reddit results found"

        formatted = []
        for r in web_results:
            age = r.get("age", "")
            age_label = f" ({age})" if age else ""
            formatted.append(
                f"**{r.get('title', 'No title')}**{age_label}\n"
                f"{r.get('url', '')}\n"
                f"{r.get('description', '')}"
            )
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Error searching Reddit (Brave fallback): {str(e)}"
