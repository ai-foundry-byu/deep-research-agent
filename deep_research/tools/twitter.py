"""X/Twitter search tool — twitterapi.io (real tweets) with a Brave fallback.

The official X API killed its free read tier (pay-per-use only as of Feb 2026),
x.com returns HTTP 402 to scrapers, and Nitter is dead. So real, dated tweets
come from twitterapi.io — a third-party read API at ~$0.15/1k tweets
(~$0.03 for a 200-tweet run). Set TWITTERAPI_IO_KEY in the env to enable it.

Path selection:
  1. twitterapi.io (preferred): if TWITTERAPI_IO_KEY is set, hit the advanced
     search endpoint for real tweets with author, exact date, text, engagement,
     and a precise `since_time:` day-window. This is the high-fidelity path.
  2. Brave fallback (zero-cred): site:x.com search via Brave. Sparse, undated,
     low fidelity — used only when no twitterapi.io key is present.

Both return the same formatted-string shape so the research loop is agnostic.
Get a key (free starter credits, no card) at https://twitterapi.io.
"""
import httpx
import os
import re

_APIIO_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"


async def twitter_search(
    query: str,
    count: int = 10,
    freshness: str = "pw",
    days: int | None = None,
) -> str:
    """Search X/Twitter for real, dated posts.

    Args:
        query: search keywords (twitterapi.io also accepts operators like
            from:user, OR, since_time:/until_time:)
        count: max tweets to return
        freshness: Brave recency window for the fallback path (pd/pw/pm/py/"")
        days: trailing-window in days for the twitterapi.io path (e.g. 90 =
            past ~3 months). Translated to a `since_time:` epoch operator and
            paginated newest-first.

    Returns:
        Formatted tweet results with author, exact date, text, URL, engagement.
    """
    apiio_key = os.environ.get("TWITTERAPI_IO_KEY")
    if apiio_key:
        try:
            return await _twitter_via_apiio(query, count, days, apiio_key)
        except Exception as e:
            brave = await _twitter_via_brave(query, count, freshness)
            return f"(twitterapi.io failed: {str(e)[:120]} — fell back to Brave)\n\n{brave}"

    return await _twitter_via_brave(query, count, freshness)


async def _twitter_via_apiio(query: str, count: int, days: int | None, api_key: str) -> str:
    """High-fidelity path: real tweets via twitterapi.io advanced search.

    Contract (verified 2026-06-08):
      GET https://api.twitterapi.io/twitter/tweet/advanced_search
      header: x-api-key
      params: query, queryType=Latest|Top, cursor
      resp:   {tweets:[{id,text,url,createdAt,author:{userName},
                        likeCount,retweetCount}], has_next_page, next_cursor}
      ~20 tweets/page; $0.15/1k tweets.
    """
    import time

    q = query
    if days:
        since_epoch = int(time.time() - days * 86400)
        if "since_time:" not in q:
            q = f"{q} since_time:{since_epoch}"

    tweets = []
    cursor = ""
    pages = 0
    max_pages = min(10, max(1, (count // 20) + 1))  # ~20/page; bound the spend

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(tweets) < count and pages < max_pages:
            params = {"query": q, "queryType": "Latest"}
            if cursor:
                params["cursor"] = cursor
            resp = await client.get(_APIIO_URL, headers={"x-api-key": api_key}, params=params)
            resp.raise_for_status()
            data = resp.json()
            tweets.extend(data.get("tweets", []) or [])
            pages += 1
            if not data.get("has_next_page"):
                break
            cursor = data.get("next_cursor", "") or ""
            if not cursor:
                break

    tweets = tweets[:count]
    if not tweets:
        return "No X/Twitter results found"

    formatted = []
    for t in tweets:
        author = (t.get("author") or {}).get("userName", "")
        date = _normalize_date(t.get("createdAt", ""))
        likes = t.get("likeCount", 0)
        rts = t.get("retweetCount", 0)
        url = t.get("url", "") or (f"https://x.com/{author}/status/{t.get('id','')}" if author else "")
        text = (t.get("text", "") or "").replace("\n", " ").strip()
        entry = f"**@{author}** [{date}] ({likes} likes, {rts} reposts)\n{url}\n{text}"
        formatted.append(entry)
    return "\n\n---\n\n".join(formatted)


def _normalize_date(raw: str) -> str:
    """Normalize twitterapi.io createdAt to YYYY-MM-DD; pass through on failure."""
    if not raw:
        return "unknown date"
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d")  # "Tue Dec 10 07:00:00 +0000 2024"
    except Exception:
        pass
    import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return raw[:10]


async def _twitter_via_brave(query: str, count: int = 10, freshness: str = "pw") -> str:
    """Zero-cred fallback: X results via Brave site filter (sparse, undated)."""
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return (
            "Error: X/Twitter unavailable — set TWITTERAPI_IO_KEY (twitterapi.io, "
            "real tweets ~$0.15/1k) for the high-fidelity path, or BRAVE_API_KEY "
            "for the sparse fallback."
        )

    # Search both x.com and twitter.com domains
    site_query = f"(site:x.com OR site:twitter.com) {query}"

    async def _fetch(fresh: str):
        params = {"q": site_query, "count": count}
        if fresh:
            params["freshness"] = fresh
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params=params,
            )
            response.raise_for_status()
            return response.json().get("web", {}).get("results", [])

    try:
        web_results = await _fetch(freshness)
        # Brave's x.com index is sparse; a tight window can yield nothing.
        # Degrade gracefully to the unfiltered query rather than returning empty.
        if not web_results and freshness:
            web_results = await _fetch("")
        if not web_results:
            return "No X/Twitter results found"

        formatted = []
        for r in web_results:
            url = r.get("url", "")
            title = r.get("title", "No title")
            description = r.get("description", "")
            age = r.get("age", "")

            # Extract author from URL pattern: x.com/<username>/status/...
            author = _extract_author(url)
            # Extract engagement hints from description if available
            engagement = _extract_engagement(description)

            entry = f"**@{author}**" if author else f"**{title}**"
            if age:
                entry += f" ({age})"
            entry += f"\n{url}"
            entry += f"\n{description}"
            if engagement:
                entry += f"\nEngagement: {engagement}"

            formatted.append(entry)

        return "\n\n---\n\n".join(formatted)

    except Exception as e:
        return f"Error searching X/Twitter: {str(e)}"


def _extract_author(url: str) -> str:
    """Extract username from x.com or twitter.com URL."""
    match = re.search(r'(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)(?:/|$)', url)
    if match:
        username = match.group(1)
        # Filter out non-user paths
        non_users = {"search", "explore", "hashtag", "i", "intent", "home", "settings"}
        if username.lower() not in non_users:
            return username
    return ""


def _extract_engagement(description: str) -> str:
    """Try to extract engagement metrics from search snippet."""
    metrics = []

    # Look for common patterns in search snippets
    like_match = re.search(r'(\d+(?:\.\d+)?[KMkm]?)\s*(?:likes?|hearts?)', description, re.IGNORECASE)
    rt_match = re.search(r'(\d+(?:\.\d+)?[KMkm]?)\s*(?:retweets?|reposts?|RTs?)', description, re.IGNORECASE)
    reply_match = re.search(r'(\d+(?:\.\d+)?[KMkm]?)\s*(?:replies|comments?)', description, re.IGNORECASE)

    if like_match:
        metrics.append(f"{like_match.group(1)} likes")
    if rt_match:
        metrics.append(f"{rt_match.group(1)} reposts")
    if reply_match:
        metrics.append(f"{reply_match.group(1)} replies")

    return ", ".join(metrics) if metrics else ""
