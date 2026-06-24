"""Brave Search tool"""
import httpx
import os

async def brave_search(query: str, count: int = 5, freshness: str = "") -> str:
    """Search the web via Brave Search API.

    Args:
        query: search keywords
        count: number of results to request
        freshness: Brave recency filter — pd (past day), pw (past week),
            pm (past month), py (past year), or "" for no time filter.
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return "Error: BRAVE_API_KEY not set"

    try:
        params = {"q": query, "count": count}
        if freshness:
            params["freshness"] = freshness
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": api_key},
                params=params
            )
            response.raise_for_status()
            results = response.json()

            formatted = []
            for r in results.get("web", {}).get("results", []):
                age = r.get("age", "")
                age_label = f" ({age})" if age else ""
                formatted.append(
                    f"**{r.get('title', 'No title')}**{age_label}\n"
                    f"{r.get('url', '')}\n"
                    f"{r.get('description', 'No description')}\n"
                )

            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching Brave: {str(e)}"
