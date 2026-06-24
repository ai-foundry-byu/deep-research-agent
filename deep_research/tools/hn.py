"""Hacker News Algolia search tool"""
import httpx

async def hn_search(query: str, count: int = 5) -> str:
    """Search Hacker News via Algolia API"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = "https://hn.algolia.com/api/v1/search"
            params = {"query": query, "tags": "story", "hitsPerPage": count}
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            formatted = []
            for hit in data.get("hits", []):
                formatted.append(
                    f"**{hit.get('title', 'No title')}** ({hit.get('points', 0)} points)\n"
                    f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}\n"
                    f"By: {hit.get('author', 'unknown')}, {hit.get('num_comments', 0)} comments\n"
                    f"{hit.get('url', 'No URL')}\n"
                )
            
            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching HN: {str(e)}"
