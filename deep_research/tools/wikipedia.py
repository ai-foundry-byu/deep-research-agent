"""Wikipedia search and summary tool"""
import httpx
import urllib.parse


async def wikipedia_search(query: str) -> str:
    """Search Wikipedia and return article summaries for factual grounding"""
    try:
        headers = {"User-Agent": os.environ.get(
            "RESEARCH_USER_AGENT",
            "deep-research-agent/1.0 (https://github.com/jddavenportOpen/deep-research-agent)",
        )}
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            # First, search for matching articles
            search_response = await client.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": 3,
                    "format": "json",
                },
            )
            search_response.raise_for_status()
            search_data = search_response.json()

            results = search_data.get("query", {}).get("search", [])
            if not results:
                return "No Wikipedia articles found"

            # Fetch summaries for top results via the REST API
            formatted = []
            for result in results:
                title = result.get("title", "")
                encoded_title = urllib.parse.quote(title.replace(" ", "_"))

                try:
                    summary_response = await client.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}",
                    )
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()

                    extract = summary_data.get("extract", "No summary available")
                    if len(extract) > 500:
                        extract = extract[:500] + "..."

                    page_url = summary_data.get(
                        "content_urls", {}
                    ).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{encoded_title}")

                    formatted.append(
                        f"**{summary_data.get('title', title)}**\n"
                        f"{page_url}\n"
                        f"{extract}\n"
                    )
                except Exception:
                    # Fall back to basic info if summary API fails
                    import html
                    import re
                    snippet = re.sub(r"<[^>]+>", "", result.get("snippet", ""))
                    snippet = html.unescape(snippet)
                    formatted.append(
                        f"**{title}**\n"
                        f"https://en.wikipedia.org/wiki/{encoded_title}\n"
                        f"{snippet}\n"
                    )

            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"
