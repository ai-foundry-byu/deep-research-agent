"""Stack Overflow search tool via Stack Exchange API"""
import httpx
import html


async def stackoverflow_search(query: str, limit: int = 5) -> str:
    """Search Stack Overflow for questions and answers"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search for relevant questions
            response = await client.get(
                "https://api.stackexchange.com/2.3/search/advanced",
                params={
                    "order": "desc",
                    "sort": "votes",
                    "q": query,
                    "site": "stackoverflow",
                    "pagesize": limit,
                    "filter": "withbody",
                },
            )
            response.raise_for_status()
            data = response.json()

            formatted = []
            for item in data.get("items", []):
                title = html.unescape(item.get("title", "No title"))
                score = item.get("score", 0)
                is_answered = item.get("is_answered", False)
                answer_count = item.get("answer_count", 0)
                link = item.get("link", "")
                tags = ", ".join(item.get("tags", [])[:5])

                # Extract a snippet from the body (HTML), strip tags roughly
                body = item.get("body", "")
                # Simple HTML tag removal for snippet
                import re
                body_text = re.sub(r"<[^>]+>", " ", body)
                body_text = html.unescape(body_text).strip()
                if len(body_text) > 300:
                    body_text = body_text[:300] + "..."

                status = "Answered" if is_answered else f"{answer_count} answers"

                formatted.append(
                    f"**{title}** ({score} votes, {status})\n"
                    f"{link}\n"
                    f"Tags: {tags}\n"
                    f"{body_text}\n"
                )

            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching Stack Overflow: {str(e)}"
