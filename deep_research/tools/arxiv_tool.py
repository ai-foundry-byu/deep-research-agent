"""arXiv paper search tool"""
import httpx

async def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for academic papers"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # Parse XML response (simple text extraction)
            text = response.text
            entries = text.split("<entry>")[1:]  # Skip header
            
            formatted = []
            for entry in entries[:max_results]:
                try:
                    title = entry.split("<title>")[1].split("</title>")[0].strip()
                    summary = entry.split("<summary>")[1].split("</summary>")[0].strip()[:300]
                    link = entry.split("<id>")[1].split("</id>")[0].strip()
                    published = entry.split("<published>")[1].split("</published>")[0].strip()[:10]
                    
                    formatted.append(
                        f"**{title}**\n"
                        f"{link}\n"
                        f"Published: {published}\n"
                        f"{summary}...\n"
                    )
                except:
                    continue
            
            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"
