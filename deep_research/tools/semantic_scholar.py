"""Semantic Scholar academic paper search tool"""
import httpx


async def semantic_scholar_search(query: str, limit: int = 10) -> str:
    """Search Semantic Scholar for academic papers with citation data"""
    try:
        headers = {"User-Agent": "ResearchAgent/1.0 (academic research tool)"}
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,authors,year,citationCount,abstract,openAccessPdf",
                },
            )
            response.raise_for_status()
            data = response.json()

            formatted = []
            for paper in data.get("data", []):
                authors = ", ".join(
                    a.get("name", "Unknown") for a in (paper.get("authors") or [])[:3]
                )
                if len(paper.get("authors") or []) > 3:
                    authors += " et al."

                pdf_url = ""
                if paper.get("openAccessPdf"):
                    pdf_url = f"\nPDF: {paper['openAccessPdf'].get('url', '')}"

                abstract = paper.get("abstract") or "No abstract available"
                if len(abstract) > 400:
                    abstract = abstract[:400] + "..."

                formatted.append(
                    f"**{paper.get('title', 'No title')}**\n"
                    f"Authors: {authors}\n"
                    f"Year: {paper.get('year', 'N/A')} | Citations: {paper.get('citationCount', 0)}"
                    f"{pdf_url}\n"
                    f"{abstract}\n"
                )

            return "\n---\n".join(formatted) if formatted else "No results found"
    except Exception as e:
        return f"Error searching Semantic Scholar: {str(e)}"
