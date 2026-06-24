"""Citation graph analysis — track source citations, identify seminal papers,
and find citation clusters.

Uses Semantic Scholar's citation graph API (free, no auth required) to:
1. Track which sources cite which other sources
2. Identify highly-cited (seminal) papers
3. Find recent papers that build on key papers (citation velocity)
4. Surface citation clusters around the research topic

Phase 3: Advanced Intelligence.
"""
import asyncio
import re
from dataclasses import dataclass, field
from typing import Optional
import httpx


# Semantic Scholar API configuration
S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"
S2_TIMEOUT = 20.0
S2_HEADERS = {"User-Agent": "ResearchAgent/1.0 (citation-graph)"}

# Rate limiting: S2 free tier allows ~100 requests/5 minutes
S2_REQUEST_DELAY = 0.5  # seconds between requests


@dataclass
class CitedPaper:
    """A paper in the citation graph."""
    paper_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    citation_count: int = 0
    citation_velocity: float = 0.0  # citations per year
    abstract: str = ""
    url: str = ""
    is_seminal: bool = False  # high citation count relative to field

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": self.authors[:3],
            "year": self.year,
            "citation_count": self.citation_count,
            "citation_velocity": round(self.citation_velocity, 1),
            "is_seminal": self.is_seminal,
            "url": self.url,
        }


@dataclass
class CitationEdge:
    """A citation relationship between two papers."""
    citing_paper_id: str
    citing_title: str
    cited_paper_id: str
    cited_title: str


@dataclass
class CitationGraphResult:
    """Full citation graph analysis output."""
    papers: list[CitedPaper] = field(default_factory=list)
    edges: list[CitationEdge] = field(default_factory=list)
    seminal_papers: list[CitedPaper] = field(default_factory=list)
    recent_high_impact: list[CitedPaper] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "total_papers": len(self.papers),
            "seminal_papers": [p.to_dict() for p in self.seminal_papers],
            "recent_high_impact": [p.to_dict() for p in self.recent_high_impact],
            "citation_edges": len(self.edges),
        }

    def format_for_report(self) -> str:
        """Format citation graph results for inclusion in the final report."""
        if not self.papers:
            return ""

        sections = []

        if self.seminal_papers:
            lines = ["### Seminal Papers (highly cited)"]
            for p in self.seminal_papers[:5]:
                authors_str = ", ".join(p.authors[:3])
                if len(p.authors) > 3:
                    authors_str += " et al."
                lines.append(
                    f"- **{p.title}** ({p.year}) — {p.citation_count:,} citations "
                    f"({p.citation_velocity:.0f}/year) — {authors_str}"
                )
            sections.append("\n".join(lines))

        if self.recent_high_impact:
            lines = ["### Recent High-Impact Papers (high citation velocity)"]
            for p in self.recent_high_impact[:5]:
                authors_str = ", ".join(p.authors[:3])
                if len(p.authors) > 3:
                    authors_str += " et al."
                lines.append(
                    f"- **{p.title}** ({p.year}) — {p.citation_count:,} citations "
                    f"({p.citation_velocity:.0f}/year) — {authors_str}"
                )
            sections.append("\n".join(lines))

        if self.edges:
            # Show citation relationships
            lines = [f"### Citation Network ({len(self.edges)} connections found)"]
            # Group by cited paper
            cited_groups: dict[str, list[str]] = {}
            for edge in self.edges:
                key = edge.cited_title
                if key not in cited_groups:
                    cited_groups[key] = []
                cited_groups[key].append(edge.citing_title)

            for cited, citers in sorted(cited_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                lines.append(f"- **{cited}** cited by: {', '.join(citers[:3])}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


async def build_citation_graph(
    query: str,
    all_findings: str,
    sources: list[str],
    max_papers: int = 10,
    max_citations_per_paper: int = 5,
) -> CitationGraphResult:
    """Build a citation graph for the research topic.

    1. Search Semantic Scholar for key papers related to the query
    2. For each paper, fetch its citations and references
    3. Identify seminal papers and recent high-impact work
    4. Return structured citation graph data

    Args:
        query: the research query
        all_findings: accumulated findings (used to extract paper titles)
        sources: list of source labels from the research run
        max_papers: maximum number of seed papers to analyze
        max_citations_per_paper: max "cited by" papers to fetch per seed

    Returns:
        CitationGraphResult with papers, edges, and analysis
    """
    result = CitationGraphResult()

    try:
        # Step 1: Find seed papers via Semantic Scholar search
        seed_papers = await _search_papers(query, limit=max_papers)
        if not seed_papers:
            result.summary = "No papers found for citation analysis"
            return result

        # Also try to find papers mentioned in findings
        mentioned_titles = _extract_paper_titles(all_findings)
        if mentioned_titles:
            extra_papers = await _search_papers_by_titles(mentioned_titles[:3])
            # Merge without duplicates
            seen_ids = {p.paper_id for p in seed_papers}
            for p in extra_papers:
                if p.paper_id not in seen_ids:
                    seed_papers.append(p)
                    seen_ids.add(p.paper_id)

        # Step 2: Fetch citation data for each seed paper
        all_papers: dict[str, CitedPaper] = {}
        edges: list[CitationEdge] = []

        for paper in seed_papers[:max_papers]:
            all_papers[paper.paper_id] = paper

            # Fetch "cited by" papers (who cites this paper)
            citers = await _get_citations(paper.paper_id, limit=max_citations_per_paper)
            for citer in citers:
                all_papers.setdefault(citer.paper_id, citer)
                edges.append(CitationEdge(
                    citing_paper_id=citer.paper_id,
                    citing_title=citer.title,
                    cited_paper_id=paper.paper_id,
                    cited_title=paper.title,
                ))

            await asyncio.sleep(S2_REQUEST_DELAY)  # rate limiting

        # Step 3: Compute citation velocity and identify seminal papers
        import datetime
        current_year = datetime.datetime.now().year

        for paper in all_papers.values():
            if paper.year and paper.year > 0:
                years_since = max(1, current_year - paper.year)
                paper.citation_velocity = paper.citation_count / years_since

        # Seminal: high total citations (top 20% or >= 100 citations)
        sorted_by_citations = sorted(all_papers.values(), key=lambda p: p.citation_count, reverse=True)
        citation_threshold = max(100, sorted_by_citations[0].citation_count * 0.2) if sorted_by_citations else 100
        for paper in sorted_by_citations:
            if paper.citation_count >= citation_threshold:
                paper.is_seminal = True

        seminal = [p for p in sorted_by_citations if p.is_seminal][:5]

        # Recent high-impact: published in last 3 years with high velocity
        recent = [
            p for p in all_papers.values()
            if p.year and p.year >= current_year - 3 and p.citation_velocity >= 10
        ]
        recent.sort(key=lambda p: p.citation_velocity, reverse=True)

        result.papers = list(all_papers.values())
        result.edges = edges
        result.seminal_papers = seminal
        result.recent_high_impact = recent[:5]
        result.summary = (
            f"Analyzed {len(all_papers)} papers, {len(edges)} citation links. "
            f"{len(seminal)} seminal papers, {len(recent)} recent high-impact papers."
        )

    except Exception as e:
        result.summary = f"Citation graph analysis failed: {e}"
        print(f"   Warning: Citation graph failed: {e}")

    return result


async def _search_papers(query: str, limit: int = 10) -> list[CitedPaper]:
    """Search Semantic Scholar for papers matching the query."""
    try:
        async with httpx.AsyncClient(timeout=S2_TIMEOUT, headers=S2_HEADERS) as client:
            response = await client.get(
                f"{S2_BASE_URL}/paper/search",
                params={
                    "query": query,
                    "limit": limit,
                    "fields": "title,authors,year,citationCount,abstract,url",
                },
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for raw in data.get("data", []):
                if not raw.get("paperId"):
                    continue
                authors = [a.get("name", "") for a in (raw.get("authors") or [])[:5]]
                papers.append(CitedPaper(
                    paper_id=raw["paperId"],
                    title=raw.get("title", "Unknown"),
                    authors=authors,
                    year=raw.get("year"),
                    citation_count=raw.get("citationCount", 0),
                    abstract=(raw.get("abstract") or "")[:300],
                    url=raw.get("url", ""),
                ))
            return papers
    except Exception as e:
        print(f"   Warning: S2 search failed: {e}")
        return []


async def _search_papers_by_titles(titles: list[str]) -> list[CitedPaper]:
    """Search for specific papers by title."""
    papers = []
    for title in titles:
        found = await _search_papers(title, limit=1)
        papers.extend(found)
        await asyncio.sleep(S2_REQUEST_DELAY)
    return papers


async def _get_citations(paper_id: str, limit: int = 5) -> list[CitedPaper]:
    """Get papers that cite the given paper (cited-by)."""
    try:
        async with httpx.AsyncClient(timeout=S2_TIMEOUT, headers=S2_HEADERS) as client:
            response = await client.get(
                f"{S2_BASE_URL}/paper/{paper_id}/citations",
                params={
                    "limit": limit,
                    "fields": "title,authors,year,citationCount,url",
                },
            )
            response.raise_for_status()
            data = response.json()

            papers = []
            for entry in data.get("data", []):
                raw = entry.get("citingPaper", {})
                if not raw.get("paperId"):
                    continue
                authors = [a.get("name", "") for a in (raw.get("authors") or [])[:5]]
                papers.append(CitedPaper(
                    paper_id=raw["paperId"],
                    title=raw.get("title", "Unknown"),
                    authors=authors,
                    year=raw.get("year"),
                    citation_count=raw.get("citationCount", 0),
                    url=raw.get("url", ""),
                ))
            return papers
    except Exception as e:
        print(f"   Warning: S2 citations fetch failed for {paper_id}: {e}")
        return []


def _extract_paper_titles(findings: str) -> list[str]:
    """Extract potential paper titles from research findings.

    Looks for patterns like **Title** or "Title" that appear to be academic paper names.
    """
    titles = []

    # Match bold text that looks like a paper title (4+ words)
    bold_pattern = re.findall(r'\*\*([^*]{20,120})\*\*', findings)
    for match in bold_pattern:
        # Filter out non-title patterns
        if any(kw in match.lower() for kw in ['findings', 'iteration', 'summary', 'analysis', 'results from']):
            continue
        titles.append(match)

    # Match quoted titles
    quoted_pattern = re.findall(r'"([^"]{20,120})"', findings)
    for match in quoted_pattern:
        if any(kw in match.lower() for kw in ['findings', 'iteration', 'summary']):
            continue
        titles.append(match)

    return titles[:5]  # limit to avoid excessive API calls
