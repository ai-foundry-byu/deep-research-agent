"""Research agent subagents — profile-aware with cost tracking."""
import json
from ..models import query_llm, CostTracker


async def run_knowledge_gap_agent(
    query_text: str,
    findings_so_far: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> str:
    """Identify remaining knowledge gaps.  Returns gap list or 'COMPLETE'."""
    prompt = f"""You are a Knowledge Gap Analyst. Your job is to identify what we still don't know.

ORIGINAL RESEARCH QUERY: {query_text}

FINDINGS SO FAR:
{findings_so_far if findings_so_far else "No findings yet - this is the first iteration."}

Analyze the findings. What specific knowledge gaps remain? What questions are unanswered?
What would a thorough researcher still want to know?

If the research is comprehensive and no significant gaps remain, respond with exactly: COMPLETE

Otherwise, list each gap as a specific, searchable question. Format as:
1. [Specific question about gap 1]
2. [Specific question about gap 2]
3. [Specific question about gap 3]

Be specific and focused. Max 5 gaps."""

    try:
        return await query_llm(prompt, role="knowledge_gap", profile=profile, cost_tracker=cost_tracker)
    except Exception as e:
        print(f"   ⚠️  Knowledge Gap Agent error: {e}")
        return "COMPLETE"


async def run_tool_selector_agent(
    knowledge_gaps: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> list[dict]:
    """Select tools and queries for each knowledge gap."""
    prompt = f"""You are a Tool Selector. Given knowledge gaps, select the best tools and queries.

KNOWLEDGE GAPS:
{knowledge_gaps}

Available tools:
- brave_search: general web, news, current info (biased to the past month)
- github_search: code, repos, implementations
- reddit_search: CURRENT practitioner opinions & discussion threads (newest-first, trailing ~3 months)
- arxiv_search: academic papers
- hn_search: tech community consensus
- semantic_scholar_search: academic papers with citation counts and open-access PDFs
- stackoverflow_search: technical Q&A, practical solutions, code examples
- wikipedia_search: factual background, definitions, established knowledge
- twitter_search: CURRENT trending discussions & expert takes on X/Twitter (newest-first, trailing ~3 months)

RECENCY: For anything time-sensitive — trending topics, recent releases, "what's
new", current sentiment, fast-moving tools/products — strongly prefer
twitter_search and reddit_search (they fetch the latest posts) plus brave_search.
Use arxiv/semantic_scholar/wikipedia for evergreen/foundational knowledge only.

For each gap, select 1-2 tools MAX. Keep total under 10 queries.

Respond ONLY with valid JSON array:
[
  {{"gap": "gap description", "tool": "brave_search", "query": "specific search query"}},
  {{"gap": "gap description", "tool": "github_search", "query": "specific search query"}}
]"""

    try:
        result_text = await query_llm(prompt, role="tool_selector", profile=profile, cost_tracker=cost_tracker)

        start = result_text.find("[")
        end = result_text.rfind("]") + 1
        if start >= 0 and end > start:
            queries = json.loads(result_text[start:end])
            return queries[:10]
        else:
            print("   ⚠️  Tool Selector returned non-JSON, using defaults")
            return [{"gap": "general info", "tool": "brave_search", "query": knowledge_gaps.split('\n')[0][:100]}]
    except Exception as e:
        print(f"   ⚠️  Tool Selector error: {e}, using defaults")
        return [{"gap": "general info", "tool": "brave_search", "query": knowledge_gaps.split('\n')[0][:100]}]


async def run_observations_agent(
    raw_results: str,
    current_findings: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> str:
    """Extract structured observations from search results."""
    prompt = f"""Analyze new search results and extract key insights.

CURRENT FINDINGS:
{current_findings[:4000] if current_findings else "No prior findings."}

NEW SEARCH RESULTS:
{raw_results[:8000]}

What did you learn? Provide structured observations in 150 words or less.
Prefer academic and authoritative sources over blog posts and SEO content."""

    try:
        return await query_llm(prompt, role="observations", profile=profile, cost_tracker=cost_tracker)
    except Exception as e:
        print(f"   ⚠️  Observations Agent error: {e}")
        return f"Search results gathered ({len(raw_results)} chars). Analysis failed: {str(e)[:100]}"


async def run_devils_advocate_agent(
    all_findings: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> str:
    """Challenge findings and identify weaknesses."""
    prompt = f"""You are a Devil's Advocate. Challenge these research findings rigorously.

FINDINGS:
{all_findings[:5000]}

Identify:
1. Claims that may be wrong or unsubstantiated
2. Missing perspectives or counter-arguments
3. Sources that may be biased or low-quality
4. Gaps that would weaken the final report

Be specific. 200 words max."""

    try:
        return await query_llm(prompt, role="devils_advocate", profile=profile, cost_tracker=cost_tracker)
    except Exception as e:
        print(f"   ⚠️  Devil's Advocate error: {e}")
        return "Could not generate critique due to API error."


async def run_critique_gap_agent(
    critique: str,
    original_query: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> str:
    """Turn Devil's Advocate critique into actionable knowledge gaps for another research pass.
    Returns gap list or 'COMPLETE' if critique doesn't warrant more research."""
    prompt = f"""A Devil's Advocate has critiqued our research findings. Your job is to decide
if any of the criticisms warrant additional research.

ORIGINAL QUERY: {original_query}

CRITIQUE:
{critique}

If the critique raises valid points that need more research, list 1-3 specific searchable
questions that would address the most important gaps. Format:
1. [Question]
2. [Question]

If the critique is minor or the gaps are not addressable via search, respond: COMPLETE"""

    try:
        return await query_llm(prompt, role="knowledge_gap", profile=profile, cost_tracker=cost_tracker)
    except Exception as e:
        print(f"   ⚠️  Critique Gap Agent error: {e}")
        return "COMPLETE"


async def run_writer_agent(
    query: str,
    all_findings: str,
    observations: str,
    critique: str,
    sources: list[str],
    profile: str = "opus",
    cost_tracker: CostTracker = None,
    confidence_header: str = "",
    triangulation_text: str = "",
    citation_text: str = "",
) -> str:
    """Synthesize all findings into a final report.

    Phase 3 additions: confidence score header, cross-source triangulation
    results, and citation highlights are injected into the writer prompt.
    """
    sources_text = "\n".join(f"- {s}" for s in sources[:30])

    # Build Phase 3 sections for the prompt
    phase3_sections = ""
    if confidence_header:
        phase3_sections += f"\nCONFIDENCE ASSESSMENT:\n{confidence_header}\n"
    if triangulation_text:
        phase3_sections += f"\nCROSS-SOURCE TRIANGULATION:\n{triangulation_text}\n"
    if citation_text:
        phase3_sections += f"\nCITATION HIGHLIGHTS:\n{citation_text}\n"

    prompt = f"""Write a comprehensive research report.

QUERY: {query}

FINDINGS:
{all_findings[:6000]}

KEY OBSERVATIONS:
{observations[:4000]}

CRITIQUE TO ADDRESS:
{critique[:500]}
{phase3_sections}
SOURCES CONSULTED:
{sources_text}

Write a well-structured markdown report with:
- Confidence score header (if provided above, include it verbatim at the top of the report)
- Executive summary (3-4 sentences)
- Key findings (detailed bullet points with source attribution where possible)
- Cross-source verification (if triangulation data is provided, include a section showing which claims are well-supported vs. single-source)
- Citation highlights (if citation data is provided, mention seminal and high-impact papers)
- Analysis (synthesize across sources, note consensus vs disagreement)
- Limitations (what we couldn't verify, potential biases, low-confidence claims)
- Recommendations (actionable next steps)

For each substantive claim, indicate confidence level inline: [high confidence], [medium confidence], or [low confidence] based on the triangulation data.

Prioritize academic and authoritative sources over blog posts. Note where sources conflict."""

    try:
        return await query_llm(
            prompt, role="writer", profile=profile, cost_tracker=cost_tracker, max_tokens=8192
        )
    except Exception as e:
        print(f"   ⚠️  Writer Agent error: {e}")
        fallback = f"""# Research Report: {query}

"""
        if confidence_header:
            fallback += f"{confidence_header}\n\n"
        fallback += f"""## Executive Summary
Research completed with {len(sources)} sources consulted. Full analysis unavailable due to API error.

## Key Findings
{all_findings[:1500]}

## Limitations
Report generation encountered API errors. See raw findings above.

## Sources Consulted
{len(sources)} sources across multiple platforms.
"""
        return fallback
