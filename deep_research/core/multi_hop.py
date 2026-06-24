"""Multi-hop reasoning — decompose complex queries into sub-questions and research in parallel.

When the knowledge-gap analysis identifies questions that cannot be answered by current
sources, this module decomposes them into independent sub-queries, researches each in
parallel, and synthesizes the results back into the main research thread.

Phase 3: Advanced Intelligence.
"""
import asyncio
import json
from dataclasses import dataclass, field
from ..models import query_llm, CostTracker


@dataclass
class SubQueryResult:
    """Result from researching a single sub-question."""
    question: str
    findings: str
    sources: list[str] = field(default_factory=list)
    success: bool = True


@dataclass
class MultiHopResult:
    """Combined result from multi-hop reasoning."""
    sub_queries: list[str]
    sub_results: list[SubQueryResult]
    synthesis: str
    total_sub_queries: int = 0

    @property
    def successful_results(self) -> list[SubQueryResult]:
        return [r for r in self.sub_results if r.success]


async def decompose_query(
    query: str,
    knowledge_gaps: str,
    existing_findings: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
    max_sub_queries: int = 4,
) -> list[str]:
    """Decompose a complex query into 2-4 independent sub-questions.

    Uses the knowledge gaps and existing findings to identify what aspects
    of the query need deeper investigation via separate research threads.
    """
    prompt = f"""You are a Research Decomposition Agent. Your job is to break a complex
research query into independent sub-questions that can be researched separately.

ORIGINAL QUERY: {query}

KNOWLEDGE GAPS (from gap analysis):
{knowledge_gaps[:2000]}

EXISTING FINDINGS (what we already know):
{existing_findings[:2000] if existing_findings else "No findings yet."}

Decompose the unresolved gaps into 2-{max_sub_queries} independent sub-questions.
Each sub-question should:
- Be specific and searchable
- Address a distinct aspect of the original query
- NOT overlap significantly with other sub-questions
- NOT repeat what we already know from existing findings

If the query is already simple enough or gaps are minor, return fewer sub-questions.
If no decomposition is needed, return an empty list.

Respond ONLY with a JSON array of strings:
["Sub-question 1", "Sub-question 2", "Sub-question 3"]

Return an empty array [] if decomposition is not needed."""

    try:
        result_text = await query_llm(
            prompt, role="knowledge_gap", profile=profile, cost_tracker=cost_tracker,
        )
        start = result_text.find("[")
        end = result_text.rfind("]") + 1
        if start >= 0 and end > start:
            sub_queries = json.loads(result_text[start:end])
            # Validate: must be list of strings
            sub_queries = [q for q in sub_queries if isinstance(q, str) and len(q.strip()) > 10]
            return sub_queries[:max_sub_queries]
        return []
    except Exception as e:
        print(f"   Warning: Query decomposition failed ({e})")
        return []


async def research_sub_query(
    sub_query: str,
    tool_selector_fn,
    execute_tools_fn,
    observations_fn,
    config,
    cost_tracker: CostTracker = None,
) -> SubQueryResult:
    """Research a single sub-question using the standard tool pipeline.

    This is a lightweight version of one research iteration — it selects tools,
    executes them, and extracts observations, but does not run the full
    knowledge-gap loop.

    Args:
        sub_query: the sub-question to research
        tool_selector_fn: async callable for tool selection (run_tool_selector_agent)
        execute_tools_fn: async callable for tool execution (execute_tools)
        observations_fn: async callable for observation extraction (run_observations_agent)
        config: ResearchConfig instance
        cost_tracker: optional cost tracker
    """
    profile = config.profile

    try:
        # 1. Select tools for this sub-query
        tool_selections = await tool_selector_fn(
            sub_query, profile=profile, cost_tracker=cost_tracker,
        )
        if not tool_selections:
            return SubQueryResult(
                question=sub_query, findings="No tools selected.", success=False,
            )

        # 2. Execute tools
        raw_results, sources = await execute_tools_fn(tool_selections, config)

        # 3. Extract observations
        observations = await observations_fn(
            raw_results, "", profile=profile, cost_tracker=cost_tracker,
        )

        return SubQueryResult(
            question=sub_query,
            findings=observations,
            sources=sources,
            success=True,
        )
    except Exception as e:
        print(f"   Warning: Sub-query research failed for '{sub_query[:60]}': {e}")
        return SubQueryResult(
            question=sub_query, findings=f"Research failed: {e}", success=False,
        )


async def run_parallel_sub_queries(
    sub_queries: list[str],
    tool_selector_fn,
    execute_tools_fn,
    observations_fn,
    config,
    cost_tracker: CostTracker = None,
) -> list[SubQueryResult]:
    """Research multiple sub-queries in parallel using asyncio.

    Each sub-query gets its own tool selection, execution, and observation
    extraction pipeline running concurrently.
    """
    tasks = [
        research_sub_query(
            sq, tool_selector_fn, execute_tools_fn, observations_fn,
            config, cost_tracker,
        )
        for sq in sub_queries
    ]
    return await asyncio.gather(*tasks)


async def synthesize_sub_results(
    original_query: str,
    sub_results: list[SubQueryResult],
    existing_findings: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> str:
    """Synthesize sub-query results into a coherent addition to the main findings.

    Merges the independently-researched sub-question answers into a unified
    narrative that addresses the original query.
    """
    successful = [r for r in sub_results if r.success]
    if not successful:
        return ""

    sub_results_text = "\n\n".join(
        f"### Sub-question: {r.question}\n{r.findings}"
        for r in successful
    )

    prompt = f"""You are a Research Synthesis Agent. Multiple sub-questions were researched
independently. Your job is to synthesize these results into coherent findings.

ORIGINAL QUERY: {original_query}

EXISTING FINDINGS:
{existing_findings[:2000] if existing_findings else "None yet."}

SUB-QUESTION RESULTS:
{sub_results_text[:4000]}

Synthesize these sub-results into a unified set of findings that:
1. Eliminates redundancy between sub-results
2. Highlights connections between different sub-topics
3. Notes any contradictions between sub-results
4. Adds context from the original query

Write 200-400 words of synthesized findings. Focus on new information not in existing findings."""

    try:
        return await query_llm(
            prompt, role="observations", profile=profile, cost_tracker=cost_tracker,
        )
    except Exception as e:
        # Fallback: concatenate raw sub-results
        print(f"   Warning: Synthesis failed ({e}), using raw sub-results")
        return sub_results_text


async def run_multi_hop(
    query: str,
    knowledge_gaps: str,
    existing_findings: str,
    tool_selector_fn,
    execute_tools_fn,
    observations_fn,
    config,
    cost_tracker: CostTracker = None,
    max_sub_queries: int = 4,
) -> MultiHopResult:
    """Full multi-hop reasoning pipeline.

    1. Decompose query into sub-questions based on knowledge gaps
    2. Research each sub-question in parallel
    3. Synthesize results back into the main thread

    Returns a MultiHopResult with all sub-query results and synthesis.
    """
    profile = config.profile

    # Step 1: Decompose
    print("   🔀 Decomposing into sub-queries...")
    sub_queries = await decompose_query(
        query, knowledge_gaps, existing_findings,
        profile=profile, cost_tracker=cost_tracker,
        max_sub_queries=max_sub_queries,
    )

    if not sub_queries:
        print("   🔀 No sub-queries needed")
        return MultiHopResult(sub_queries=[], sub_results=[], synthesis="", total_sub_queries=0)

    print(f"   🔀 {len(sub_queries)} sub-queries:")
    for i, sq in enumerate(sub_queries):
        print(f"      {i+1}. {sq[:80]}")

    # Step 2: Research in parallel
    print(f"   🔀 Researching {len(sub_queries)} sub-queries in parallel...")
    sub_results = await run_parallel_sub_queries(
        sub_queries, tool_selector_fn, execute_tools_fn, observations_fn,
        config, cost_tracker,
    )

    successful = sum(1 for r in sub_results if r.success)
    print(f"   🔀 {successful}/{len(sub_queries)} sub-queries completed successfully")

    # Step 3: Synthesize
    if successful > 0:
        print("   🔀 Synthesizing sub-query results...")
        synthesis = await synthesize_sub_results(
            query, sub_results, existing_findings,
            profile=profile, cost_tracker=cost_tracker,
        )
    else:
        synthesis = ""

    return MultiHopResult(
        sub_queries=sub_queries,
        sub_results=sub_results,
        synthesis=synthesis,
        total_sub_queries=len(sub_queries),
    )
