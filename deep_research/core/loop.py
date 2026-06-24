"""Iterative knowledge gap research loop — profile-aware with cost tracking, critique loop,
multi-hop reasoning, cross-source triangulation, citation graph, and confidence scoring.

Phase 4: Production Polish — progress streaming, caching, format options.
Phase 2 Checkpointing: saves state after each iteration; resumes on restart.
"""
import asyncio
import hashlib
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import time

from ..vendor.checkpoint import CheckpointManager

_ckpt_logger = logging.getLogger(__name__ + ".checkpoint")

from .agents import (
    run_knowledge_gap_agent,
    run_tool_selector_agent,
    run_observations_agent,
    run_devils_advocate_agent,
    run_critique_gap_agent,
    run_writer_agent,
)
from .source_curator import score_sources
from .context_compressor import compress_context, get_embeddings, cosine_similarity
from .multi_hop import run_multi_hop
from .triangulator import triangulate
from .citations import build_citation_graph
from .confidence import compute_confidence
from .verifier import verify_report, annotate_report
from .progress import ProgressReporter
from .cache import ResearchCache
from .formatter import format_report, VALID_FORMATS
from ..tools.brave import brave_search
from ..tools.github_tool import github_search
from ..tools.reddit_tool import reddit_search
from ..tools.arxiv_tool import arxiv_search
from ..tools.hn import hn_search
from ..tools.firecrawl_tool import firecrawl_extract
from ..tools.obsidian import save_to_obsidian
from ..tools.semantic_scholar import semantic_scholar_search
from ..tools.stackoverflow import stackoverflow_search
from ..tools.wikipedia import wikipedia_search
from ..tools.twitter import twitter_search
from ..config import ResearchConfig
from ..models import CostTracker


def _research_task_id(query: str, depth: str) -> str:
    """Deterministic checkpoint task_id from query + depth."""
    h = hashlib.sha256(f"{query}::{depth}".encode()).hexdigest()[:16]
    return f"research-{h}"


def _sanitize_for_checkpoint(text: str, max_chars: int = 50_000) -> str:
    """Strip oversized raw content before checkpointing."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[... truncated for checkpoint ...]"
    return text


@dataclass
class ResearchResult:
    report: str
    obsidian_path: str
    iteration_count: int
    sources_consulted: list[str]
    confidence_level: str
    findings_history: list[str]
    profile: str = "opus"
    cost_summary: dict = field(default_factory=dict)


# Tool dispatcher
TOOL_MAP = {
    "brave_search": brave_search,
    "github_search": github_search,
    "reddit_search": reddit_search,
    "arxiv_search": arxiv_search,
    "hn_search": hn_search,
    "firecrawl_extract": firecrawl_extract,
    "semantic_scholar_search": semantic_scholar_search,
    "stackoverflow_search": stackoverflow_search,
    "wikipedia_search": wikipedia_search,
    "twitter_search": twitter_search,
}

# Auto-scrape limits per depth preset
AUTO_SCRAPE_LIMITS = {
    "shallow": 1,
    "standard": 3,
    "deep": 5,
}


def make_refute_search(config: ResearchConfig):
    """Build the refutation-search adapter the verifier injects into each voter.

    Reuses the researcher's OWN live tooling so per-claim verification can pull
    first-party evidence from the same surfaces the report drew on — general web
    (Brave) PLUS X/Twitter and Reddit. Biased toward surfacing contradicting
    evidence: we phrase the X/Reddit queries skeptically and run all three in
    parallel, returning the combined text for the voter to adjudicate.
    """
    async def refute_search(claim: str, query: str) -> str:
        claim_q = claim[:200]
        searches = [
            brave_search(claim_q, count=4, freshness=config.web_freshness),
            twitter_search(
                claim_q, count=config.twitter_results,
                freshness=config.social_freshness, days=config.social_lookback_days,
            ),
            reddit_search(
                claim_q, limit=config.reddit_results,
                freshness=config.social_freshness,
                time_filter=config.reddit_time_filter, days=config.social_lookback_days,
            ),
        ]
        labels = ["web (Brave)", "X/Twitter", "Reddit"]
        results = await asyncio.gather(*searches, return_exceptions=True)
        chunks = []
        for label, res in zip(labels, results):
            if isinstance(res, Exception):
                chunks.append(f"=== {label}: search error: {str(res)[:120]} ===")
            else:
                chunks.append(f"=== Evidence from {label} ===\n{res}")
        return "\n\n".join(chunks)

    return refute_search


async def execute_tools(tool_selections: list[dict], config: ResearchConfig) -> tuple[str, list[str]]:
    """Execute all selected tools in parallel, then auto-scrape top Brave URLs."""
    tasks = []
    sources = []
    brave_indices = []  # Track which results are from brave_search

    for selection in tool_selections:
        tool_name = selection.get("tool")
        query = selection.get("query")

        if tool_name not in TOOL_MAP:
            continue

        tool_func = TOOL_MAP[tool_name]

        if tool_name == "reddit_search" and "r/" in query:
            parts = query.split("r/")
            if len(parts) > 1:
                subreddit = parts[1].split()[0]
                actual_query = query.replace(f"r/{subreddit}", "").strip()
                tasks.append(tool_func(actual_query, subreddit=subreddit, limit=config.reddit_results,
                                       freshness=config.social_freshness, time_filter=config.reddit_time_filter,
                                       days=config.social_lookback_days))
            else:
                tasks.append(tool_func(query, limit=config.reddit_results,
                                       freshness=config.social_freshness, time_filter=config.reddit_time_filter,
                                       days=config.social_lookback_days))
        elif tool_name == "brave_search":
            tasks.append(tool_func(query, count=config.brave_results_per_query, freshness=config.web_freshness))
            brave_indices.append(len(tasks) - 1)
        elif tool_name == "github_search":
            tasks.append(tool_func(query, count=config.github_results_per_query))
        elif tool_name == "arxiv_search":
            tasks.append(tool_func(query, max_results=config.arxiv_max_results))
        elif tool_name == "hn_search":
            tasks.append(tool_func(query, count=config.hn_results_per_query))
        elif tool_name == "semantic_scholar_search":
            tasks.append(tool_func(query, limit=config.semantic_scholar_results))
        elif tool_name == "stackoverflow_search":
            tasks.append(tool_func(query, limit=config.stackoverflow_results))
        elif tool_name == "twitter_search":
            tasks.append(tool_func(query, count=config.twitter_results, freshness=config.social_freshness,
                                   days=config.social_lookback_days))
        elif tool_name == "firecrawl_extract":
            tasks.append(tool_func(query))
        else:
            tasks.append(tool_func(query))

        sources.append(f"{tool_name}: {query}")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    combined = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            combined.append(f"Error from {tool_selections[i]['tool']}: {result}")
        else:
            combined.append(f"=== Results from {tool_selections[i]['tool']} ===\n{result}")

    # Auto-scrape top URLs from Brave results via Firecrawl
    if config.auto_scrape_limit > 0 and brave_indices:
        urls_to_scrape = []
        for idx in brave_indices:
            if idx < len(results) and not isinstance(results[idx], Exception):
                urls_to_scrape.extend(_extract_urls_from_brave(str(results[idx])))

        urls_to_scrape = urls_to_scrape[:config.auto_scrape_limit]
        if urls_to_scrape:
            print(f"   🔗 Auto-scraping {len(urls_to_scrape)} top Brave URLs...")
            scrape_tasks = [firecrawl_extract(url) for url in urls_to_scrape]
            scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

            for i, scrape_result in enumerate(scrape_results):
                if isinstance(scrape_result, Exception):
                    combined.append(f"Error auto-scraping {urls_to_scrape[i]}: {scrape_result}")
                else:
                    combined.append(f"=== Auto-scraped: {urls_to_scrape[i]} ===\n{scrape_result}")
                    sources.append(f"firecrawl_extract (auto): {urls_to_scrape[i]}")

    return "\n\n".join(combined), sources


def _extract_urls_from_brave(brave_result: str) -> list[str]:
    """Extract URLs from Brave search result text."""
    import re
    # Match HTTP(S) URLs on their own lines (the Brave tool puts URLs on separate lines)
    urls = re.findall(r'https?://[^\s\n]+', brave_result)
    # Filter out common non-content URLs
    filtered = []
    skip_domains = {'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 'x.com'}
    for url in urls:
        domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
        if domain_match:
            domain = domain_match.group(1)
            if not any(skip in domain for skip in skip_domains):
                filtered.append(url)
    return filtered


def _deduplicate_results(raw_results: str, all_sources: list[str]) -> tuple[str, int]:
    """Remove duplicate source blocks that reference the same URL.

    If multiple tool results contain the same URL, keep the first (usually
    richer) block and drop subsequent duplicates. Also deduplicates the
    all_sources list in place.

    Returns:
        (deduplicated_results, count_of_removed_duplicates)
    """
    import re

    # Split into blocks by source headers
    block_pattern = r'(===\s*(?:Results from|Auto-scraped:)\s*.+?\s*===)'
    parts = re.split(block_pattern, raw_results)

    # Reassemble into (header, content) pairs
    blocks = []
    i = 0
    while i < len(parts):
        if re.match(block_pattern, parts[i].strip()):
            header = parts[i]
            content = parts[i + 1] if i + 1 < len(parts) else ""
            blocks.append((header, content))
            i += 2
        else:
            if parts[i].strip():
                blocks.append(("", parts[i]))
            i += 1

    # Extract all URLs from each block and track seen URLs
    seen_urls: set[str] = set()
    dedup_blocks = []
    removed = 0

    for header, content in blocks:
        block_text = header + content
        urls_in_block = set(re.findall(r'https?://[^\s\n\)]+', block_text))

        # Normalize URLs (strip trailing punctuation, fragments)
        normalized = set()
        for url in urls_in_block:
            url = url.rstrip(".,;:!?)")
            # Strip fragment
            url = url.split("#")[0]
            normalized.add(url)

        # Check if all URLs in this block have been seen before
        new_urls = normalized - seen_urls
        if normalized and not new_urls:
            # Entirely duplicate block
            removed += 1
            continue

        seen_urls.update(normalized)
        dedup_blocks.append(f"{header}{content}")

    # Deduplicate the sources list (preserve order)
    seen_source_keys: set[str] = set()
    unique_sources = []
    for s in all_sources:
        # Normalize: extract the query/url part after the tool name
        key = s.strip().lower()
        if key not in seen_source_keys:
            seen_source_keys.add(key)
            unique_sources.append(s)

    all_sources.clear()
    all_sources.extend(unique_sources)

    return "\n\n".join(dedup_blocks), removed


async def _research_iteration(
    iteration: int,
    max_iter: int,
    query: str,
    all_findings: str,
    all_observations: list,
    all_sources: list,
    config: ResearchConfig,
    cost_tracker: CostTracker,
    gaps_override: str = None,
    all_source_scores: list = None,
) -> tuple[str, bool]:
    """Run one research iteration.  Returns (updated_findings, is_complete).

    Phase 3: When multi_hop is enabled and the gap analysis identifies complex
    sub-questions, decomposes them into parallel sub-queries.
    """
    profile = config.profile
    if all_source_scores is None:
        all_source_scores = []

    print(f"   1️⃣  Analyzing knowledge gaps...")
    if gaps_override:
        gaps = gaps_override
    else:
        gaps = await run_knowledge_gap_agent(query, all_findings, profile=profile, cost_tracker=cost_tracker)

    if "COMPLETE" in gaps:
        print("   ✅ No significant gaps remain")
        return all_findings, True

    print(f"   📋 Gaps:\n{gaps[:200]}...")

    # Phase 3: Multi-hop reasoning — decompose and research sub-queries in parallel
    if config.multi_hop and not gaps_override:
        multi_hop_result = await run_multi_hop(
            query=query,
            knowledge_gaps=gaps,
            existing_findings=all_findings,
            tool_selector_fn=run_tool_selector_agent,
            execute_tools_fn=execute_tools,
            observations_fn=run_observations_agent,
            config=config,
            cost_tracker=cost_tracker,
            max_sub_queries=config.max_sub_queries,
        )
        if multi_hop_result.synthesis:
            # Merge sub-query sources and findings into the main thread
            for sr in multi_hop_result.successful_results:
                all_sources.extend(sr.sources)
            all_observations.append(
                f"### Iteration {iteration + 1} (Multi-Hop)\n{multi_hop_result.synthesis}"
            )
            all_findings += (
                f"\n\n### Iteration {iteration + 1} Multi-Hop Findings\n"
                f"{multi_hop_result.synthesis}\n"
            )
            print(f"   🔀 Multi-hop added {len(multi_hop_result.successful_results)} sub-query results")
            # Don't return early — continue with the standard iteration
            # to fill any remaining gaps not addressed by sub-queries

    # Tool selection
    print("   2️⃣  Selecting tools...")
    tool_selections = await run_tool_selector_agent(gaps, profile=profile, cost_tracker=cost_tracker)
    if not tool_selections:
        print("   ⚠️  No tools selected")
        return all_findings, True
    print(f"   🔧 {len(tool_selections)} queries")

    # Execute
    print("   3️⃣  Executing tools...")
    raw_results, sources = await execute_tools(tool_selections, config)
    all_sources.extend(sources)
    print(f"   📦 {len(sources)} results")

    # Deduplicate sources: merge results from the same URL
    raw_results, dedup_count = _deduplicate_results(raw_results, all_sources)
    if dedup_count > 0:
        print(f"   🔀 Deduplicated {dedup_count} duplicate source(s)")

    # Source credibility scoring (Phase 2)
    if config.source_scoring:
        print("   📊 Scoring source credibility...")
        raw_results, scores = await score_sources(
            raw_results, query, sources,
            profile=profile,
            cost_tracker=cost_tracker,
            threshold=config.source_score_threshold,
        )
        if scores:
            high = sum(1 for s in scores if s.is_high_quality)
            print(f"   📊 {high}/{len(scores)} sources scored as high quality")
            all_source_scores.extend(scores)

    # Context compression (Phase 2)
    if config.context_compression:
        print("   🗜️  Compressing context...")
        compressed_results = await compress_context(
            raw_results, query,
            max_chars=config.compressed_context_chars,
        )
        compressed_findings = await compress_context(
            all_findings, query,
            max_chars=config.compressed_findings_chars,
        ) if all_findings else all_findings
        print(f"   🗜️  {len(raw_results)} -> {len(compressed_results)} chars (results), "
              f"{len(all_findings)} -> {len(compressed_findings)} chars (findings)")
    else:
        compressed_results = raw_results
        compressed_findings = all_findings

    # Observe
    print("   4️⃣  Extracting observations...")
    observations = await run_observations_agent(
        compressed_results, compressed_findings,
        profile=profile, cost_tracker=cost_tracker,
    )
    all_observations.append(f"### Iteration {iteration + 1}\n{observations}")

    all_findings += f"\n\n### Iteration {iteration + 1} Findings\n{observations}\n\nRaw Results:\n{raw_results[:1000]}...\n"
    print("   💡 Observations recorded\n")

    return all_findings, False


async def run_research(
    query: str,
    max_iterations: Optional[int] = None,
    max_time_minutes: Optional[int] = None,
    depth: str = "standard",
    profile: str = "opus",
    resume: bool = False,
    progress: Optional[ProgressReporter] = None,
    use_cache: bool = True,
    output_format: str = "markdown",
    output_path: Optional[str] = None,
    verify_claims: Optional[bool] = None,
) -> ResearchResult:
    """Run the iterative knowledge gap research loop.

    Phase 4 additions:
    - progress: ProgressReporter for Telegram streaming updates
    - use_cache: check/save results in SQLite cache (skip with --no-cache)
    - output_format: markdown|html|pdf|json
    - output_path: custom output file path

    Phase 2 Checkpointing:
    - Saves state after each research iteration
    - When *resume* is True, restores from the most recent checkpoint
    """

    config = ResearchConfig().for_depth(depth).for_profile(profile)
    if max_iterations is not None:
        config.max_iterations = max_iterations
    if max_time_minutes is not None:
        config.max_time_minutes = max_time_minutes
    # CLI --verify-claims / --no-verify-claims overrides the depth gate.
    if verify_claims is not None:
        config.claim_verification = verify_claims

    # Phase 4: Setup progress reporter
    if progress is None:
        progress = ProgressReporter(enabled=False)
    progress.set_query(query)

    # Phase 4: Check cache
    cache = ResearchCache()
    if use_cache:
        cached = cache.get_cached(query, depth)
        if cached:
            print(f"\n📦 Cache hit! ({cached.age_human})")
            await progress.update(f"Cache hit ({cached.age_human}), returning cached result", emoji="📦", force=True)
            cached_data = cached.results_json
            return ResearchResult(
                report=cached_data.get("report", ""),
                obsidian_path=cached_data.get("obsidian_path", cached.report_path),
                iteration_count=cached_data.get("iteration_count", 0),
                sources_consulted=cached_data.get("sources_consulted", []),
                confidence_level=cached_data.get("confidence_level", "unknown"),
                findings_history=cached_data.get("findings_history", []),
                profile=cached_data.get("profile", profile),
                cost_summary=cached_data.get("cost_summary", {"total_cost_usd": 0, "note": "cached"}),
            )

    cost_tracker = CostTracker()
    start_time = time.time()
    max_seconds = config.max_time_minutes * 60

    # ── Checkpoint setup ────────────────────────────────────────────────
    ckpt_mgr = CheckpointManager()
    task_id = _research_task_id(query, depth)
    start_iteration = 0

    all_findings = ""
    all_observations: list[str] = []
    all_sources: list[str] = []
    all_source_scores: list = []  # Phase 3: collect credibility scores across iterations
    iteration_count = 0

    # Try to resume from checkpoint
    if resume:
        saved, resumed_from = ckpt_mgr.resume_or_start(
            task_id,
            initial_state={},
            stage="iteration",
        )
        if resumed_from:
            start_iteration = saved.get("iteration_count", 0)
            all_findings = saved.get("all_findings", "")
            all_observations = saved.get("all_observations", [])
            all_sources = saved.get("all_sources", [])
            all_source_scores = saved.get("all_source_scores", [])
            iteration_count = start_iteration
            _ckpt_logger.info(
                "Resumed research '%s' from iteration %d (checkpoint %s)",
                task_id, start_iteration, resumed_from,
            )
            print(f"♻️  Resumed from checkpoint — skipping to iteration {start_iteration + 1}")

    profile_label = "🟣 Opus Premium" if profile == "opus" else "🟢 Multi-Model Budget"
    print(f"\n🔎 Starting research: {query}")
    print(f"   Profile: {profile_label}")
    print(f"   Depth: {depth}, Max iterations: {config.max_iterations}, Max time: {config.max_time_minutes}m")
    if config.multi_hop:
        print(f"   Multi-hop: enabled (max {config.max_sub_queries} sub-queries)")
    print()

    await progress.update(
        f"Starting research ({depth} depth, {config.max_iterations} max iterations)",
        emoji="🔎", force=True,
    )

    # ── Main research loop ──────────────────────────────────────────────
    for iteration in range(start_iteration, config.max_iterations):
        iteration_count = iteration + 1
        elapsed = time.time() - start_time

        print(f"📍 Iteration {iteration + 1}/{config.max_iterations} (elapsed: {int(elapsed)}s)")

        if elapsed > max_seconds:
            print(f"⏰ Time limit reached ({config.max_time_minutes}m)")
            await progress.update("Time limit reached", emoji="⏰")
            break

        await progress.update(
            f"Iteration {iteration + 1}/{config.max_iterations} — searching sources...",
            emoji="🔍",
        )

        all_findings, is_complete = await _research_iteration(
            iteration, config.max_iterations, query,
            all_findings, all_observations, all_sources,
            config, cost_tracker,
            all_source_scores=all_source_scores,
        )

        await progress.update(
            f"Iteration {iteration + 1} complete — {len(all_sources)} sources so far",
            emoji="📚",
        )

        # ── Checkpoint after each iteration ─────────────────────────────
        try:
            ckpt_mgr.save(
                task_id,
                stage="iteration",
                state_dict={
                    "query": query,
                    "depth": depth,
                    "profile": profile,
                    "iteration_count": iteration_count,
                    "all_findings": _sanitize_for_checkpoint(all_findings),
                    "all_observations": all_observations,
                    "all_sources": all_sources,
                    "all_source_scores": [
                        s.__dict__ if hasattr(s, "__dict__") else str(s)
                        for s in all_source_scores
                    ],
                },
                agent_name="researcher",
                description=f"After iteration {iteration_count}/{config.max_iterations}",
            )
            _ckpt_logger.info("Checkpointed iteration %d for %s", iteration_count, task_id)
        except Exception as ckpt_err:
            _ckpt_logger.warning("Checkpoint save failed (non-fatal): %s", ckpt_err)

        if is_complete:
            break

    # ── Post-loop: Devil's Advocate + optional critique loop ────────────
    print("\n📊 Research loop complete. Running final analysis...\n")
    await progress.update("Running final analysis (critique, triangulation)...", emoji="🔬")

    print("   5️⃣  Running Devil's Advocate...")
    critique = await run_devils_advocate_agent(all_findings, profile=profile, cost_tracker=cost_tracker)

    # Critique loop — feed critique back as new gaps
    if config.critique_loop:
        for ci in range(config.max_critique_iterations):
            elapsed = time.time() - start_time
            if elapsed > max_seconds:
                break

            print(f"   🔄 Critique loop {ci + 1}/{config.max_critique_iterations}...")
            critique_gaps = await run_critique_gap_agent(
                critique, query, profile=profile, cost_tracker=cost_tracker
            )

            if "COMPLETE" in critique_gaps:
                print("   ✅ Critique addressed — no further research needed")
                break

            # Run one more research pass targeting critique gaps
            iteration_count += 1
            print(f"📍 Critique iteration {ci + 1} (elapsed: {int(elapsed)}s)")
            all_findings, _ = await _research_iteration(
                iteration_count - 1, config.max_iterations + ci, query,
                all_findings, all_observations, all_sources,
                config, cost_tracker,
                gaps_override=critique_gaps,
                all_source_scores=all_source_scores,
            )

    # ── Phase 3: Post-loop analysis ────────────────────────────────────
    # Run triangulation, citation graph, and confidence scoring in parallel
    # where possible, then feed results into the writer agent.

    triangulation_result = None
    citation_result = None
    confidence_result = None

    # Step A: Triangulation + Citation graph can run in parallel
    print("   🔺 Running cross-source triangulation...")
    print("   📚 Building citation graph...")
    await progress.update("Cross-source triangulation + citation analysis...", emoji="🔺")

    phase3_tasks = [
        triangulate(
            all_findings, all_sources, query,
            profile=profile, cost_tracker=cost_tracker,
        ),
        build_citation_graph(
            query, all_findings, all_sources,
            max_papers=config.citation_max_papers,
            max_citations_per_paper=config.citation_max_per_paper,
        ),
    ]
    triangulation_result, citation_result = await asyncio.gather(*phase3_tasks)

    if triangulation_result and triangulation_result.claims:
        print(f"   🔺 {triangulation_result.summary}")
    if citation_result and citation_result.papers:
        print(f"   📚 {citation_result.summary}")

    # Step B: Confidence scoring (depends on triangulation)
    print("   📈 Computing confidence score...")
    confidence_result = await compute_confidence(
        query=query,
        all_findings=all_findings,
        sources=all_sources,
        triangulation=triangulation_result,
        source_scores=all_source_scores if all_source_scores else None,
        profile=profile,
        cost_tracker=cost_tracker,
    )
    print(f"   📈 Confidence: {confidence_result.total}/100 ({confidence_result.label})")

    # Format Phase 3 outputs for the writer
    confidence_header = confidence_result.format_for_report() if confidence_result else ""
    triangulation_text = triangulation_result.format_for_report() if triangulation_result else ""
    citation_text = citation_result.format_for_report() if citation_result else ""

    # ── Final report ────────────────────────────────────────────────────
    print("   6️⃣  Writing final report...")
    await progress.update("Writing final report...", emoji="✍️")
    report = await run_writer_agent(
        query=query,
        all_findings=all_findings,
        observations="\n\n".join(all_observations),
        critique=critique,
        sources=all_sources,
        profile=profile,
        cost_tracker=cost_tracker,
        confidence_header=confidence_header,
        triangulation_text=triangulation_text,
        citation_text=citation_text,
    )

    # ── Per-claim adversarial verification (deep-research graft) ────────────
    # Extract falsifiable claims from the synthesized draft, then run an N-vote
    # refute-or-survive check on each (reusing live web + X/Reddit tooling).
    # Annotate the report with a Claim Verification table + inline flags.
    # Depth-gated via config; skipped on stub/error reports.
    _stub_markers_for_verify = (
        "Full analysis unavailable due to API error",
        "Report generation encountered API errors",
    )
    _report_is_stub = any(m in report for m in _stub_markers_for_verify)
    verification_result = None
    if config.claim_verification and not _report_is_stub:
        print("   🛡️  Adversarial claim verification...")
        await progress.update("Adversarially verifying claims...", emoji="🛡️")
        try:
            verification_result = await verify_report(
                report_markdown=report,
                query=query,
                refute_search=make_refute_search(config),
                profile=profile,
                cost_tracker=cost_tracker,
                votes_per_claim=config.claim_verify_votes,
                max_claims=config.claim_verify_max_claims,
            )
            if verification_result and verification_result.claims:
                print(f"   🛡️  {verification_result.summary()}")
                report = annotate_report(report, verification_result)
            else:
                print("   🛡️  No verifiable claims extracted")
        except Exception as verify_err:
            print(f"   ⚠️  Claim verification failed (non-fatal): {verify_err}")
    elif _report_is_stub:
        print("   🛡️  Skipping claim verification (stub report)")

    # Phase 4: Format conversion (if not markdown)
    if output_format != "markdown":
        print(f"   📄 Converting to {output_format}...")
        metadata = {
            "profile": profile,
            "depth": depth,
            "iterations": iteration_count,
            "sources_count": len(all_sources),
            "confidence": confidence_result.label if confidence_result else "unknown",
            "cost_usd": cost_tracker.summary().get("total_cost_usd", 0),
        }
        formatted = format_report(
            report_markdown=report,
            format=output_format,
            query=query,
            metadata=metadata,
            output_path=output_path,
        )
        if formatted.file_path:
            print(f"   📄 {output_format.upper()} saved: {formatted.file_path}")

    # Save to Obsidian (always save markdown to vault)
    print("   7️⃣  Saving to Obsidian...")
    confidence_level = confidence_result.label if confidence_result else "moderate"
    cost_summary = cost_tracker.summary()
    obsidian_path = await save_to_obsidian(
        query=query,
        report=report,
        iteration_count=iteration_count,
        sources_consulted=all_sources,
        confidence_level=confidence_level,
        vault_path=config.output_dir,
        profile=profile,
        cost_summary=cost_summary,
    )

    # Phase 4: Save to cache — but ONLY if the writer actually produced a real
    # synthesis. Stub-fallback reports (writer crashed, report generation errored)
    # must not be cached, otherwise a re-run within TTL silently returns the same
    # broken output. Caught 2026-05-08 on CR-2 LinkedIn growth research.
    _stub_markers = (
        "Full analysis unavailable due to API error",
        "Report generation encountered API errors",
    )
    _is_stub = any(m in report for m in _stub_markers)
    if use_cache and not _is_stub:
        cache.save_cache(
            query=query,
            depth=depth,
            results={
                "report": report,
                "obsidian_path": obsidian_path,
                "iteration_count": iteration_count,
                "sources_consulted": all_sources[:50],
                "confidence_level": confidence_level,
                "findings_history": [all_findings[:10000]],
                "profile": profile,
                "cost_summary": cost_summary,
            },
            report_path=obsidian_path,
        )
        print("   💾 Results cached")
    elif use_cache and _is_stub:
        print("   ⚠️  Stub report detected (writer errored) — NOT caching")

    # ── Clean up checkpoint on successful completion ──────────────────
    try:
        removed = ckpt_mgr.delete(task_id)
        if removed:
            _ckpt_logger.info("Cleaned up %d checkpoint(s) for completed research %s", removed, task_id)
    except Exception as ckpt_err:
        _ckpt_logger.warning("Checkpoint cleanup failed (non-fatal): %s", ckpt_err)

    print(f"\n✅ Research complete!")
    print(f"   Profile: {profile_label}")
    print(f"   Iterations: {iteration_count}")
    print(f"   Sources: {len(all_sources)}")
    print(f"   Confidence: {confidence_result.total}/100 ({confidence_level})")
    print(f"   Cost: ${cost_summary['total_cost_usd']:.4f}")
    print(f"   Saved to: {obsidian_path}\n")

    # Phase 4: Telegram completion notification
    await progress.flush()
    await progress.complete(
        summary=(
            f"📊 {iteration_count} iterations, {len(all_sources)} sources\n"
            f"🎯 Confidence: {confidence_result.total}/100 ({confidence_level})"
        ),
        report_path=obsidian_path,
        cost_usd=cost_summary.get("total_cost_usd", 0),
    )

    return ResearchResult(
        report=report,
        obsidian_path=obsidian_path,
        iteration_count=iteration_count,
        sources_consulted=all_sources,
        confidence_level=confidence_level,
        findings_history=[all_findings],
        profile=profile,
        cost_summary=cost_summary,
    )
