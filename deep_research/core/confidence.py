"""Confidence scoring — compute an overall research confidence score for the final report.

Score is based on:
1. Number of sources consulted
2. Source quality scores (from SourceCurator)
3. Cross-source agreement (from triangulation)
4. Recency of sources
5. Coverage of the original query

Output: confidence score (0-100) with breakdown and explanation.

Phase 3: Advanced Intelligence.
"""
import json
from dataclasses import dataclass, field
from typing import Optional
from .triangulator import TriangulationResult
from ..models import query_llm, CostTracker


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of the confidence score."""
    source_count_score: float = 0.0        # 0-20: based on number of sources
    source_quality_score: float = 0.0      # 0-25: based on credibility scores
    cross_source_agreement: float = 0.0    # 0-25: based on triangulation
    recency_score: float = 0.0             # 0-15: based on source freshness
    coverage_score: float = 0.0            # 0-15: based on query coverage
    explanation: str = ""

    @property
    def total(self) -> int:
        """Overall confidence score (0-100)."""
        raw = (
            self.source_count_score
            + self.source_quality_score
            + self.cross_source_agreement
            + self.recency_score
            + self.coverage_score
        )
        return max(0, min(100, round(raw)))

    @property
    def label(self) -> str:
        """Human-readable confidence label."""
        score = self.total
        if score >= 80:
            return "high"
        elif score >= 55:
            return "moderate"
        elif score >= 35:
            return "low"
        else:
            return "very low"

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "label": self.label,
            "breakdown": {
                "source_count": round(self.source_count_score, 1),
                "source_quality": round(self.source_quality_score, 1),
                "cross_source_agreement": round(self.cross_source_agreement, 1),
                "recency": round(self.recency_score, 1),
                "coverage": round(self.coverage_score, 1),
            },
            "explanation": self.explanation,
        }

    def format_for_report(self) -> str:
        """Format confidence score for inclusion in the report header."""
        score = self.total
        label = self.label.upper()
        bar = _score_bar(score)

        lines = [
            f"**Research Confidence: {score}/100 ({label})** {bar}",
            "",
            "| Factor | Score | Max |",
            "|--------|-------|-----|",
            f"| Source Count | {self.source_count_score:.0f} | 20 |",
            f"| Source Quality | {self.source_quality_score:.0f} | 25 |",
            f"| Cross-Source Agreement | {self.cross_source_agreement:.0f} | 25 |",
            f"| Source Recency | {self.recency_score:.0f} | 15 |",
            f"| Query Coverage | {self.coverage_score:.0f} | 15 |",
            "",
        ]
        if self.explanation:
            lines.append(f"*{self.explanation}*")

        return "\n".join(lines)


def _score_bar(score: int, width: int = 20) -> str:
    """Create a visual score bar."""
    filled = round(score / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


async def compute_confidence(
    query: str,
    all_findings: str,
    sources: list[str],
    triangulation: Optional[TriangulationResult] = None,
    source_scores: Optional[list] = None,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> ConfidenceBreakdown:
    """Compute overall research confidence score.

    Combines quantitative metrics (source count, quality scores, triangulation)
    with an LLM assessment of query coverage and recency.

    Args:
        query: original research query
        all_findings: accumulated research findings
        sources: list of source labels
        triangulation: optional TriangulationResult from triangulator
        source_scores: optional list of SourceScore objects from curator
        profile: model profile
        cost_tracker: optional cost tracker

    Returns:
        ConfidenceBreakdown with score, breakdown, and explanation.
    """
    breakdown = ConfidenceBreakdown()

    # ── Factor 1: Source Count (0-20) ─────────────────────────────────────
    n_sources = len(sources)
    if n_sources >= 15:
        breakdown.source_count_score = 20.0
    elif n_sources >= 10:
        breakdown.source_count_score = 16.0
    elif n_sources >= 6:
        breakdown.source_count_score = 12.0
    elif n_sources >= 3:
        breakdown.source_count_score = 8.0
    elif n_sources >= 1:
        breakdown.source_count_score = 4.0
    else:
        breakdown.source_count_score = 0.0

    # ── Factor 2: Source Quality (0-25) ───────────────────────────────────
    if source_scores:
        total_composite = sum(getattr(s, 'composite', 9) for s in source_scores)
        avg_composite = total_composite / len(source_scores) if source_scores else 0
        # Scale: avg composite 6/15 = 0, 15/15 = 25
        breakdown.source_quality_score = max(0, min(25, (avg_composite - 6) / 9 * 25))
    else:
        # No scoring data — assume moderate quality
        breakdown.source_quality_score = 12.5

    # ── Factor 3: Cross-Source Agreement (0-25) ───────────────────────────
    if triangulation and triangulation.claims:
        total_claims = len(triangulation.claims)
        high_conf = len(triangulation.high_confidence_claims)
        low_conf = len(triangulation.low_confidence_claims)
        contradictions = len(triangulation.contradictions)

        # High agreement ratio boosts score
        agreement_ratio = high_conf / total_claims if total_claims > 0 else 0
        # Contradictions reduce score
        contradiction_penalty = min(10, contradictions * 3)
        # Single-source claims reduce score
        low_conf_penalty = min(8, low_conf * 2)

        raw_agreement = agreement_ratio * 25
        breakdown.cross_source_agreement = max(0, raw_agreement - contradiction_penalty - low_conf_penalty)
    else:
        # No triangulation data
        breakdown.cross_source_agreement = 10.0

    # ── Factors 4 & 5: Recency and Coverage (LLM-assessed) ───────────────
    try:
        recency, coverage, explanation = await _llm_assess_quality(
            query, all_findings, sources, profile, cost_tracker,
        )
        breakdown.recency_score = recency
        breakdown.coverage_score = coverage
        breakdown.explanation = explanation
    except Exception as e:
        print(f"   Warning: LLM confidence assessment failed ({e})")
        breakdown.recency_score = 7.5
        breakdown.coverage_score = 7.5
        breakdown.explanation = "Quality assessment unavailable; scores estimated."

    return breakdown


async def _llm_assess_quality(
    query: str,
    all_findings: str,
    sources: list[str],
    profile: str,
    cost_tracker: CostTracker = None,
) -> tuple[float, float, str]:
    """Use an LLM to assess source recency and query coverage.

    Returns (recency_score, coverage_score, explanation).
    Both scores are on a 0-15 scale.
    """
    sources_text = "\n".join(f"- {s}" for s in sources[:20])

    prompt = f"""You are a Research Quality Assessor. Evaluate the quality of research findings.

RESEARCH QUERY: {query}

SOURCES USED:
{sources_text}

FINDINGS SUMMARY (first 2000 chars):
{all_findings[:2000]}

Assess two factors:

1. RECENCY (0-15): How current are the sources? Are they from the last year? Last month?
   15 = all sources are from the last 3 months
   10 = most sources from the last year
   5 = sources are 1-3 years old
   0 = sources are outdated (3+ years)

2. COVERAGE (0-15): How well do the findings cover the original query?
   15 = every aspect of the query is thoroughly addressed
   10 = main aspects covered, some gaps
   5 = partial coverage with significant gaps
   0 = findings barely address the query

Respond ONLY with JSON:
{{
  "recency": <number 0-15>,
  "coverage": <number 0-15>,
  "explanation": "<1-2 sentence explanation of the overall research quality>"
}}"""

    result_text = await query_llm(
        prompt, role="tool_selector", profile=profile, cost_tracker=cost_tracker,
    )

    # Parse JSON
    start = result_text.find("{")
    end = result_text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(result_text[start:end])
        recency = max(0, min(15, float(data.get("recency", 7.5))))
        coverage = max(0, min(15, float(data.get("coverage", 7.5))))
        explanation = data.get("explanation", "")
        return recency, coverage, explanation

    return 7.5, 7.5, "Could not parse quality assessment"
