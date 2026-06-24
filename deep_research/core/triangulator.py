"""Cross-source triangulation — identify claims across multiple sources, score confidence,
and flag contradictions.

After all sources are gathered, this module:
1. Extracts key claims from findings
2. Maps each claim to its supporting sources
3. Scores claim confidence by independent source count
4. Flags contradictions between sources

Phase 3: Advanced Intelligence.
"""
import json
from dataclasses import dataclass, field
from ..models import query_llm, CostTracker


@dataclass
class TriangulatedClaim:
    """A claim extracted from research findings with cross-source analysis."""
    claim: str
    confidence: str              # "high", "medium", "low"
    supporting_sources: list[str] = field(default_factory=list)
    contradicting_sources: list[str] = field(default_factory=list)
    source_count: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "confidence": self.confidence,
            "supporting_sources": self.supporting_sources,
            "contradicting_sources": self.contradicting_sources,
            "source_count": self.source_count,
            "notes": self.notes,
        }


@dataclass
class TriangulationResult:
    """Full triangulation output for a research run."""
    claims: list[TriangulatedClaim] = field(default_factory=list)
    contradictions: list[dict] = field(default_factory=list)
    summary: str = ""

    @property
    def high_confidence_claims(self) -> list[TriangulatedClaim]:
        return [c for c in self.claims if c.confidence == "high"]

    @property
    def low_confidence_claims(self) -> list[TriangulatedClaim]:
        return [c for c in self.claims if c.confidence == "low"]

    def to_dict(self) -> dict:
        return {
            "claims": [c.to_dict() for c in self.claims],
            "contradictions": self.contradictions,
            "high_confidence_count": len(self.high_confidence_claims),
            "low_confidence_count": len(self.low_confidence_claims),
            "total_claims": len(self.claims),
        }

    def format_for_report(self) -> str:
        """Format triangulation results for inclusion in the final report."""
        if not self.claims:
            return ""

        sections = []

        # High confidence claims
        high = self.high_confidence_claims
        if high:
            lines = ["### High-Confidence Claims (3+ independent sources)"]
            for c in high:
                sources_str = ", ".join(c.supporting_sources[:5])
                lines.append(f"- **{c.claim}** — supported by {c.source_count} sources ({sources_str})")
            sections.append("\n".join(lines))

        # Medium confidence
        medium = [c for c in self.claims if c.confidence == "medium"]
        if medium:
            lines = ["### Medium-Confidence Claims (2 independent sources)"]
            for c in medium:
                sources_str = ", ".join(c.supporting_sources[:3])
                lines.append(f"- {c.claim} — {sources_str}")
            sections.append("\n".join(lines))

        # Low confidence (flagged)
        low = self.low_confidence_claims
        if low:
            lines = ["### Low-Confidence Claims (single source - verify independently)"]
            for c in low:
                source = c.supporting_sources[0] if c.supporting_sources else "unknown"
                lines.append(f"- *{c.claim}* — only from {source}")
            sections.append("\n".join(lines))

        # Contradictions
        if self.contradictions:
            lines = ["### Contradictions Between Sources"]
            for ctr in self.contradictions:
                lines.append(
                    f"- **{ctr.get('topic', 'Unknown')}**: "
                    f"{ctr.get('claim_a', '')} vs. {ctr.get('claim_b', '')} "
                    f"({ctr.get('source_a', '')} vs. {ctr.get('source_b', '')})"
                )
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


async def triangulate(
    all_findings: str,
    sources: list[str],
    query: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
) -> TriangulationResult:
    """Run cross-source triangulation on research findings.

    Extracts claims, maps them to sources, scores confidence, and flags
    contradictions. Uses an LLM to perform the analysis since claim
    extraction requires understanding of the content.

    Args:
        all_findings: all research findings accumulated across iterations
        sources: list of source labels (e.g., "brave_search: query")
        query: the original research query
        profile: model profile
        cost_tracker: optional cost tracker

    Returns:
        TriangulationResult with scored claims and contradictions.
    """
    if not all_findings or not sources:
        return TriangulationResult()

    sources_text = "\n".join(f"- [{i+1}] {s}" for i, s in enumerate(sources[:30]))

    prompt = f"""You are a Cross-Source Verification Agent. Analyze research findings to identify
claims, their source support, and any contradictions.

RESEARCH QUERY: {query}

SOURCES CONSULTED:
{sources_text}

RESEARCH FINDINGS:
{all_findings[:6000]}

For each substantive claim in the findings:
1. Identify the claim
2. List which sources support it (by source number from the list above)
3. List any sources that contradict it
4. Note any contradictions between sources

Respond ONLY with valid JSON:
{{
  "claims": [
    {{
      "claim": "concise statement of the claim",
      "supporting_sources": [1, 3, 5],
      "contradicting_sources": [],
      "notes": "optional context"
    }}
  ],
  "contradictions": [
    {{
      "topic": "what the contradiction is about",
      "claim_a": "first position",
      "source_a": "source supporting first position",
      "claim_b": "second position",
      "source_b": "source supporting second position"
    }}
  ]
}}

Include 5-15 key claims. Focus on substantive, verifiable claims, not opinions or meta-observations.
List source numbers as integers corresponding to the source list above."""

    try:
        result_text = await query_llm(
            prompt, role="observations", profile=profile, cost_tracker=cost_tracker,
        )
        return _parse_triangulation(result_text, sources)
    except Exception as e:
        print(f"   Warning: Triangulation failed ({e})")
        return TriangulationResult(summary=f"Triangulation failed: {e}")


def _parse_triangulation(result_text: str, sources: list[str]) -> TriangulationResult:
    """Parse LLM triangulation output into structured result."""
    # Extract JSON from response
    start = result_text.find("{")
    end = result_text.rfind("}") + 1
    if start < 0 or end <= start:
        return TriangulationResult(summary="Could not parse triangulation output")

    try:
        data = json.loads(result_text[start:end])
    except json.JSONDecodeError:
        return TriangulationResult(summary="Invalid JSON in triangulation output")

    claims = []
    for raw_claim in data.get("claims", []):
        supporting_indices = raw_claim.get("supporting_sources", [])
        contradicting_indices = raw_claim.get("contradicting_sources", [])

        # Resolve source indices to labels
        supporting = _resolve_sources(supporting_indices, sources)
        contradicting = _resolve_sources(contradicting_indices, sources)

        source_count = len(supporting)
        if source_count >= 3:
            confidence = "high"
        elif source_count == 2:
            confidence = "medium"
        else:
            confidence = "low"

        claims.append(TriangulatedClaim(
            claim=raw_claim.get("claim", "Unknown claim"),
            confidence=confidence,
            supporting_sources=supporting,
            contradicting_sources=contradicting,
            source_count=source_count,
            notes=raw_claim.get("notes", ""),
        ))

    contradictions = data.get("contradictions", [])

    # Build summary
    high = sum(1 for c in claims if c.confidence == "high")
    medium = sum(1 for c in claims if c.confidence == "medium")
    low = sum(1 for c in claims if c.confidence == "low")
    summary = (
        f"{len(claims)} claims analyzed: {high} high-confidence, "
        f"{medium} medium-confidence, {low} single-source. "
        f"{len(contradictions)} contradiction(s) found."
    )

    return TriangulationResult(
        claims=claims,
        contradictions=contradictions,
        summary=summary,
    )


def _resolve_sources(indices: list, sources: list[str]) -> list[str]:
    """Resolve 1-based source indices to source label strings."""
    resolved = []
    for idx in indices:
        try:
            i = int(idx) - 1  # convert 1-based to 0-based
            if 0 <= i < len(sources):
                resolved.append(sources[i])
            else:
                resolved.append(f"source_{idx}")
        except (ValueError, TypeError):
            # If the LLM returned a string instead of int, use as-is
            resolved.append(str(idx))
    return resolved
