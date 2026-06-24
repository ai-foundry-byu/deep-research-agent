"""Source Credibility Scoring — LLM-based evaluation of search result quality.

Runs after tool execution, before the observations agent. Scores each source
on three axes (authority, recency, relevance) and attaches metadata so that
downstream agents can prioritize high-quality sources.
"""
import json
import re
from dataclasses import dataclass
from ..models import query_llm, CostTracker


@dataclass
class SourceScore:
    """Credibility score for a single source."""
    source_label: str
    authority: int      # 1-5: how authoritative is the source?
    recency: int        # 1-5: how recent/timely is the information?
    relevance: int      # 1-5: how relevant to the research query?

    @property
    def composite(self) -> int:
        return self.authority + self.recency + self.relevance

    @property
    def is_high_quality(self) -> bool:
        return self.composite >= 6

    def to_dict(self) -> dict:
        return {
            "source": self.source_label,
            "authority": self.authority,
            "recency": self.recency,
            "relevance": self.relevance,
            "composite": self.composite,
        }


async def score_sources(
    raw_results: str,
    query: str,
    sources: list[str],
    profile: str = "opus",
    cost_tracker: CostTracker = None,
    threshold: int = 6,
) -> tuple[str, list[SourceScore]]:
    """Score each source block in raw_results and return reordered results.

    Returns:
        (reordered_results, scores) — results with high-quality sources first
        and deprioritized sources moved to a "lower confidence" section.
    """
    if not sources:
        return raw_results, []

    # Build a condensed view of the sources for scoring
    source_summaries = _extract_source_blocks(raw_results)
    if not source_summaries:
        return raw_results, []

    # Build the scoring prompt
    source_list = "\n".join(
        f"SOURCE {i+1} [{s['label']}]:\n{s['snippet'][:400]}"
        for i, s in enumerate(source_summaries)
    )

    prompt = f"""You are a Source Credibility Evaluator. Score each source on three axes (1-5 each).

RESEARCH QUERY: {query}

SOURCES:
{source_list}

Score each source on:
- authority (1-5): Is this a credible, authoritative source? (5 = peer-reviewed/official, 1 = anonymous blog)
- recency (1-5): Is the information current? (5 = last month, 3 = last year, 1 = 5+ years old or undated)
- relevance (1-5): How directly does this address the research query? (5 = exactly on topic, 1 = tangential)

Respond ONLY with a JSON array, one object per source:
[
  {{"source": 1, "authority": 4, "recency": 3, "relevance": 5}},
  {{"source": 2, "authority": 2, "recency": 4, "relevance": 3}}
]"""

    try:
        result_text = await query_llm(
            prompt,
            role="tool_selector",  # use a lighter-weight model role
            profile=profile,
            cost_tracker=cost_tracker,
        )

        scores = _parse_scores(result_text, source_summaries)
    except Exception as e:
        print(f"   Warning: Source scoring failed ({e}), passing results unscored")
        return raw_results, []

    # Reorder: high-quality sources first, deprioritized at the end
    high_quality_blocks = []
    low_quality_blocks = []

    for i, summary in enumerate(source_summaries):
        score = scores[i] if i < len(scores) else None
        block = summary["raw_block"]

        if score and score.composite >= threshold:
            label = f"[Score: {score.composite}/15 | Auth:{score.authority} Rec:{score.recency} Rel:{score.relevance}]"
            high_quality_blocks.append(f"{label}\n{block}")
        elif score:
            label = f"[Score: {score.composite}/15 | Auth:{score.authority} Rec:{score.recency} Rel:{score.relevance} | DEPRIORITIZED]"
            low_quality_blocks.append(f"{label}\n{block}")
        else:
            high_quality_blocks.append(block)

    reordered_parts = high_quality_blocks
    if low_quality_blocks:
        reordered_parts.append(
            "\n=== Lower Confidence Sources (composite score < {}) ===".format(threshold)
        )
        reordered_parts.extend(low_quality_blocks)

    return "\n\n".join(reordered_parts), scores


def _extract_source_blocks(raw_results: str) -> list[dict]:
    """Split raw_results into per-source blocks."""
    # Tool results are delimited by "=== Results from <tool> ===" or "=== Auto-scraped: <url> ==="
    pattern = r'(===\s*(?:Results from|Auto-scraped:)\s*.+?\s*===)'
    parts = re.split(pattern, raw_results)

    blocks = []
    i = 0
    while i < len(parts):
        if re.match(pattern, parts[i].strip()):
            header = parts[i].strip()
            content = parts[i + 1] if i + 1 < len(parts) else ""
            # Extract a label from the header
            label_match = re.search(r'===\s*(?:Results from|Auto-scraped:)\s*(.+?)\s*===', header)
            label = label_match.group(1) if label_match else f"source_{len(blocks)}"
            blocks.append({
                "label": label,
                "snippet": content.strip()[:600],
                "raw_block": f"{header}\n{content}",
            })
            i += 2
        else:
            # Content before the first header or between headers without a match
            if parts[i].strip():
                blocks.append({
                    "label": f"source_{len(blocks)}",
                    "snippet": parts[i].strip()[:600],
                    "raw_block": parts[i],
                })
            i += 1

    return blocks


def _parse_scores(result_text: str, source_summaries: list[dict]) -> list[SourceScore]:
    """Parse LLM JSON response into SourceScore objects."""
    start = result_text.find("[")
    end = result_text.rfind("]") + 1
    if start < 0 or end <= start:
        return []

    raw_scores = json.loads(result_text[start:end])
    scores = []
    for i, entry in enumerate(raw_scores):
        label = source_summaries[i]["label"] if i < len(source_summaries) else f"source_{i}"
        scores.append(SourceScore(
            source_label=label,
            authority=_clamp(entry.get("authority", 3), 1, 5),
            recency=_clamp(entry.get("recency", 3), 1, 5),
            relevance=_clamp(entry.get("relevance", 3), 1, 5),
        ))

    return scores


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer to [lo, hi]."""
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return 3  # default mid-score on parse failure
