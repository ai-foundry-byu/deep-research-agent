"""Per-claim adversarial verification — grafts the /deep-research skill's
3-vote refute-or-survive pattern onto the researcher pipeline.

Replicated from the Claude Code `deep-research` skill workflow
(`deep-research-wf_*.js`), specifically its Verify phase:

    "3-vote adversarial verification per claim (need 2/3 refutes to kill)"

Pipeline position (see core/loop.py): runs AFTER the writer synthesizes the
draft report and BEFORE the Obsidian write. Stages:

1. extract_claims()  — parse the synthesized draft into discrete, falsifiable
   factual claims (names, numbers, dates, causal assertions).
2. verify_claim()    — for each claim, run N independent adversarial voters,
   each of which TRIES TO REFUTE the claim using a fresh refutation search.
   Reuses the researcher's own live tooling (brave/twitter/reddit/...) so
   verification can pull first-party X/Reddit evidence too.
3. Quorum → status:
     VERIFIED   — adjudicated, survived, at least one voter found supporting
                  evidence with a source URL.
     UNVERIFIED — adjudicated but no source/evidence found either way (skeptical
                  default: "we couldn't confirm this"), OR too many abstentions
                  to reach quorum.
     REFUTED    — quorum of voters found contradicting evidence.
4. ClaimVerificationResult.format_for_report() — renders the "Claim
   Verification" table and the inline-flagging directives the writer/annotator
   uses to down-weight UNVERIFIED/REFUTED claims in the body.

Default-to-skeptical: any uncertainty resolves toward REFUTED at the vote
level (matching the skill), and a claim only earns VERIFIED if a voter actually
produced a supporting source URL.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from ..models import query_llm, CostTracker

# ── Tunables (mirror the deep-research skill constants) ──────────────────────
VOTES_PER_CLAIM = 3            # adversarial voters per claim
REFUTATIONS_REQUIRED = 2       # >= this many refute votes kills the claim (2/3)
MAX_VERIFY_CLAIMS = 25         # cap on claims verified (cost/latency guard)

# Status constants
VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
REFUTED = "REFUTED"

# A search function reused from loop.execute_tools' TOOL_MAP. Signature:
#   async (tool_name: str, query: str) -> str   (formatted evidence text)
# Injected so verifier.py never imports loop.py (avoids a cycle) and so tests
# can pass a hermetic stub.
RefuteSearchFn = Callable[[str, str], Awaitable[str]]


@dataclass
class ClaimVerdict:
    """One adversarial voter's verdict on a claim."""
    refuted: bool
    confidence: str          # "high" | "medium" | "low"
    evidence: str = ""
    source_url: str = ""     # supporting source if refuted=False; counter-source if refuted=True

    def to_dict(self) -> dict:
        return {
            "refuted": self.refuted,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "source_url": self.source_url,
        }


@dataclass
class VerifiedClaim:
    """A claim plus its adjudicated verification status."""
    claim: str
    importance: str = "supporting"   # "central" | "supporting" | "tangential"
    status: str = UNVERIFIED         # VERIFIED | UNVERIFIED | REFUTED
    source_url: str = ""             # best supporting (VERIFIED) or counter (REFUTED) source
    confidence: str = "low"          # best voter confidence backing the status
    refute_votes: int = 0
    support_votes: int = 0
    abstentions: int = 0
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    note: str = ""

    @property
    def vote_str(self) -> str:
        return f"{self.support_votes}-{self.refute_votes}"

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "importance": self.importance,
            "status": self.status,
            "source_url": self.source_url,
            "confidence": self.confidence,
            "refute_votes": self.refute_votes,
            "support_votes": self.support_votes,
            "abstentions": self.abstentions,
            "note": self.note,
        }


@dataclass
class ClaimVerificationResult:
    """Full output of the per-claim adversarial verification stage."""
    claims: list[VerifiedClaim] = field(default_factory=list)
    enabled: bool = True
    note: str = ""

    @property
    def verified(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.status == VERIFIED]

    @property
    def unverified(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.status == UNVERIFIED]

    @property
    def refuted(self) -> list[VerifiedClaim]:
        return [c for c in self.claims if c.status == REFUTED]

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "total": len(self.claims),
            "verified": len(self.verified),
            "unverified": len(self.unverified),
            "refuted": len(self.refuted),
            "claims": [c.to_dict() for c in self.claims],
        }

    def summary(self) -> str:
        if not self.claims:
            return "no claims verified"
        return (
            f"{len(self.verified)} verified, {len(self.unverified)} unverified, "
            f"{len(self.refuted)} refuted (of {len(self.claims)})"
        )

    def format_for_report(self) -> str:
        """Render the 'Claim Verification' section appended to the report.

        Includes a status table plus an explicit list of the shaky
        (UNVERIFIED/REFUTED) claims so the reader sees what's solid vs not.
        """
        if not self.claims:
            return ""

        emoji = {VERIFIED: "✅", UNVERIFIED: "⚠️", REFUTED: "❌"}
        lines: list[str] = []
        lines.append("## Claim Verification")
        lines.append("")
        lines.append(
            f"_Adversarial per-claim fact-check ({VOTES_PER_CLAIM}-vote refute-or-survive, "
            f"≥{REFUTATIONS_REQUIRED} refutations kill a claim). "
            f"{self.summary()}._"
        )
        lines.append("")
        lines.append("| Claim | Status | Vote (✓-✗) | Source / Confidence |")
        lines.append("|---|---|---|---|")
        for c in self.claims:
            claim_cell = c.claim.replace("|", "\\|")
            if len(claim_cell) > 140:
                claim_cell = claim_cell[:137] + "…"
            status_cell = f"{emoji.get(c.status, '')} {c.status}"
            if c.source_url:
                src_cell = f"[{c.confidence}]({c.source_url})"
            else:
                src_cell = f"{c.confidence} confidence / no source"
            lines.append(f"| {claim_cell} | {status_cell} | {c.vote_str} | {src_cell} |")

        shaky = self.unverified + self.refuted
        if shaky:
            lines.append("")
            lines.append("### Flagged claims (treat with caution)")
            for c in shaky:
                tag = "REFUTED" if c.status == REFUTED else "UNVERIFIED"
                detail = c.note or (
                    "contradicting evidence found" if c.status == REFUTED
                    else "no supporting source located"
                )
                lines.append(f"- **[{tag}]** {c.claim} — {detail}")

        return "\n".join(lines)


# ── Stage 1: claim extraction ────────────────────────────────────────────────
async def extract_claims(
    report_markdown: str,
    query: str,
    profile: str = "opus",
    cost_tracker: Optional[CostTracker] = None,
    max_claims: int = MAX_VERIFY_CLAIMS,
) -> list[VerifiedClaim]:
    """Parse the synthesized draft into discrete falsifiable factual claims.

    Targets names, numbers, dates, and causal assertions — the kind of
    statement that can be independently checked. Subjective/recommendation
    sentences are skipped.
    """
    prompt = f"""You are a Claim Extractor. Pull the discrete, FALSIFIABLE factual
claims out of the research report below so each can be independently fact-checked.

RESEARCH QUESTION: {query}

REPORT DRAFT:
{report_markdown[:9000]}

A falsifiable claim is a concrete, checkable statement: a named entity, a number
or statistic, a date, a version/release, or a causal/comparative assertion
("X causes Y", "A is faster than B"). SKIP subjective opinions, recommendations,
hedges, and vague generalities.

For each claim, rate its importance to the research question:
- "central"    — directly answers the question
- "supporting" — backs up a central point
- "tangential" — peripheral detail

Return ONLY a JSON array (max {max_claims} items), most important first:
[
  {{"claim": "concrete checkable statement", "importance": "central"}},
  {{"claim": "another statement", "importance": "supporting"}}
]"""

    try:
        raw = await query_llm(
            prompt, role="claim_extractor", profile=profile, cost_tracker=cost_tracker
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"   ⚠️  Claim extractor error: {e}")
        return []

    claims = _parse_claims_json(raw)
    return claims[:max_claims]


def _parse_claims_json(raw: str) -> list[VerifiedClaim]:
    """Tolerant JSON-array parse → VerifiedClaim list."""
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start:end])
    except Exception:
        return []

    out: list[VerifiedClaim] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("claim", "")).strip()
            imp = str(item.get("importance", "supporting")).strip().lower()
        elif isinstance(item, str):
            text, imp = item.strip(), "supporting"
        else:
            continue
        if not text:
            continue
        if imp not in ("central", "supporting", "tangential"):
            imp = "supporting"
        out.append(VerifiedClaim(claim=text, importance=imp))
    # Rank central > supporting > tangential, preserving extractor order within rank
    rank = {"central": 0, "supporting": 1, "tangential": 2}
    out.sort(key=lambda c: rank.get(c.importance, 1))
    return out


# ── Stage 2: per-claim adversarial verification ──────────────────────────────
_URL_RE = re.compile(r"https?://[^\s\)\]]+")


async def _one_vote(
    claim: VerifiedClaim,
    query: str,
    refute_search: RefuteSearchFn,
    voter_idx: int,
    profile: str,
    cost_tracker: Optional[CostTracker],
) -> Optional[ClaimVerdict]:
    """Run a single adversarial voter. Returns None on hard error (= abstain)."""
    # Pull fresh evidence the voter can use to refute. Reuse the live tooling
    # (brave web + X/Reddit first-party). Best-effort: if search fails, the
    # voter still reasons from the claim/quote alone.
    evidence_text = ""
    try:
        evidence_text = await refute_search(claim.claim, query)
    except Exception as e:  # pragma: no cover - defensive
        evidence_text = f"(refutation search failed: {str(e)[:120]})"

    prompt = f"""You are an Adversarial Claim Verifier (voter {voter_idx + 1}/{VOTES_PER_CLAIM}).
Be SKEPTICAL. Your job is to TRY TO REFUTE the claim. Default to refuted=true if uncertain.

RESEARCH QUESTION: {query}

CLAIM UNDER REVIEW:
"{claim.claim}"

EVIDENCE GATHERED (from a fresh search across web + X/Reddit):
{evidence_text[:6000] if evidence_text else "(no evidence returned)"}

CHECKLIST:
1. Does the gathered evidence SUPPORT the claim, CONTRADICT it, or say nothing?
2. Is the source quality sufficient for the claim's strength? (extraordinary
   claims need primary/authoritative sources, not blogs/forums/marketing.)
3. Is the claim outdated for a fast-moving topic? (old = suspect)
4. Is this a marketing/press-release/cherry-picked/speculative claim?

DECISION:
- refuted=true if: contradicted by evidence, unsupported by any credible source,
  low-quality source for a strong claim, outdated, or marketing fluff.
- refuted=false ONLY if: a credible source in the evidence actually SUPPORTS the
  claim. When refuted=false you MUST provide that supporting source_url.
- If the evidence is silent / inconclusive, set refuted=true (skeptical default)
  and leave source_url empty.

Respond ONLY with JSON:
{{"refuted": true|false, "confidence": "high|medium|low", "evidence": "1-2 sentence justification", "source_url": "https://... or empty"}}"""

    try:
        raw = await query_llm(
            prompt, role="claim_verifier", profile=profile, cost_tracker=cost_tracker
        )
    except Exception as e:  # pragma: no cover - defensive
        print(f"   ⚠️  Verifier vote error: {e}")
        return None

    return _parse_verdict(raw, evidence_text)


def _parse_verdict(raw: str, evidence_text: str = "") -> Optional[ClaimVerdict]:
    """Tolerant JSON-object parse → ClaimVerdict. None if unparseable (abstain)."""
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(raw[start:end])
    except Exception:
        return None

    refuted = bool(obj.get("refuted", True))  # skeptical default
    confidence = str(obj.get("confidence", "low")).strip().lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    evidence = str(obj.get("evidence", "")).strip()
    source_url = str(obj.get("source_url", "")).strip()

    # Backstop: if the model claimed support but gave no URL, try to recover one
    # from the gathered evidence text so a VERIFIED status always carries a source.
    if not refuted and not source_url and evidence_text:
        m = _URL_RE.search(evidence_text)
        if m:
            source_url = m.group(0).rstrip(".,;:!?)")

    return ClaimVerdict(
        refuted=refuted, confidence=confidence, evidence=evidence, source_url=source_url
    )


def _adjudicate(claim: VerifiedClaim, verdicts: list[Optional[ClaimVerdict]]) -> VerifiedClaim:
    """Apply quorum logic to a claim's votes and set its status.

    Mirrors the deep-research skill:
      - abstentions (None) don't count toward quorum
      - REFUTED if >= REFUTATIONS_REQUIRED valid voters refute
      - otherwise survives IF a quorum of valid votes was reached
    A surviving claim is VERIFIED only when a non-refuting voter produced a
    supporting source URL; otherwise it's UNVERIFIED (couldn't confirm).
    """
    valid = [v for v in verdicts if v is not None]
    claim.verdicts = valid
    claim.abstentions = len(verdicts) - len(valid)
    claim.refute_votes = sum(1 for v in valid if v.refuted)
    claim.support_votes = sum(1 for v in valid if not v.refuted)

    # Not enough valid votes to adjudicate → UNVERIFIED (skeptical: don't pass).
    if len(valid) < REFUTATIONS_REQUIRED:
        claim.status = UNVERIFIED
        claim.confidence = "low"
        claim.note = "insufficient votes to adjudicate (too many abstentions)"
        return claim

    if claim.refute_votes >= REFUTATIONS_REQUIRED:
        claim.status = REFUTED
        # best counter-source = highest-confidence refuting voter with a URL
        counter = sorted(
            (v for v in valid if v.refuted),
            key=lambda v: {"high": 0, "medium": 1, "low": 2}.get(v.confidence, 2),
        )
        claim.confidence = counter[0].confidence if counter else "low"
        for v in counter:
            if v.source_url:
                claim.source_url = v.source_url
                break
        claim.note = (counter[0].evidence if counter else "") or "contradicting evidence found"
        return claim

    # Survives the refute vote. VERIFIED only if a supporter found a real source.
    supporters = sorted(
        (v for v in valid if not v.refuted and v.source_url),
        key=lambda v: {"high": 0, "medium": 1, "low": 2}.get(v.confidence, 2),
    )
    if supporters:
        claim.status = VERIFIED
        claim.source_url = supporters[0].source_url
        claim.confidence = supporters[0].confidence
        claim.note = supporters[0].evidence
    else:
        # Survived (not enough refutations) but nobody produced a source → can't confirm.
        claim.status = UNVERIFIED
        claim.confidence = "low"
        claim.note = "survived adversarial vote but no supporting source located"
    return claim


async def verify_claim(
    claim: VerifiedClaim,
    query: str,
    refute_search: RefuteSearchFn,
    profile: str = "opus",
    cost_tracker: Optional[CostTracker] = None,
    votes: int = VOTES_PER_CLAIM,
) -> VerifiedClaim:
    """Run `votes` independent adversarial voters on one claim, then adjudicate."""
    verdicts = await asyncio.gather(
        *[
            _one_vote(claim, query, refute_search, i, profile, cost_tracker)
            for i in range(votes)
        ],
        return_exceptions=False,
    )
    return _adjudicate(claim, list(verdicts))


# ── Stage orchestration ──────────────────────────────────────────────────────
async def verify_report(
    report_markdown: str,
    query: str,
    refute_search: RefuteSearchFn,
    profile: str = "opus",
    cost_tracker: Optional[CostTracker] = None,
    votes_per_claim: int = VOTES_PER_CLAIM,
    max_claims: int = MAX_VERIFY_CLAIMS,
) -> ClaimVerificationResult:
    """Full per-claim adversarial verification of a synthesized report.

    1. Extract falsifiable claims from the draft.
    2. Verify each in parallel (N adversarial voters apiece).
    3. Return the adjudicated result (table + flags via format_for_report()).
    """
    claims = await extract_claims(
        report_markdown, query, profile=profile,
        cost_tracker=cost_tracker, max_claims=max_claims,
    )
    if not claims:
        return ClaimVerificationResult(claims=[], enabled=True, note="no claims extracted")

    verified = await asyncio.gather(
        *[
            verify_claim(
                c, query, refute_search,
                profile=profile, cost_tracker=cost_tracker, votes=votes_per_claim,
            )
            for c in claims
        ]
    )
    return ClaimVerificationResult(claims=list(verified), enabled=True)


def annotate_report(report_markdown: str, result: ClaimVerificationResult) -> str:
    """Append the Claim Verification section and inline-flag shaky claims.

    The table is always appended. For REFUTED/UNVERIFIED claims, we additionally
    inject an inline marker the first time the claim's lead phrase appears in the
    body, so a reader scanning prose sees the flag without scrolling to the table.
    """
    if not result or not result.claims:
        return report_markdown

    body = report_markdown
    # Inline down-weighting markers for shaky claims (best-effort substring match
    # on the first ~60 chars of the claim — cheap, no false-positive risk on miss).
    for c in result.unverified + result.refuted:
        marker = " `[REFUTED]`" if c.status == REFUTED else " `[UNVERIFIED]`"
        lead = c.claim[:60].rstrip()
        if not lead:
            continue
        idx = body.find(lead)
        if idx >= 0 and marker.strip() not in body[idx: idx + len(lead) + 14]:
            insert_at = idx + len(lead)
            body = body[:insert_at] + marker + body[insert_at:]

    table = result.format_for_report()
    if table:
        body = body.rstrip() + "\n\n" + table + "\n"
    return body
