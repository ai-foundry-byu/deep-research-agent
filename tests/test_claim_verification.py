"""Tests for the per-claim adversarial verification stage (deep-research graft).

Hermetic: every external call (LLM via query_llm, the refutation search) is
mocked. Covers:
  - claim extraction parses falsifiable claims from a draft, ranks by importance
  - a supported claim with a source → VERIFIED (carries the source URL)
  - an unsupported claim (no source found) → UNVERIFIED
  - a contradicted claim (quorum refutes) → REFUTED
  - quorum/abstention logic: too many abstentions → UNVERIFIED (doesn't pass)
  - the Claim Verification table is appended to the report by annotate_report()
  - inline flags are injected for shaky claims
  - depth gating: shallow off, standard light, deep full
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from deep_research.core import verifier  # noqa: E402
from deep_research.core.verifier import (  # noqa: E402
    VerifiedClaim,
    ClaimVerdict,
    ClaimVerificationResult,
    extract_claims,
    verify_claim,
    verify_report,
    annotate_report,
    _adjudicate,
    _parse_claims_json,
    _parse_verdict,
    VERIFIED,
    UNVERIFIED,
    REFUTED,
)
from deep_research.config import ResearchConfig  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


# ── Stub helpers ─────────────────────────────────────────────────────────────
def make_query_llm_stub(responses: dict):
    """Return an async stub for query_llm keyed by role.

    responses[role] may be a string (returned every call) or a list (popped
    per call, so successive voters can return different verdicts).
    """
    state = {k: (list(v) if isinstance(v, list) else v) for k, v in responses.items()}

    async def _stub(prompt, role, profile="opus", cost_tracker=None, max_tokens=4096):
        val = state.get(role, "{}")
        if isinstance(val, list):
            return val.pop(0) if val else "{}"
        return val

    return _stub


async def _noop_search(claim, query):
    return "=== Evidence from web ===\n(no decisive evidence)"


# ── Pure-function parsing ────────────────────────────────────────────────────
def test_parse_claims_json_ranks_central_first():
    raw = """Here are the claims:
[
  {"claim": "GPT-4 launched in 2023", "importance": "supporting"},
  {"claim": "Transformers use self-attention", "importance": "central"},
  {"claim": "Some side note", "importance": "tangential"}
]
done"""
    claims = _parse_claims_json(raw)
    assert len(claims) == 3
    # central is ranked first
    assert claims[0].importance == "central"
    assert claims[0].claim == "Transformers use self-attention"
    assert claims[-1].importance == "tangential"


def test_parse_claims_json_handles_garbage():
    assert _parse_claims_json("no json here") == []
    assert _parse_claims_json("[broken json") == []


def test_parse_verdict_skeptical_default():
    # Missing 'refuted' field → skeptical default True
    v = _parse_verdict('{"confidence": "low", "evidence": "unclear"}')
    assert v is not None
    assert v.refuted is True
    # Unparseable → None (abstain)
    assert _parse_verdict("not json") is None


def test_parse_verdict_recovers_url_from_evidence():
    # Model said supported (refuted=false) but gave no source_url — recover from evidence text
    v = _parse_verdict(
        '{"refuted": false, "confidence": "high", "evidence": "supported", "source_url": ""}',
        evidence_text="See https://arxiv.org/abs/1706.03762 for the proof.",
    )
    assert v.refuted is False
    assert v.source_url == "https://arxiv.org/abs/1706.03762"


# ── Stage 1: claim extraction ────────────────────────────────────────────────
def test_extract_claims(monkeypatch):
    stub = make_query_llm_stub({
        "claim_extractor": '[{"claim": "Rust is memory-safe without GC", "importance": "central"}]'
    })
    monkeypatch.setattr(verifier, "query_llm", stub)

    claims = _run(extract_claims("draft text", "is Rust memory safe?", profile="opus"))
    assert len(claims) == 1
    assert isinstance(claims[0], VerifiedClaim)
    assert "memory-safe" in claims[0].claim
    assert claims[0].importance == "central"


# ── Stage 2: a single claim, three statuses ──────────────────────────────────
def test_verified_claim_gets_source(monkeypatch):
    # All voters say supported + provide a source → VERIFIED with the URL
    verdict = '{"refuted": false, "confidence": "high", "evidence": "official docs confirm", "source_url": "https://doc.rust-lang.org/book"}'
    stub = make_query_llm_stub({"claim_verifier": verdict})
    monkeypatch.setattr(verifier, "query_llm", stub)

    claim = VerifiedClaim(claim="Rust has no garbage collector", importance="central")
    out = _run(verify_claim(claim, "rust", _noop_search, profile="opus", votes=3))

    assert out.status == VERIFIED
    assert out.source_url == "https://doc.rust-lang.org/book"
    assert out.support_votes == 3
    assert out.refute_votes == 0


def test_unsupported_claim_is_unverified(monkeypatch):
    # Voters don't refute, but nobody supplies a supporting source → UNVERIFIED.
    # (refuted=false but source_url empty AND no URL in the search evidence)
    verdict = '{"refuted": false, "confidence": "low", "evidence": "could not find a source", "source_url": ""}'
    stub = make_query_llm_stub({"claim_verifier": verdict})
    monkeypatch.setattr(verifier, "query_llm", stub)

    claim = VerifiedClaim(claim="An obscure unfounded assertion", importance="supporting")
    out = _run(verify_claim(claim, "topic", _noop_search, profile="opus", votes=3))

    assert out.status == UNVERIFIED
    assert out.source_url == ""


def test_refuted_claim(monkeypatch):
    # 3 voters all refute with a counter-source → REFUTED (>= 2/3)
    verdict = '{"refuted": true, "confidence": "high", "evidence": "contradicted by primary source", "source_url": "https://example.com/debunk"}'
    stub = make_query_llm_stub({"claim_verifier": verdict})
    monkeypatch.setattr(verifier, "query_llm", stub)

    claim = VerifiedClaim(claim="The earth is 6000 years old", importance="central")
    out = _run(verify_claim(claim, "geology", _noop_search, profile="opus", votes=3))

    assert out.status == REFUTED
    assert out.refute_votes == 3
    assert out.source_url == "https://example.com/debunk"


# ── Quorum / abstention adjudication (pure) ──────────────────────────────────
def test_adjudicate_refute_quorum():
    claim = VerifiedClaim(claim="x")
    verdicts = [
        ClaimVerdict(refuted=True, confidence="high", source_url="https://a"),
        ClaimVerdict(refuted=True, confidence="medium"),
        ClaimVerdict(refuted=False, confidence="low", source_url="https://b"),
    ]
    out = _adjudicate(claim, verdicts)
    assert out.status == REFUTED          # 2 refutes >= REFUTATIONS_REQUIRED
    assert out.source_url == "https://a"  # best (high-confidence) counter source


def test_adjudicate_too_many_abstentions_is_unverified():
    claim = VerifiedClaim(claim="x")
    # 2 abstentions (None) + 1 support → only 1 valid vote < quorum (2) → UNVERIFIED
    verdicts = [None, None, ClaimVerdict(refuted=False, confidence="high", source_url="https://a")]
    out = _adjudicate(claim, verdicts)
    assert out.status == UNVERIFIED
    assert out.abstentions == 2


def test_adjudicate_survives_with_source_is_verified():
    claim = VerifiedClaim(claim="x")
    verdicts = [
        ClaimVerdict(refuted=False, confidence="medium", source_url="https://src"),
        ClaimVerdict(refuted=False, confidence="low"),
        ClaimVerdict(refuted=True, confidence="low"),
    ]
    out = _adjudicate(claim, verdicts)  # 1 refute < quorum → survives; supporter has source
    assert out.status == VERIFIED
    assert out.source_url == "https://src"


# ── End-to-end verify_report (extraction + verification) ─────────────────────
def test_verify_report_end_to_end(monkeypatch):
    extractor = '[{"claim": "Python 3.12 shipped in 2023", "importance": "central"}, {"claim": "made-up stat", "importance": "supporting"}]'
    # First claim: supported w/ source. Second claim: refuted.
    # Voters share one role queue; we return support for the run via a function
    # that keys off the prompt content.
    def keyed_stub(prompt, role, profile="opus", cost_tracker=None, max_tokens=4096):
        if role == "claim_extractor":
            return extractor
        if "Python 3.12 shipped in 2023" in prompt:
            return '{"refuted": false, "confidence": "high", "evidence": "release notes", "source_url": "https://python.org/downloads"}'
        return '{"refuted": true, "confidence": "high", "evidence": "no such stat exists", "source_url": "https://example.com/no"}'

    async def astub(prompt, role, profile="opus", cost_tracker=None, max_tokens=4096):
        return keyed_stub(prompt, role, profile, cost_tracker, max_tokens)

    monkeypatch.setattr(verifier, "query_llm", astub)

    result = _run(verify_report(
        "Python 3.12 shipped in 2023. Also, made-up stat is true.",
        "python releases",
        _noop_search,
        profile="opus",
        votes_per_claim=3,
        max_claims=10,
    ))
    assert len(result.claims) == 2
    statuses = {c.claim: c.status for c in result.claims}
    assert statuses["Python 3.12 shipped in 2023"] == VERIFIED
    assert statuses["made-up stat"] == REFUTED
    assert len(result.verified) == 1
    assert len(result.refuted) == 1


# ── Report annotation: table + inline flags ──────────────────────────────────
def test_format_for_report_table():
    result = ClaimVerificationResult(claims=[
        VerifiedClaim(claim="A solid fact", status=VERIFIED, source_url="https://src", confidence="high", support_votes=3),
        VerifiedClaim(claim="A shaky claim", status=UNVERIFIED, confidence="low", note="no source"),
    ])
    table = result.format_for_report()
    assert "## Claim Verification" in table
    assert "| Claim | Status |" in table
    assert "VERIFIED" in table
    assert "UNVERIFIED" in table
    assert "https://src" in table
    # shaky claims get a flagged list
    assert "Flagged claims" in table
    assert "A shaky claim" in table


def test_annotate_report_appends_table_and_inline_flags():
    body = "# Report\n\nThe earth is flat according to some. More text here.\n"
    result = ClaimVerificationResult(claims=[
        VerifiedClaim(claim="The earth is flat according to some", status=REFUTED,
                      source_url="https://nasa.gov", confidence="high", refute_votes=3,
                      note="contradicted by all evidence"),
    ])
    annotated = annotate_report(body, result)
    # table appended
    assert "## Claim Verification" in annotated
    # inline flag injected near the claim's lead phrase
    assert "`[REFUTED]`" in annotated
    # original body preserved
    assert "# Report" in annotated


def test_annotate_report_noop_when_empty():
    body = "# Report\n\nbody"
    assert annotate_report(body, ClaimVerificationResult(claims=[])) == body


# ── Depth gating ─────────────────────────────────────────────────────────────
def test_depth_gating():
    assert ResearchConfig().for_depth("shallow").claim_verification is False
    std = ResearchConfig().for_depth("standard")
    assert std.claim_verification is True
    assert std.claim_verify_votes == 1          # light pass
    deep = ResearchConfig().for_depth("deep")
    assert deep.claim_verification is True
    assert deep.claim_verify_votes == 3          # full adversarial
