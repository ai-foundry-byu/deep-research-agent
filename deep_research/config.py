"""Research agent configuration"""
from dataclasses import dataclass, field
import os

VALID_PROFILES = ("opus", "multi")

@dataclass
class ResearchConfig:
    # Profile — default to "multi" (cheap cloud models) to avoid Opus dependency
    profile: str = "multi"

    # Loop control
    max_iterations: int = 5
    max_time_minutes: int = 15
    critique_loop: bool = True       # feed devil's advocate back into research
    max_critique_iterations: int = 2  # extra iterations from critique

    # Depth presets
    shallow_iterations: int = 2
    standard_iterations: int = 5
    deep_iterations: int = 10

    # Depth label (set by for_depth)
    depth: str = "standard"

    # Tool configuration
    brave_results_per_query: int = 5
    github_results_per_query: int = 5
    reddit_results: int = 10
    arxiv_max_results: int = 5
    hn_results_per_query: int = 5
    semantic_scholar_results: int = 10
    stackoverflow_results: int = 5

    # Auto-scrape: number of top Brave URLs to scrape via Firecrawl
    auto_scrape_limit: int = 3

    # X/Twitter search results
    twitter_results: int = 10

    # Recency / freshness bias.
    # Brave 'freshness' param: pd=past day, pw=past week, pm=past month, py=past year, "" = no filter.
    # Social sources (X/Twitter, Reddit) are ephemeral — bias HARD toward current posts.
    # General web is biased toward recent but not brutal; set "" for evergreen/academic topics.
    social_freshness: str = "pw"      # X/Twitter + Reddit-via-Brave: past week (Brave-fallback recency hint)
    web_freshness: str = "pm"         # general Brave web: past month
    reddit_time_filter: str = "week"  # Reddit OAuth 't' param when REDDIT_CLIENT_ID/SECRET are set

    # Social trailing-window for the HIGH-FIDELITY paths (Reddit OAuth, X via twitterapi.io).
    # Compile X/Reddit resources from a trailing ~2-3 month window, surfacing the
    # most-recent items first. This precise day-window overrides the coarse
    # Brave 'freshness' enum (which has no "past quarter") when real API creds are present.
    social_lookback_days: int = 90    # ~3 months; newest-first within the window

    # Source credibility scoring
    source_scoring: bool = True       # run SourceCurator after tool execution
    source_score_threshold: int = 6   # composite score below this = deprioritized

    # Context compression
    context_compression: bool = True  # use embedding-based compression
    compressed_context_chars: int = 8000   # character budget for compressed observations context
    compressed_findings_chars: int = 4000  # character budget for compressed findings context

    # Phase 3: Multi-hop reasoning
    multi_hop: bool = True            # enable multi-hop sub-query decomposition
    max_sub_queries: int = 4          # max sub-questions per decomposition

    # Phase 3: Citation graph
    citation_max_papers: int = 10     # max seed papers for citation analysis
    citation_max_per_paper: int = 5   # max "cited by" papers per seed

    # Adversarial per-claim verification (deep-research graft).
    # After synthesis, extract falsifiable claims and run an N-vote refute-or-survive
    # check on each, reusing the live web + X/Reddit tooling. Depth-gated:
    #   shallow → off (keep it fast), standard → light (1 vote, fewer claims),
    #   deep    → full (3-vote adversarial). Forced on/off via CLI --verify-claims.
    claim_verification: bool = True
    claim_verify_votes: int = 3       # adversarial voters per claim (deep default)
    claim_verify_max_claims: int = 25 # cap on claims verified per run

    # Output
    # Where finished reports are written (Markdown). Override with the
    # DEEP_RESEARCH_OUTPUT_DIR env var (e.g. point it at an Obsidian vault).
    output_dir: str = field(default_factory=lambda: os.path.expanduser(
        os.environ.get("DEEP_RESEARCH_OUTPUT_DIR", "~/deep-research-reports")
    ))

    def for_depth(self, depth: str) -> "ResearchConfig":
        """Adjust config based on research depth"""
        self.depth = depth
        if depth == "shallow":
            self.max_iterations = self.shallow_iterations
            self.max_critique_iterations = 1
            self.auto_scrape_limit = 1
            self.multi_hop = False          # skip multi-hop for shallow
            self.max_sub_queries = 2
            self.citation_max_papers = 5
            # shallow stays fast — no per-claim verification by default
            self.claim_verification = False
        elif depth == "deep":
            self.max_iterations = self.deep_iterations
            self.max_critique_iterations = 3
            self.auto_scrape_limit = 5
            self.multi_hop = True
            self.max_sub_queries = 4
            self.citation_max_papers = 15
            self.citation_max_per_paper = 8
            # deep = full adversarial pass
            self.claim_verification = True
            self.claim_verify_votes = 3
            self.claim_verify_max_claims = 25
        else:
            self.max_iterations = self.standard_iterations
            self.max_critique_iterations = 2
            self.auto_scrape_limit = 3
            self.multi_hop = True
            self.max_sub_queries = 3
            self.citation_max_papers = 10
            # standard = lighter pass (single vote, fewer claims) to bound cost
            self.claim_verification = True
            self.claim_verify_votes = 1
            self.claim_verify_max_claims = 12
        return self

    def for_profile(self, profile: str) -> "ResearchConfig":
        """Set the model profile"""
        if profile not in VALID_PROFILES:
            raise ValueError(f"Unknown profile '{profile}'. Must be one of {VALID_PROFILES}")
        self.profile = profile
        return self
