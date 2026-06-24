# Deep Research Agent

An iterative, multi-source, adversarially-verified research agent. You give it a
question; it runs a **knowledge-gap loop** that searches **9 sources in parallel**,
scrapes the best pages, challenges its own conclusions with a **devil's-advocate
pass**, and then fact-checks **every individual claim** in the final report with a
**3-vote refute-or-survive** verification stage. The output is a cited Markdown
report with a per-claim verification table.

It is built to be cheap by default (cloud models via OpenRouter + a local Ollama
worker step) and premium on demand (Claude Opus/Sonnet via the Claude SDK).

> **Credit where it's due:** the per-claim **Verify** stage — claim extraction →
> N independent adversarial voters per claim → 2-of-3 quorum to kill a claim — is
> **ported from Claude Code's `/deep-research` skill**. That skill's "3-vote
> adversarial verification per claim (need 2/3 refutes to kill)" pattern is the
> direct ancestor of [`deep_research/core/verifier.py`](deep_research/core/verifier.py).
> The rest of the pipeline (the gap loop, the 9-source fan-out, multi-hop
> decomposition, the citation graph, embedding compression) is original.

---

## Why it's different from "search → summarize"

Most "research agents" do one round of search and ask an LLM to summarize. This
one treats research as a **loop with a stopping condition** and treats the draft
as a **hypothesis to be attacked**, not an answer to be trusted:

- It keeps researching until a *Knowledge Gap* agent says `COMPLETE` (or it hits
  the iteration/time budget) — so shallow questions stop fast and hard questions
  get more passes.
- It runs a *Devil's Advocate* against its own findings and **feeds the critique
  back in as new gaps** — a second-order research loop.
- It extracts the falsifiable claims from the final draft and **tries to refute
  each one** with a fresh skeptical search, defaulting to "unverified" when it
  can't find a supporting source. Shaky claims get flagged inline.

---

## Architecture

### The iterative knowledge-gap loop (8 steps)

```
                            ┌─────────────────────────────────────────────┐
                            │                 RESEARCH LOOP                │
   query ──▶ ❶ Knowledge ──▶│  while not COMPLETE and within budget:       │
              Gap Agent     │                                              │
                            │   ❷ Tool Selector ── picks sources per gap   │
                            │           │                                  │
                            │           ▼                                  │
                            │   ❸ Parallel Tool Execution (9 sources)      │
                            │      Brave · GitHub · Reddit · arXiv · HN ·  │
                            │      Semantic Scholar · StackOverflow ·      │
                            │      Wikipedia · X/Twitter                   │
                            │           │  └─▶ Firecrawl scrapes top URLs  │
                            │           ▼                                  │
                            │   dedupe ▶ source-credibility scoring ▶      │
                            │   embedding context compression              │
                            │           │                                  │
                            │           ▼                                  │
                            │   ❹ Observations Agent ── what did we learn? │
                            │           │                                  │
                            │   ❺ loop back to ❶ until COMPLETE / budget   │
                            └───────────┬──────────────────────────────────┘
                                        ▼
   ❻ Devil's Advocate ──▶ critique fed back as new gaps (up to N critique passes)
                                        ▼
       triangulation  +  citation-graph expansion  +  confidence scoring
                                        ▼
   ❼ Writer Agent ──▶ synthesizes the cited Markdown report
                                        ▼
   🛡  Per-claim adversarial verification  (the /deep-research Verify graft)
        extract claims ▶ N skeptical voters each ▶ 2/3 refutes kills a claim
        ▶ annotate report with a Claim Verification table + inline flags
                                        ▼
   ❽ Save the cited Markdown report
```

The eight numbered stages map to functions in
[`deep_research/core/loop.py`](deep_research/core/loop.py) and
[`deep_research/core/agents.py`](deep_research/core/agents.py):

| # | Stage | What it does |
|---|-------|--------------|
| ❶ | **Knowledge Gap** | Looks at the question + findings so far and asks "what don't we know yet?" Returns `COMPLETE` when nothing material is missing. |
| ❷ | **Tool Selector** | For each gap, picks which of the 9 sources to hit and writes the per-source query. |
| ❸ | **Parallel Tool Execution** | Fires all selected source queries concurrently (`asyncio.gather`), then **auto-scrapes** the top Brave URLs with Firecrawl for full-text. |
| ❹ | **Observations** | Distills the raw results into structured findings that feed the next gap analysis. |
| ❺ | **Loop** | Repeat ❶–❹ until `COMPLETE` or the iteration/time budget is spent. |
| ❻ | **Devil's Advocate** | Attacks the accumulated findings; its critique is fed back as fresh gaps for extra passes. |
| ❼ | **Writer** | Synthesizes everything into a cited report, with confidence, triangulation, and citation-graph context injected. |
| 🛡 | **Verify** | Per-claim 3-vote refute-or-survive fact-check (ported from `/deep-research`). |
| ❽ | **Save** | Writes the annotated Markdown report to your output directory. |

### The supporting machinery

Beyond the core loop, several subsystems make the output sharper:

- **9 parallel sources** ([`deep_research/tools/`](deep_research/tools/)) — Brave
  web search, GitHub, Reddit (official OAuth, recency-windowed), arXiv, Hacker
  News, Semantic Scholar, StackOverflow, Wikipedia, and X/Twitter
  (twitterapi.io). Each returns a uniform formatted-string shape so the loop is
  source-agnostic. Social sources (X, Reddit) bias hard toward a trailing ~90-day
  window, newest-first.

- **Firecrawl scraping** ([`tools/firecrawl_tool.py`](deep_research/tools/firecrawl_tool.py))
  — search snippets are thin, so the top web results are scraped to full Markdown
  and fed back into the same iteration. The number of pages scraped scales with
  depth (1 / 3 / 5 for shallow / standard / deep).

- **Multi-hop sub-query decomposition**
  ([`core/multi_hop.py`](deep_research/core/multi_hop.py)) — when a gap is really
  several questions ("compare A vs B on X, Y, and Z"), it's decomposed into
  parallel sub-queries, each researched independently, then synthesized — so the
  agent reasons across hops instead of one flat search.

- **Citation-graph expansion**
  ([`core/citations.py`](deep_research/core/citations.py)) — seed papers from the
  findings are expanded one hop through the Semantic Scholar citation graph to
  surface foundational and follow-on work the keyword search missed.

- **Source-credibility scoring**
  ([`core/source_curator.py`](deep_research/core/source_curator.py)) — every
  source is scored; low-quality ones are deprioritized before they reach the
  writer.

- **Embedding context compression**
  ([`core/context_compressor.py`](deep_research/core/context_compressor.py)) —
  raw results and accumulated findings are compressed *by relevance to the query*
  (embedding similarity) to a character budget, so deep runs with 90+ sources
  still fit the writer's context window without dumping noise.

- **Cross-source triangulation + confidence scoring**
  ([`core/triangulator.py`](deep_research/core/triangulator.py),
  [`core/confidence.py`](deep_research/core/confidence.py)) — claims corroborated
  by multiple independent sources are surfaced; the report ships a 0–100
  confidence score.

- **Per-claim adversarial verification** (the `/deep-research` graft,
  [`core/verifier.py`](deep_research/core/verifier.py)) — after the writer
  produces a draft, every falsifiable claim is extracted and handed to **N
  independent adversarial voters**. Each voter runs a *fresh* skeptical search
  (reusing the agent's own web + X + Reddit tooling) and tries to **refute** the
  claim. Quorum logic:
  - **REFUTED** — ≥ 2 of 3 voters find contradicting evidence.
  - **VERIFIED** — survives the vote *and* a voter produced a real supporting
    source URL.
  - **UNVERIFIED** — survives but nobody found a source (skeptical default:
    "we couldn't confirm this").

  The report gets a **Claim Verification table** plus inline `[UNVERIFIED]` /
  `[REFUTED]` flags so a reader sees what's solid at a glance.

- **Checkpoint + resume** ([`deep_research/vendor/checkpoint.py`](deep_research/vendor/checkpoint.py))
  — state is checkpointed after every iteration (atomic file writes). `--resume`
  picks up a long deep run exactly where it left off after a crash or restart.

### Two model profiles

The agent never hardcodes a model — every role (gap, selector, observations,
devil's advocate, writer, claim extractor, claim verifier) is mapped to a model
per profile in [`deep_research/models.py`](deep_research/models.py):

| Role | `multi` profile (cheap, default) | `opus` profile (premium) |
|------|----------------------------------|--------------------------|
| Knowledge Gap | Gemini 2.5 Flash | Claude Opus |
| Tool Selector | Gemini 2.5 Flash | Claude Sonnet |
| Observations | **local Ollama** | Claude Sonnet |
| Devil's Advocate | Gemini 2.5 Flash | Claude Opus |
| Writer | DeepSeek v3 | Claude Opus |
| Claim Extract/Verify | Gemini 2.5 Flash | Claude Sonnet |
| Routed via | OpenRouter (+ Ollama) | Claude SDK / Claude Code CLI |

- **`multi`** (cheap cloud + local) routes through **OpenRouter** for cloud models
  and offloads the observations step to a **local Ollama** model. If OpenRouter
  runs out of credits (402), it **transparently falls back to the direct Anthropic
  SDK**, substituting the cheapest equivalent Claude model and keeping cost
  tracking accurate.
- **`opus`** (premium) routes every role through the **Claude SDK / Claude Code
  CLI** for the best synthesis quality.

Every call is **cost-tracked** (tokens + USD per model) and the totals are written
into the report's frontmatter.

---

## Install

```bash
git clone https://github.com/JDDavenport/deep-research-agent.git
cd deep-research-agent

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e ".[all]"

cp .env.example .env                      # then fill in your keys
```

Only `httpx`, `requests`, and `python-dotenv` are hard requirements. The HTTP
server, Anthropic fallback, scheduled jobs, and HTML/PDF export are optional
extras — see `pyproject.toml`.

---

## Configure

Copy `.env.example` → `.env` and add keys. **Everything is optional except a way
to call an LLM** (`OPENROUTER_API_KEY` for the cheap profile). Recommended:

| Key | For | Notes |
|-----|-----|-------|
| `OPENROUTER_API_KEY` | the `multi` profile's cloud models | required for `multi` |
| `BRAVE_API_KEY` | web search | the backbone source; free tier available |
| `FIRECRAWL_API_KEY` | full-page scraping | thin snippets → full text |
| `TWITTERAPI_IO_KEY` | X/Twitter source | pennies per run; else sparse Brave fallback |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | Reddit source | free "script" app; else sparse Brave fallback |
| `GITHUB_TOKEN` | GitHub source | raises rate limits |
| `DEEP_RESEARCH_OUTPUT_DIR` | where reports are saved | defaults to `~/deep-research-reports` |

No keys are ever hardcoded — they all come from the environment.

---

## Usage

### CLI

```bash
# Quick, shallow pass
python -m deep_research.main "What is LangGraph?" --depth shallow

# Standard research (5 iterations, light claim verification)
python -m deep_research.main "best multi-agent orchestration patterns 2026"

# Deep research (10 iterations, full 3-vote adversarial verification)
python -m deep_research.main "state of autonomous agents" --depth deep --profile opus

# Cheap cloud profile, force per-claim verification on, export HTML
python -m deep_research.main "LangGraph vs CrewAI" --profile multi --verify-claims --format html

# Resume a crashed deep run exactly where it left off
python -m deep_research.main "state of autonomous agents" --depth deep --resume
```

Depth presets (set in [`config.py`](deep_research/config.py)):

| Depth | Iterations | Claim verification | Auto-scrape |
|-------|-----------|--------------------|-------------|
| `shallow` | 2 | off | 1 page |
| `standard` | 5 | light (1 vote, ≤12 claims) | 3 pages |
| `deep` | 10 | full (3 votes, ≤25 claims) | 5 pages |

### Library

```python
import asyncio
from deep_research.core.loop import run_research

result = asyncio.run(run_research(
    query="best multi-agent orchestration patterns 2026",
    depth="deep",
    profile="multi",
))

print(result.report)              # cited Markdown, with the Claim Verification table
print(result.confidence_level)    # e.g. "high"
print(result.cost_summary)        # tokens + USD, broken down by model
print(result.obsidian_path)       # where the .md was saved
```

### HTTP API (optional)

```bash
pip install -e ".[server]"
uvicorn deep_research.server:app --host 0.0.0.0 --port 3002

curl -X POST http://localhost:3002/research \
  -H "Content-Type: application/json" \
  -d '{"query": "latest AI research trends 2026", "depth": "standard"}'
```

### Scheduled jobs (optional)

```bash
pip install -e ".[schedule]"
python -m deep_research.core.scheduler --list      # writes an example config on first run
```

---

## Output

Reports are written as Markdown to `DEEP_RESEARCH_OUTPUT_DIR`
(default `~/deep-research-reports`) as `YYYY-MM-DD-<profile>-<slug>.md`, with YAML
frontmatter (query, date, iterations, sources, confidence, cost), the cited body,
the **Claim Verification table**, a full source list, and a per-model cost
breakdown. Point the output dir at an Obsidian vault or any synced folder to file
reports automatically.

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers the OpenRouter → Anthropic-direct fallback logic and the
per-claim verification quorum/adjudication (extraction, voting, refute/survive,
report annotation) with hermetic stubs — no network or API keys needed.

---

## Project layout

```
deep_research/
  config.py            depth presets, profiles, budgets, output dir
  models.py            model routing, cost tracking, OpenRouter↔Anthropic fallback
  main.py              CLI entry point
  server.py            optional FastAPI HTTP server
  mcp_server.py        optional MCP server (for Claude Code / MCP clients)
  core/
    loop.py            the 8-step research loop orchestrator
    agents.py          the gap / selector / observations / devil's-advocate / writer agents
    verifier.py        per-claim 3-vote adversarial verification  (ported from /deep-research)
    multi_hop.py       sub-query decomposition
    citations.py       Semantic Scholar citation-graph expansion
    triangulator.py    cross-source corroboration
    confidence.py      0–100 confidence scoring
    source_curator.py  source-credibility scoring
    context_compressor.py  embedding-based relevance compression
    formatter.py       markdown / html / pdf / json export
    cache.py           SQLite result cache
    progress.py        optional Telegram progress streaming
    scheduler.py       optional recurring-job runner
  tools/               the 9 sources + Firecrawl scraper
  vendor/              dependency-free vendored utilities (checkpoint, ollama client)
tests/                 fallback + verification tests
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

The per-claim **Verify** stage is ported from the **Claude Code `/deep-research`
skill**'s adversarial verification workflow (claim extraction + 3-vote
refute-or-survive, 2/3 to kill a claim). Thank you to that design.
