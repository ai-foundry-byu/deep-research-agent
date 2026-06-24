"""Model profiles and routing for research agents.

Two profiles:
  - opus: Premium. Uses Claude Agent SDK (claude code CLI under the hood).
          Opus orchestrator + Sonnet workers. No API key needed — uses CLI auth.
  - multi: Budget. Uses OpenRouter for Gemini Flash + DeepSeek, Ollama for local.
"""
import httpx
import os
import sys
import asyncio
import json
import time
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Load .env from the repo root (one level above the package).
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

from .vendor.ollama_client import call_ollama as _call_ollama_sync  # noqa: E402

# ── Model cost table (per 1M tokens) ────────────────────────────────────
MODEL_COSTS = {
    "claude-opus-4":                 {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4":               {"input": 3.00,  "output": 15.00},
    "claude-haiku-4":                {"input": 1.00,  "output": 5.00},
    # Anthropic 4.5 series — used by direct-SDK fallback. Same public pricing as -4.
    "claude-opus-4-5":               {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-5":             {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":              {"input": 1.00,  "output": 5.00},
    "google/gemini-2.5-flash":       {"input": 0.15,  "output": 0.60},
    "deepseek/deepseek-chat-v3":     {"input": 0.14,  "output": 0.28},
    "local/ollama":                  {"input": 0.00,  "output": 0.00},
}


# ── Anthropic-direct fallback model substitution ────────────────────────
# When OpenRouter is down (402 / insufficient credits), we fall back to direct
# Anthropic SDK calls. Non-Anthropic models (gemini, deepseek, ollama) get
# substituted to the cheapest Claude equivalent.
_ANTHROPIC_DIRECT_MODEL_MAP = {
    # Already-Anthropic models — promote -4 → -4-5 (the SDK-named flavors)
    "claude-opus-4":               "claude-opus-4-5",
    "claude-sonnet-4":             "claude-sonnet-4-5",
    "claude-haiku-4":              "claude-haiku-4-5",
    # Non-Anthropic OpenRouter models — substitute cheapest Claude
    "google/gemini-2.5-flash":     "claude-haiku-4-5",
    "deepseek/deepseek-chat-v3":   "claude-haiku-4-5",
    "local/ollama":                "claude-haiku-4-5",
}


def _substitute_for_anthropic_direct(model: str) -> str:
    """Map a researcher model id to the closest direct-Anthropic equivalent."""
    if model in _ANTHROPIC_DIRECT_MODEL_MAP:
        return _ANTHROPIC_DIRECT_MODEL_MAP[model]
    # Anything that already looks like a direct-SDK Claude id passes through.
    if model.startswith("claude-"):
        return model
    # Any other openrouter slug (provider/model) → safest is haiku.
    return "claude-haiku-4-5"


def _resolve_anthropic_api_key() -> str | None:
    """Return the first usable direct-API Anthropic key (NOT an OAuth token).

    OAuth tokens (sk-ant-oat...) belong to the Claude Code CLI and CANNOT be
    used by anthropic.Anthropic() — they 401 immediately. By convention
    ANTHROPIC_API_KEY_SUB1 holds a funded direct-API key if you keep your
    OAuth token in ANTHROPIC_API_KEY (e.g. for the Claude Code CLI).
    """
    candidates = (
        os.getenv("ANTHROPIC_API_KEY_SUB1"),
        os.getenv("ANTHROPIC_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY_SUB2"),
    )
    for key in candidates:
        if key and not key.startswith("sk-ant-oat"):
            return key
    return None

# ── Role → model mapping per profile ────────────────────────────────────
PROFILES = {
    "opus": {
        "knowledge_gap":    "claude-opus-4",
        "tool_selector":    "claude-sonnet-4",
        "observations":     "claude-sonnet-4",
        "devils_advocate":  "claude-opus-4",
        "writer":           "claude-opus-4",
        # claim extraction + per-claim adversarial verification (deep-research graft).
        # Sonnet: high-volume, parallel, cheaper than Opus — extraction/verification
        # are bounded judgement calls, not open-ended synthesis.
        "claim_extractor":  "claude-sonnet-4",
        "claim_verifier":   "claude-sonnet-4",
        "_provider":        "claude_sdk",      # route via Claude Agent SDK
    },
    "multi": {
        "knowledge_gap":    "google/gemini-2.5-flash",
        "tool_selector":    "google/gemini-2.5-flash",
        "observations":     "local/ollama",
        "devils_advocate":  "google/gemini-2.5-flash",
        "writer":           "deepseek/deepseek-chat-v3",
        "claim_extractor":  "google/gemini-2.5-flash",
        "claim_verifier":   "google/gemini-2.5-flash",
        "_provider":        "openrouter",
    },
}

# Fallback for local/ollama failures
OLLAMA_CLOUD_FALLBACK = "google/gemini-2.5-flash"


@dataclass
class CostTracker:
    """Tracks token usage and cost across a research session."""
    calls: list = field(default_factory=list)

    def record(self, role: str, model: str, input_tokens: int, output_tokens: int, latency_ms: int):
        costs = MODEL_COSTS.get(model, {"input": 0, "output": 0})
        cost = (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000
        self.calls.append({
            "role": role,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
            "latency_ms": latency_ms,
        })

    @property
    def total_cost(self) -> float:
        return round(sum(c["cost_usd"] for c in self.calls), 4)

    @property
    def total_input_tokens(self) -> int:
        return sum(c["input_tokens"] for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c["output_tokens"] for c in self.calls)

    def summary(self) -> dict:
        return {
            "total_cost_usd": self.total_cost,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "calls": len(self.calls),
            "by_model": self._by_model(),
        }

    def _by_model(self) -> dict:
        models = {}
        for c in self.calls:
            m = c["model"]
            if m not in models:
                models[m] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            models[m]["calls"] += 1
            models[m]["cost_usd"] = round(models[m]["cost_usd"] + c["cost_usd"], 6)
            models[m]["tokens"] += c["input_tokens"] + c["output_tokens"]
        return models


# ── Claude Agent SDK ────────────────────────────────────────────────────
async def query_claude_sdk(prompt: str, model: str = "claude-sonnet-4", max_tokens: int = 4096) -> tuple[str, int, int]:
    """Query via Claude Code CLI with JSON output. Uses CLI auth — no API key needed.
    Returns (text, input_tokens, output_tokens).

    Model names: 'opus', 'sonnet', 'haiku' (CLI shorthand).
    """
    import subprocess

    # Map full model names to CLI shorthand
    model_map = {
        "claude-opus-4": "opus",
        "claude-sonnet-4": "sonnet",
        "claude-haiku-4": "haiku",
    }
    cli_model = model_map.get(model, model)

    # Write prompt to temp file to avoid shell escaping issues
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        # Build clean env — remove ANTHROPIC_API_KEY so CLI uses its own OAuth auth
        clean_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        # Use shell pipe: cat file | claude -p -
        # Run from /tmp to avoid loading CLAUDE.md (saves ~17K tokens per call)
        # --disallowed-tools '*' kills all tool access — these are pure-text synthesis
        # calls, never agentic. Without this, Claude can decide to WebFetch a URL
        # mentioned in the prompt, hit a permission error, set is_error=true, and
        # return empty result. Caught 2026-05-08 on CR-2 LinkedIn growth research.
        # --disallowedTools enumerated by name — '*' isn't a valid wildcard for
        # this flag. We list every default tool Claude CLI exposes so the writer
        # can't decide to WebFetch a URL from the prompt and crash on permission.
        _NO_TOOLS = (
            "Bash Edit Write Read NotebookEdit WebFetch WebSearch "
            "Glob Grep Task TodoWrite ExitPlanMode KillShell BashOutput "
            "SlashCommand Skill"
        )
        proc = await asyncio.create_subprocess_shell(
            f'cat "{prompt_file}" | claude -p - --model {cli_model} --output-format json --max-turns 1 --disallowedTools {_NO_TOOLS}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=clean_env,
            cwd="/tmp",  # run from /tmp so the CLI doesn't load a project CLAUDE.md
        )
        # Writer step on deep runs synthesizes 90+ sources into 8192 output tokens —
        # the 5-min limit was tripping the writer. Bumped to 15 min. Other roles
        # finish in <60s so they don't pay the bigger budget.
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""
    finally:
        os.unlink(prompt_file)

    if proc.returncode != 0 and not stdout_str.strip().startswith("{"):
        raise Exception(f"Claude CLI error (exit {proc.returncode}): {stderr_str[:300] or stdout_str[:300]}")

    try:
        data = json.loads(stdout_str)
    except json.JSONDecodeError:
        # Sometimes output has non-JSON prefix; try to find the JSON object
        start = stdout_str.find("{")
        if start >= 0:
            data = json.loads(stdout_str[start:])
        else:
            raise Exception(f"Claude CLI returned non-JSON: {stdout_str[:200]}")

    if data.get("is_error") and not data.get("result"):
        raise Exception(f"Claude CLI returned error with no result. stderr={stderr_str[:200]}")
    # If is_error but has result text, treat as success (e.g. permission denied on tools but text generated)

    text = data.get("result", "")
    usage = data.get("usage", {})
    in_tok = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
    out_tok = usage.get("output_tokens", 0)

    return text, in_tok, out_tok


# ── Ollama ──────────────────────────────────────────────────────────────
async def _get_ollama_model(base_url: str) -> str:
    """Get the best available Ollama model, checking what's actually installed."""
    preferred = ["qwen2.5:7b", "qwen3:8b", "llama3.1:8b", "gemma3:12b", "deepseek-r1:7b"]
    env_model = os.environ.get("OLLAMA_LOCAL_MODEL", "")

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.is_success:
                installed = [m["name"] for m in r.json().get("models", [])]
                if env_model and env_model in installed:
                    return env_model
                for p in preferred:
                    if p in installed:
                        return p
                if installed:
                    return installed[0]
    except Exception:
        pass

    return env_model or "qwen2.5:7b"


async def query_ollama(prompt: str, model: str = None, timeout: float = 120.0) -> tuple[str, int, int]:
    """Query Ollama via the shared smart-router. Returns (text, input_tokens_est, output_tokens_est).

    Routing (MBP → Mac Mini → fail) and the 30s probe cache live in
    the vendored `ollama_client`. We still pick a sensible model name here
    if the caller didn't specify one, so we keep the "use what's installed"
    behavior of the prior code path.
    """
    # Pick a model name. Use the caller's choice, else discover the best installed
    # model on whichever host the router is currently using.
    model_name = model
    if not model_name:
        from .vendor.ollama_client import get_active_ollama_url
        active = get_active_ollama_url()
        if active is None:
            raise Exception("Ollama not available (no reachable backend)")
        model_name = await _get_ollama_model(active)

    # call_ollama is sync; bridge to async via to_thread so we don't block the loop.
    try:
        text = await asyncio.to_thread(
            _call_ollama_sync,
            prompt,
            model_name,
            system=None,
            options=None,
            timeout=timeout,
            cloud_fallback=False,
        )
    except Exception as e:
        raise Exception(f"Ollama not available: {e}")

    # No raw token counts back from call_ollama — estimate at 4 chars/token,
    # matching the prior fallback behavior when the keys were absent.
    input_tok = len(prompt) // 4
    output_tok = len(text) // 4
    return text, input_tok, output_tok


# ── Anthropic direct (fallback when OpenRouter 402s) ────────────────────
async def query_anthropic_direct(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
) -> tuple[str, int, int, str]:
    """Direct call to Anthropic's API via the official SDK.

    Used as a fallback when OpenRouter returns 402 (insufficient credits) or
    is otherwise unreachable. Returns (text, input_tokens, output_tokens, model_used).

    Non-Anthropic models (gemini, deepseek, ollama) are substituted to the
    cheapest Claude equivalent — see `_substitute_for_anthropic_direct`.
    """
    import anthropic  # local import; only loaded on fallback

    api_key = _resolve_anthropic_api_key()
    if not api_key:
        raise Exception(
            "Anthropic-direct fallback unavailable: no usable ANTHROPIC_API_KEY "
            "(need ANTHROPIC_API_KEY_SUB1 or a non-OAuth ANTHROPIC_API_KEY)"
        )

    target_model = _substitute_for_anthropic_direct(model)

    client = anthropic.Anthropic(api_key=api_key)

    def _create(model_id: str):
        return client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

    try:
        resp = await asyncio.to_thread(_create, target_model)
        used_model = target_model
    except anthropic.NotFoundError:
        # Some accounts don't have every -4-5 alias; sonnet is the safest universal.
        print(f"   ⚠️  Anthropic model {target_model} not found — retrying with claude-sonnet-4-5")
        resp = await asyncio.to_thread(_create, "claude-sonnet-4-5")
        used_model = "claude-sonnet-4-5"

    # Concatenate text blocks (anthropic returns a list of content blocks).
    text_parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    text = "".join(text_parts).strip()

    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    return text, in_tok, out_tok, used_model


# ── OpenRouter ──────────────────────────────────────────────────────────
async def query_openrouter(
    prompt: str,
    model: str,
    max_tokens: int = 4096,
    max_retries: int = 3,
) -> tuple[str, int, int, str]:
    """Query OpenRouter. Returns (text, input_tokens, output_tokens, model_used).

    On OpenRouter 402 (insufficient credits) — or after all retries fail for
    any reason — falls back to direct Anthropic SDK via `query_anthropic_direct`.
    The model_used field reflects whichever model actually produced the text,
    so cost tracking stays accurate when fallback substitutes (e.g.
    google/gemini-2.5-flash → claude-haiku-4-5).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        # No OpenRouter at all → go straight to Anthropic.
        print("   ⚠️  OPENROUTER_API_KEY not set — using direct Anthropic SDK")
        return await query_anthropic_direct(prompt, model, max_tokens=max_tokens)

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://github.com/JDDavenport/deep-research-agent",
                        "X-Title": "Research Agent",
                    },
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                    },
                )

                # 402 = insufficient credits. Retrying won't help — fall back NOW.
                if response.status_code == 402:
                    print(
                        "   ⚠️  OpenRouter 402 (insufficient credits) — "
                        "falling back to direct Anthropic SDK"
                    )
                    return await query_anthropic_direct(
                        prompt, model, max_tokens=max_tokens
                    )

                if not response.is_success:
                    if attempt < max_retries - 1:
                        wait = (2 ** attempt) * 2
                        print(f"   ⏳ API {response.status_code}, retry in {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    last_err = Exception(
                        f"OpenRouter {response.status_code}: {response.text[:300]}"
                    )
                    break  # retries exhausted → try fallback below

                data = response.json()
                if "choices" in data and data["choices"]:
                    text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return (
                        text,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        model,
                    )
                elif "error" in data:
                    last_err = Exception(f"API error: {data['error']}")
                    break
                else:
                    last_err = Exception(f"Unexpected response: {str(data)[:200]}")
                    break

        except httpx.TimeoutException as e:
            last_err = Exception("LLM request timed out after retries")
            if attempt < max_retries - 1:
                await asyncio.sleep((2 ** attempt) * 2)
                continue
            break
        except httpx.HTTPError as e:
            last_err = Exception(f"HTTP error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep((2 ** attempt) * 2)
                continue
            break

    # All retries exhausted (or non-recoverable error). Try direct Anthropic.
    print(
        f"   ⚠️  OpenRouter failed ({last_err}) — falling back to direct Anthropic SDK"
    )
    try:
        return await query_anthropic_direct(prompt, model, max_tokens=max_tokens)
    except Exception as fallback_err:
        # Surface the original OpenRouter error AND the fallback error so the caller sees both.
        raise Exception(
            f"Both OpenRouter and Anthropic-direct failed. "
            f"OpenRouter: {last_err}. Anthropic-direct: {fallback_err}"
        )


# ── Unified router ──────────────────────────────────────────────────────
async def query_llm(
    prompt: str,
    role: str,
    profile: str = "opus",
    cost_tracker: CostTracker = None,
    max_tokens: int = 4096,
) -> str:
    """Route a prompt to the right model based on profile and role.

    - opus profile → Claude Agent SDK (uses CLI auth, Opus/Sonnet models)
    - multi profile → OpenRouter for cloud, Ollama for local
    """
    profile_config = PROFILES.get(profile, PROFILES["opus"])
    model = profile_config.get(role, "claude-sonnet-4")
    provider = profile_config.get("_provider", "openrouter")

    start_ms = int(time.time() * 1000)

    # Local Ollama path
    if model == "local/ollama":
        try:
            text, in_tok, out_tok = await query_ollama(prompt)
            latency = int(time.time() * 1000) - start_ms
            if cost_tracker:
                cost_tracker.record(role, "local/ollama", in_tok, out_tok, latency)
            return text
        except Exception as e:
            print(f"   ⚠️  Ollama failed ({e}), falling back to cloud...")
            model = OLLAMA_CLOUD_FALLBACK
            provider = "openrouter"

    # Claude Agent SDK path (opus profile)
    if provider == "claude_sdk":
        text, in_tok, out_tok = await query_claude_sdk(prompt, model, max_tokens=max_tokens)
        latency = int(time.time() * 1000) - start_ms
        if cost_tracker:
            cost_tracker.record(role, model, in_tok, out_tok, latency)
        return text

    # OpenRouter path (multi profile + ollama fallbacks).
    # query_openrouter may transparently fall back to direct Anthropic SDK when
    # OpenRouter 402s — model_used reflects whichever model actually produced
    # the text so cost tracking stays accurate.
    text, in_tok, out_tok, model_used = await query_openrouter(
        prompt, model, max_tokens=max_tokens
    )
    latency = int(time.time() * 1000) - start_ms
    if cost_tracker:
        cost_tracker.record(role, model_used, in_tok, out_tok, latency)
    return text
