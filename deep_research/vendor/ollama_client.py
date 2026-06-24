"""Minimal local-Ollama client used by the `multi` model profile.

Routes /api/generate calls to the first reachable Ollama backend, probing in
priority order:
    1. OLLAMA_REMOTE_URL  (e.g. a beefier box on your LAN, if you have one)
    2. OLLAMA_LOCAL_URL   (default: http://localhost:11434)

The chosen URL is cached for 30s per-process to avoid probing on every call.

This is a slimmed, dependency-free vendored copy — the original lived in a
larger agent framework. Only the `multi` profile's local-LLM worker step uses
it; the `claude`/`openrouter` paths never touch this file.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional

import requests

REMOTE_URL: Optional[str] = os.environ.get("OLLAMA_REMOTE_URL") or None
LOCAL_URL: Optional[str] = os.environ.get("OLLAMA_LOCAL_URL") or "http://localhost:11434"

DEFAULT_MODEL: str = (
    os.environ.get("OLLAMA_REMOTE_MODEL")
    or os.environ.get("OLLAMA_LOCAL_MODEL")
    or "llama3.1:8b"
)

CACHE_TTL_SEC: float = 30.0
_cached_url: Optional[str] = None
_cached_at: float = 0.0
_first_log_emitted: bool = False


class OllamaUnreachableError(RuntimeError):
    """Raised when no Ollama backend is reachable."""


def probe_ollama(url: str, timeout: float = 1.0) -> bool:
    """Cheap reachability probe: GET /api/tags, True on HTTP 200."""
    if not url:
        return False
    try:
        r = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _pick_url() -> Optional[str]:
    if REMOTE_URL and probe_ollama(REMOTE_URL):
        return REMOTE_URL
    if LOCAL_URL and probe_ollama(LOCAL_URL):
        return LOCAL_URL
    return None


def _log_active(url: str) -> None:
    global _first_log_emitted
    if _first_log_emitted:
        return
    print(f"[ollama_client] active={url} model={DEFAULT_MODEL}", file=sys.stderr)
    _first_log_emitted = True


def get_active_ollama_url() -> Optional[str]:
    """URL the router will use right now, or None if no backend is up (30s cache)."""
    global _cached_url, _cached_at
    now = time.time()
    if _cached_url is not None and (now - _cached_at) < CACHE_TTL_SEC:
        return _cached_url
    chosen = _pick_url()
    _cached_url = chosen
    _cached_at = now
    if chosen is not None:
        _log_active(chosen)
    return chosen


def _invalidate_cache() -> None:
    global _cached_url, _cached_at
    _cached_url = None
    _cached_at = 0.0


def _post_generate(url: str, *, prompt: str, model: str, system: Optional[str],
                   options: Optional[dict], timeout: float) -> str:
    payload = {"model": model, "prompt": prompt, "options": options or {}, "stream": False}
    if system is not None:
        payload["system"] = system
    r = requests.post(f"{url.rstrip('/')}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


def call_ollama(prompt: str, model: Optional[str] = None, *, system: Optional[str] = None,
                options: Optional[dict] = None, timeout: float = 120.0,
                cloud_fallback: bool = False) -> str:
    """Generate text via the first reachable Ollama backend.

    `cloud_fallback` is accepted for API compatibility but is a no-op in this
    vendored copy — callers in this package always pass cloud_fallback=False and
    handle the cloud fallback at a higher level (see models.query_llm).
    """
    chosen_model = model or DEFAULT_MODEL
    url = get_active_ollama_url()
    if url is not None:
        try:
            return _post_generate(url, prompt=prompt, model=chosen_model,
                                  system=system, options=options, timeout=timeout)
        except Exception as e:
            print(f"[ollama_client] mid-flight failure on {url}: {e} — re-routing",
                  file=sys.stderr)
            _invalidate_cache()
            url = get_active_ollama_url()
            if url is not None:
                try:
                    return _post_generate(url, prompt=prompt, model=chosen_model,
                                          system=system, options=options, timeout=timeout)
                except Exception as e2:
                    print(f"[ollama_client] reroute also failed on {url}: {e2}",
                          file=sys.stderr)
                    _invalidate_cache()
    raise OllamaUnreachableError(
        f"No Ollama backend reachable (remote={REMOTE_URL}, local={LOCAL_URL})."
    )
