"""Embedding-based context compression using local Ollama embeddings.

Chunks raw search results into ~500-token segments, scores each chunk against
the research query via cosine similarity, and returns only the top-K most
relevant chunks. Falls back to simple truncation if Ollama is unavailable.

Target: 3x more relevant content in the same context budget.
"""
import os
import re
import math
import httpx
from typing import Optional


# ── Configuration ──────────────────────────────────────────────────────────
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE_CHARS = 2000         # ~500 tokens at 4 chars/token
CHUNK_OVERLAP_CHARS = 200       # small overlap to preserve context across chunk boundaries
DEFAULT_TOP_K = 15              # number of chunks to keep
OLLAMA_TIMEOUT = 30.0           # seconds per embedding request

# Ollama availability cache (avoid repeated failed connections)
_ollama_available: Optional[bool] = None


async def compress_context(
    raw_text: str,
    query: str,
    max_chars: int = 8000,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Compress raw_text to the most relevant chunks for the given query.

    Args:
        raw_text: raw search results / tool output
        query: the research query to score relevance against
        max_chars: maximum character budget for the output
        top_k: maximum number of chunks to consider

    Returns:
        Compressed text containing only the most relevant chunks, or
        a simple truncation if embeddings are unavailable.
    """
    if len(raw_text) <= max_chars:
        return raw_text

    chunks = _chunk_text(raw_text)
    if not chunks:
        return raw_text[:max_chars]

    # Try embedding-based compression
    if await _is_ollama_available():
        try:
            scored = await _score_chunks(chunks, query)
            return _assemble_top_k(scored, max_chars, top_k)
        except Exception as e:
            print(f"   Warning: Embedding compression failed ({e}), falling back to truncation")

    # Fallback: simple truncation with source-header awareness
    return _smart_truncate(raw_text, max_chars)


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings from Ollama for a list of texts.

    Useful for deduplication and other downstream tasks.
    Raises an exception if Ollama is unavailable.
    """
    base_url = os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434")
    embeddings = []

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT, follow_redirects=True) as client:
        for text in texts:
            response = await client.post(
                f"{base_url}/api/embed",
                json={"model": EMBED_MODEL, "input": text},
            )
            response.raise_for_status()
            data = response.json()
            # Ollama returns {"embeddings": [[...]]} for /api/embed
            emb = data.get("embeddings", [[]])[0]
            embeddings.append(emb)

    return embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Pure Python, no numpy needed."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Internal helpers ───────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    """Split text into roughly CHUNK_SIZE_CHARS segments with overlap.

    Tries to split on double-newlines or source-block boundaries first.
    """
    # Split on source block boundaries first
    block_pattern = r'(===\s*(?:Results from|Auto-scraped:)\s*.+?\s*===)'
    blocks = re.split(block_pattern, text)

    # Reassemble into header+content pairs
    sections = []
    current = ""
    for part in blocks:
        if re.match(block_pattern, part.strip()):
            if current.strip():
                sections.append(current.strip())
            current = part
        else:
            current += part

    if current.strip():
        sections.append(current.strip())

    # Now chunk each section if it's too large
    chunks = []
    for section in sections:
        if len(section) <= CHUNK_SIZE_CHARS:
            chunks.append(section)
        else:
            # Sub-chunk on paragraph boundaries
            paragraphs = re.split(r'\n\n+', section)
            current_chunk = ""
            for para in paragraphs:
                # If a single paragraph is bigger than chunk size, force-split it
                if len(para) > CHUNK_SIZE_CHARS:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    for start in range(0, len(para), CHUNK_SIZE_CHARS - CHUNK_OVERLAP_CHARS):
                        chunks.append(para[start:start + CHUNK_SIZE_CHARS])
                    continue

                if len(current_chunk) + len(para) + 2 > CHUNK_SIZE_CHARS and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Keep overlap from end of previous chunk
                    overlap = current_chunk[-CHUNK_OVERLAP_CHARS:] if len(current_chunk) > CHUNK_OVERLAP_CHARS else ""
                    current_chunk = overlap + "\n\n" + para
                else:
                    current_chunk += "\n\n" + para if current_chunk else para

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

    return chunks


async def _is_ollama_available() -> bool:
    """Check if Ollama is running and has the embedding model."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available

    base_url = os.environ.get("OLLAMA_LOCAL_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
            r = await client.get(f"{base_url}/api/tags")
            if r.is_success:
                models = [m["name"] for m in r.json().get("models", [])]
                _ollama_available = any(EMBED_MODEL in m for m in models)
            else:
                _ollama_available = False
    except Exception:
        _ollama_available = False

    if not _ollama_available:
        print(f"   Info: Ollama embedding model '{EMBED_MODEL}' not available, using truncation fallback")

    return _ollama_available


async def _score_chunks(chunks: list[str], query: str) -> list[tuple[float, str]]:
    """Score each chunk against the query using cosine similarity."""
    # Embed the query and all chunks in batch
    all_texts = [query] + chunks
    embeddings = await get_embeddings(all_texts)

    query_emb = embeddings[0]
    chunk_embs = embeddings[1:]

    scored = []
    for chunk, emb in zip(chunks, chunk_embs):
        sim = cosine_similarity(query_emb, emb)
        scored.append((sim, chunk))

    # Sort by similarity descending
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _assemble_top_k(scored: list[tuple[float, str]], max_chars: int, top_k: int) -> str:
    """Take the top-K scored chunks and assemble within the character budget."""
    selected = []
    total_chars = 0

    for sim, chunk in scored[:top_k]:
        if total_chars + len(chunk) + 2 > max_chars:
            # Add a partial chunk if we have room
            remaining = max_chars - total_chars - 50
            if remaining > 200:
                selected.append(f"[relevance: {sim:.3f}]\n{chunk[:remaining]}...")
            break
        selected.append(f"[relevance: {sim:.3f}]\n{chunk}")
        total_chars += len(chunk) + 20  # account for the relevance label

    return "\n\n".join(selected)


def _smart_truncate(text: str, max_chars: int) -> str:
    """Truncate text while preserving source-block headers for context."""
    if len(text) <= max_chars:
        return text

    # Keep the beginning (usually most important results) and note truncation
    cutoff = max_chars - 80
    # Try to cut at a paragraph boundary
    last_break = text.rfind("\n\n", 0, cutoff)
    if last_break > cutoff * 0.7:
        cutoff = last_break

    return text[:cutoff] + f"\n\n[Truncated: {len(text) - cutoff} chars omitted for context budget]"
