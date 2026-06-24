"""SQLite-based result caching for research queries.

Caches search results by query hash + depth to avoid redundant API calls.
Default TTL: 24 hours (configurable).
"""
import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Default cache location
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "research_cache.db"

# Default TTL in seconds (24 hours)
DEFAULT_TTL_SECONDS = 24 * 60 * 60


@dataclass
class CachedResult:
    """A cached research result."""
    query: str
    depth: str
    results_json: dict
    report_path: str
    cached_at: float  # Unix timestamp
    age_seconds: float  # How old the cache entry is

    @property
    def age_human(self) -> str:
        """Human-readable age string."""
        hours = self.age_seconds / 3600
        if hours < 1:
            return f"{int(self.age_seconds / 60)}m ago"
        return f"{hours:.1f}h ago"


class ResearchCache:
    """SQLite-backed research result cache.

    Usage:
        cache = ResearchCache()
        cached = cache.get_cached("AI agents", "standard")
        if cached:
            print(f"Cache hit ({cached.age_human})")
        else:
            # ... run research ...
            cache.save_cache("AI agents", "standard", results_dict, "/path/to/report.md")
    """

    def __init__(self, db_path: Optional[Path] = None, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.db_path = db_path or DEFAULT_CACHE_PATH
        self.ttl_seconds = ttl_seconds
        self._init_db()

    def _init_db(self):
        """Create the cache table if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_cache (
                    query_hash TEXT NOT NULL,
                    depth TEXT NOT NULL,
                    query TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    results_json TEXT NOT NULL,
                    report_path TEXT DEFAULT '',
                    PRIMARY KEY (query_hash, depth)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_timestamp
                ON research_cache(timestamp)
            """)

    def _conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))

    @staticmethod
    def _hash_query(query: str) -> str:
        """Create a stable hash of the query for cache lookup."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def get_cached(self, query: str, depth: str) -> Optional[CachedResult]:
        """Look up a cached result. Returns None on miss or expiry."""
        query_hash = self._hash_query(query)
        now = time.time()

        with self._conn() as conn:
            row = conn.execute(
                """SELECT query, depth, results_json, report_path, timestamp
                   FROM research_cache
                   WHERE query_hash = ? AND depth = ?""",
                (query_hash, depth),
            ).fetchone()

        if not row:
            return None

        cached_query, cached_depth, results_str, report_path, timestamp = row
        age = now - timestamp

        # Check TTL
        if age > self.ttl_seconds:
            # Expired — clean it up
            self._delete(query_hash, depth)
            return None

        try:
            results_json = json.loads(results_str)
        except json.JSONDecodeError:
            self._delete(query_hash, depth)
            return None

        return CachedResult(
            query=cached_query,
            depth=cached_depth,
            results_json=results_json,
            report_path=report_path,
            cached_at=timestamp,
            age_seconds=age,
        )

    def save_cache(
        self,
        query: str,
        depth: str,
        results: dict,
        report_path: str = "",
    ):
        """Save a research result to cache. Upserts on conflict."""
        query_hash = self._hash_query(query)
        now = time.time()

        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO research_cache
                   (query_hash, depth, query, timestamp, results_json, report_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (query_hash, depth, query, now, json.dumps(results), report_path),
            )

    def _delete(self, query_hash: str, depth: str):
        """Delete a cache entry."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM research_cache WHERE query_hash = ? AND depth = ?",
                (query_hash, depth),
            )

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed rows."""
        cutoff = time.time() - self.ttl_seconds
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM research_cache WHERE timestamp < ?",
                (cutoff,),
            )
            return cursor.rowcount

    def stats(self) -> dict:
        """Return cache statistics."""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM research_cache").fetchone()[0]
            cutoff = time.time() - self.ttl_seconds
            valid = conn.execute(
                "SELECT COUNT(*) FROM research_cache WHERE timestamp >= ?",
                (cutoff,),
            ).fetchone()[0]
            expired = total - valid

        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": expired,
            "db_path": str(self.db_path),
            "ttl_seconds": self.ttl_seconds,
        }
