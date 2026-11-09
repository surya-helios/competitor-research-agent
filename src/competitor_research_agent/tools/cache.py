"""SQLite cache for Tavily responses, so repeated searches and reruns after a
crash don't spend credits again. Entries expire; failed calls are never cached.
Responses keep the time they were first fetched, so briefs show real retrieval
dates rather than the time of a cache hit.
"""

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_cache (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    response TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class CachedResponse:
    response: dict[str, Any]
    fetched_at: datetime


def cache_key(kind: str, params: dict[str, Any]) -> str:
    canonical = json.dumps({"kind": kind, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SearchCache:
    def __init__(
        self,
        db_path: Path,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def get(self, kind: str, params: dict[str, Any]) -> CachedResponse | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT response, fetched_at, expires_at FROM search_cache WHERE key = ?",
                (cache_key(kind, params),),
            ).fetchone()
        if row is None or datetime.fromisoformat(row[2]) <= self._now():
            return None
        return CachedResponse(json.loads(row[0]), datetime.fromisoformat(row[1]))

    def set(
        self, kind: str, params: dict[str, Any], response: dict[str, Any], ttl_hours: float
    ) -> datetime:
        fetched_at = self._now()
        expires_at = fetched_at + timedelta(hours=ttl_hours)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO search_cache"
                " (key, kind, params, response, fetched_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cache_key(kind, params),
                    kind,
                    json.dumps(params, sort_keys=True, default=str),
                    json.dumps(response),
                    fetched_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return fetched_at
