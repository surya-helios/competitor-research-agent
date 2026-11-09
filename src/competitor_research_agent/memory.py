"""Remember which competitors a reviewer approved for a company, so the next
run for the same company, market, and region proposes the same ones instead of
running Discovery again. Stored in SQLite next to the search cache."""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterator

from .schemas import Candidate, CompanyProfile, ResearchConstraints

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approved_competitors (
    key TEXT PRIMARY KEY,
    company TEXT NOT NULL,
    competitors TEXT NOT NULL,
    run_id TEXT NOT NULL,
    approved_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class RememberedCompetitors:
    competitors: list[Candidate]
    run_id: str
    approved_at: datetime


def memory_key(company: CompanyProfile, constraints: ResearchConstraints) -> str:
    """The website identifies a company better than its name; market and
    region are part of the key because competitors differ by both."""
    identity = (company.domain or company.name).casefold().removeprefix("www.")
    return "|".join([identity, (constraints.market or "").casefold(), (constraints.region or "").casefold()])


class CompetitorMemory:
    def __init__(
        self,
        db_path: Path,
        max_age_days: int = 90,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_age = timedelta(days=max_age_days)
        self._now = now
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path, timeout=30, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def recall(self, company: CompanyProfile, constraints: ResearchConstraints) -> RememberedCompetitors | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT competitors, run_id, approved_at FROM approved_competitors WHERE key = ?",
                (memory_key(company, constraints),),
            ).fetchone()
        if row is None:
            return None
        approved_at = datetime.fromisoformat(row[2])
        if self._now() - approved_at > self._max_age:
            return None
        return RememberedCompetitors(
            [Candidate.model_validate(c) for c in json.loads(row[0])], row[1], approved_at
        )

    def remember(
        self,
        company: CompanyProfile,
        constraints: ResearchConstraints,
        competitors: list[Candidate],
        run_id: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO approved_competitors (key, company, competitors, run_id, approved_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    memory_key(company, constraints),
                    company.name,
                    json.dumps([c.model_dump(mode="json") for c in competitors]),
                    run_id,
                    self._now().isoformat(),
                ),
            )
