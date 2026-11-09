"""Per-run spend ledger, stored in SQLite.

Every paid call reserves its worst-case cost before it runs and settles the
actual cost afterwards. Reserving checks the limit and records the row in one
transaction, so parallel branches can't jointly overspend. Rows are written
immediately, so spend survives a crash even if the graph step doesn't finish.

Amounts are in each provider's unit: credits for Tavily, USD for Nebius.
"""

import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

Provider = Literal["tavily", "nebius"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    node TEXT NOT NULL,
    model TEXT,
    reserved REAL NOT NULL,
    actual REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_run ON ledger (run_id, provider);
"""


class BudgetExceeded(Exception):
    def __init__(self, provider: Provider, requested: float, remaining: float):
        super().__init__(
            f"{provider} budget exceeded: requested {requested:g}, remaining {remaining:g}"
        )
        self.provider = provider
        self.requested = requested
        self.remaining = remaining


@dataclass(frozen=True)
class SpendRow:
    node: str
    provider: Provider
    model: str | None
    calls: int
    amount: float


def llm_cost_usd(
    input_tokens: int, output_tokens: int, usd_per_m_input: float, usd_per_m_output: float
) -> float:
    return (input_tokens * usd_per_m_input + output_tokens * usd_per_m_output) / 1_000_000


class BudgetLedger:
    def __init__(self, db_path: Path, limits: dict[Provider, float]):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._limits = limits
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

    @staticmethod
    def _spent(conn: sqlite3.Connection, run_id: str, provider: Provider) -> float:
        row = conn.execute(
            "SELECT COALESCE(SUM(COALESCE(actual, reserved)), 0) FROM ledger"
            " WHERE run_id = ? AND provider = ?",
            (run_id, provider),
        ).fetchone()
        return float(row[0])

    def reserve(
        self,
        run_id: str,
        provider: Provider,
        amount: float,
        node: str,
        model: str | None = None,
    ) -> int:
        """Record a worst-case charge, or raise BudgetExceeded. Returns the row id
        to pass to settle()."""
        # The lock covers threads in this process; BEGIN IMMEDIATE covers others.
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                remaining = self._limits[provider] - self._spent(conn, run_id, provider)
                if amount > remaining:
                    raise BudgetExceeded(provider, amount, remaining)
                cursor = conn.execute(
                    "INSERT INTO ledger (run_id, provider, node, model, reserved, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, provider, node, model, amount, datetime.now(timezone.utc).isoformat()),
                )
                entry_id = int(cursor.lastrowid)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return entry_id

    def settle(
        self,
        entry_id: int,
        actual: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Replace a reservation with the actual cost. Use 0 for a call that
        failed without being charged."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE ledger SET actual = ?, input_tokens = ?, output_tokens = ? WHERE id = ?",
                (actual, input_tokens, output_tokens, entry_id),
            )

    def spent(self, run_id: str, provider: Provider) -> float:
        with self._connect() as conn:
            return self._spent(conn, run_id, provider)

    def remaining(self, run_id: str, provider: Provider) -> float:
        return self._limits[provider] - self.spent(run_id, provider)

    def summary(self, run_id: str) -> list[SpendRow]:
        """Spend grouped by node, provider, and model, for the UI."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT node, provider, model, COUNT(*), SUM(COALESCE(actual, reserved))"
                " FROM ledger WHERE run_id = ? GROUP BY node, provider, model"
                " ORDER BY provider, node",
                (run_id,),
            ).fetchall()
        return [SpendRow(*row) for row in rows]
