"""SQLite-backed lineage store for embed runs and per-embedding provenance."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / ".drift" / "ledger.db"

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS embed_runs (
    run_id      TEXT PRIMARY KEY,
    timestamp   TEXT,
    model       TEXT,
    sink        TEXT,
    n_rows      INTEGER,
    n_deduped   INTEGER,
    cost_usd    REAL,
    duration_s  REAL
)"""

_CREATE_PROVENANCE = """
CREATE TABLE IF NOT EXISTS embedding_provenance (
    embedding_id TEXT PRIMARY KEY,
    source_pk    TEXT,
    source_hash  TEXT,
    run_id       TEXT REFERENCES embed_runs(run_id),
    created_at   TEXT
)"""


class Ledger:
    def __init__(self, db_path: Path = DEFAULT_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_CREATE_RUNS + ";" + _CREATE_PROVENANCE)
        self._conn.commit()

    def write_run(self, run) -> None:
        """Persist an EmbedRun to the ledger."""
        self._conn.execute(
            """INSERT OR REPLACE INTO embed_runs
               VALUES (:run_id,:timestamp,:model,:sink,:n_rows_processed,
                       :n_rows_deduped,:cost_usd,:duration_s)""",
            vars(run),
        )
        self._conn.commit()

    def hash_exists(self, text_hash: str, model: str, sink: str) -> bool:
        """True if this text hash was already embedded with this (model, sink) pair."""
        cur = self._conn.execute(
            """SELECT 1 FROM embedding_provenance p
               JOIN embed_runs r ON p.run_id = r.run_id
               WHERE p.source_hash = ? AND r.model = ? AND r.sink = ?
               LIMIT 1""",
            (text_hash, model, sink),
        )
        return cur.fetchone() is not None

    def write_provenance(self, *, embedding_id: str, source_pk: str,
                         source_hash: str, run_id: str, created_at: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO embedding_provenance VALUES (?,?,?,?,?)",
            (embedding_id, source_pk, source_hash, run_id, created_at),
        )
        self._conn.commit()

    def cost_by_model(self) -> list[dict]:
        """Total cost_usd grouped by model."""
        cur = self._conn.execute(
            "SELECT model, SUM(cost_usd) FROM embed_runs GROUP BY model"
        )
        return [{"model": r[0], "cost_usd": r[1]} for r in cur.fetchall()]

    def recent_runs(self, sink: str, limit: int = 5) -> list[dict]:
        """Last N runs for a given sink."""
        cur = self._conn.execute(
            """SELECT run_id, timestamp, model, n_rows, n_deduped, cost_usd, duration_s
               FROM embed_runs WHERE sink = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (sink, limit),
        )
        cols = ["run_id", "timestamp", "model", "n_rows", "n_deduped", "cost_usd", "duration_s"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
