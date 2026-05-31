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


_CREATE_WATCH_RUNS = """
CREATE TABLE IF NOT EXISTS watch_runs (
    run_id        TEXT PRIMARY KEY,
    timestamp     TEXT,
    source_table  TEXT,
    sink          TEXT,
    since_version INTEGER,
    to_version    INTEGER,
    n_inserted    INTEGER,
    n_updated     INTEGER,
    n_deleted     INTEGER,
    duration_s    REAL
)"""


class Ledger:
    def __init__(self, db_path: Path = DEFAULT_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(
            _CREATE_RUNS + ";" + _CREATE_PROVENANCE + ";" + _CREATE_WATCH_RUNS
        )
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

    def provenance(self, embedding_id: str) -> dict | None:
        """
        Full lineage for a single vector: what text produced it, with which
        model, into which sink, at what run cost, and when.

        Returns None if the embedding_id is not in the ledger (e.g. it was
        embedded before Drift was introduced, or the ledger was wiped).

        The cost_usd returned is the total cost of the run that produced this
        vector — not the per-vector cost (which isn't tracked at v0.1 granularity).
        """
        cur = self._conn.execute(
            """SELECT p.embedding_id, p.source_pk, p.source_hash, p.created_at,
                      r.model, r.sink, r.cost_usd, r.timestamp AS run_timestamp
               FROM embedding_provenance p
               JOIN embed_runs r ON p.run_id = r.run_id
               WHERE p.embedding_id = ?
               LIMIT 1""",
            (embedding_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = ["embedding_id", "source_pk", "source_hash", "created_at",
                "model", "sink", "cost_usd", "run_timestamp"]
        return dict(zip(cols, row))

    def write_watch_run(self, run) -> None:
        """Persist a WatchRun to the ledger (checkpoint for next watch() call)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO watch_runs
               VALUES (:run_id, :timestamp, :source_table, :sink, :since_version,
                       :to_version, :n_inserted, :n_updated, :n_deleted, :duration_s)""",
            vars(run),
        )
        self._conn.commit()

    def last_watch_version(self, source_table: str, sink: str) -> int | None:
        """
        Return the to_version of the most recent watch() run for this
        (source_table, sink) pair — the starting point for the next incremental run.
        Returns None if no prior watch() run exists (first run → caller uses 0).
        """
        cur = self._conn.execute(
            """SELECT to_version FROM watch_runs
               WHERE source_table = ? AND sink = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (source_table, sink),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()
