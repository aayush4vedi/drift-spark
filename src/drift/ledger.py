"""SQLite-backed lineage store for embed runs and per-embedding provenance."""

from __future__ import annotations
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / ".drift" / "ledger.db"

CREATE_RUNS = """
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

CREATE_PROVENANCE = """
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
        self._conn.executescript(CREATE_RUNS + ";" + CREATE_PROVENANCE)
        self._conn.commit()

    def write_run(self, run) -> None:
        """Persist an EmbedRun / WatchRun / MigrateRun to the ledger."""
        raise NotImplementedError("ledger.write_run() — wire in next iteration build session")

    def hash_exists(self, text_hash: str, model: str, sink: str) -> bool:
        """Return True if this text hash was already embedded with this model+sink."""
        raise NotImplementedError("ledger.hash_exists() — wire in next iteration build session")

    def cost_by_model(self):
        """Return total cost_usd grouped by model as a list of dicts."""
        cur = self._conn.execute(
            "SELECT model, SUM(cost_usd) as total FROM embed_runs GROUP BY model"
        )
        return [{"model": r[0], "cost_usd": r[1]} for r in cur.fetchall()]