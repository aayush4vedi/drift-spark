"""Tests for the SQLite-backed Ledger — all public methods."""

import uuid

import pytest

from drift.embed import EmbedRun
from drift.ledger import Ledger
from drift.watch import WatchRun

# ── helpers ───────────────────────────────────────────────────────────────────

def _run(
    model: str = "openai/text-embedding-3-small",
    sink: str = "qdrant://localhost/col",
    n_rows: int = 5,
    n_deduped: int = 0,
    cost: float = 0.0,
) -> EmbedRun:
    return EmbedRun(
        model=model,
        sink=sink,
        n_rows_processed=n_rows,
        n_rows_deduped=n_deduped,
        cost_usd=cost,
    )


def _watch_run(
    source_table: str = "demo.docs",
    sink: str = "qdrant://localhost/col",
    since: int = 0,
    to: int = 5,
) -> WatchRun:
    return WatchRun(
        source_table=source_table,
        sink=sink,
        since_version=since,
        to_version=to,
        n_inserted=1,
    )


# ── construction ──────────────────────────────────────────────────────────────

def test_ledger_creates_db_file(tmp_path):
    db = tmp_path / "sub" / "ledger.db"
    Ledger(db_path=db).close()
    assert db.exists()


# ── write_run / recent_runs ───────────────────────────────────────────────────

def test_recent_runs_empty_initially(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    assert ledger.recent_runs(sink="qdrant://localhost/col") == []


def test_write_run_and_recent_runs(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run()
    ledger.write_run(run)

    rows = ledger.recent_runs(sink="qdrant://localhost/col")
    assert len(rows) == 1
    assert rows[0]["run_id"] == run.run_id
    assert rows[0]["model"] == "openai/text-embedding-3-small"
    assert rows[0]["n_rows"] == 5
    assert rows[0]["n_deduped"] == 0
    assert rows[0]["cost_usd"] == pytest.approx(0.0)


def test_recent_runs_respects_limit(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    for i in range(6):
        ledger.write_run(_run(n_rows=i))

    rows = ledger.recent_runs(sink="qdrant://localhost/col", limit=3)
    assert len(rows) == 3


def test_recent_runs_is_sink_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_run(_run(sink="qdrant://localhost/col_a"))
    ledger.write_run(_run(sink="qdrant://localhost/col_b"))

    assert len(ledger.recent_runs(sink="qdrant://localhost/col_a")) == 1
    assert len(ledger.recent_runs(sink="qdrant://localhost/col_b")) == 1
    assert len(ledger.recent_runs(sink="qdrant://localhost/col_c")) == 0


def test_recent_runs_ordered_most_recent_first(tmp_path):
    import time

    ledger = Ledger(db_path=tmp_path / "l.db")
    first = _run(n_rows=1)
    ledger.write_run(first)
    time.sleep(0.01)
    second = _run(n_rows=2)
    ledger.write_run(second)

    rows = ledger.recent_runs(sink="qdrant://localhost/col")
    assert rows[0]["run_id"] == second.run_id
    assert rows[1]["run_id"] == first.run_id


# ── hash_exists ───────────────────────────────────────────────────────────────

def test_hash_exists_false_before_provenance(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    assert not ledger.hash_exists("abc123", "openai/text-embedding-3-small", "qdrant://localhost/col")


def test_hash_exists_true_after_provenance(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run()
    ledger.write_run(run)
    embedding_id = str(uuid.uuid4())
    ledger.write_provenance(
        embedding_id=embedding_id,
        source_pk=embedding_id,
        source_hash="deadbeef",
        run_id=run.run_id,
        created_at="2025-01-01T00:00:00+00:00",
    )

    assert ledger.hash_exists("deadbeef", run.model, run.sink)


def test_hash_exists_model_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run(model="openai/text-embedding-3-small")
    ledger.write_run(run)
    ledger.write_provenance(
        embedding_id=str(uuid.uuid4()),
        source_pk="pk",
        source_hash="cafebabe",
        run_id=run.run_id,
        created_at="2025-01-01T00:00:00+00:00",
    )

    assert ledger.hash_exists("cafebabe", "openai/text-embedding-3-small", run.sink)
    assert not ledger.hash_exists("cafebabe", "openai/text-embedding-ada-002", run.sink)


def test_hash_exists_sink_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run(sink="qdrant://localhost/col_a")
    ledger.write_run(run)
    ledger.write_provenance(
        embedding_id=str(uuid.uuid4()),
        source_pk="pk",
        source_hash="feedface",
        run_id=run.run_id,
        created_at="2025-01-01T00:00:00+00:00",
    )

    assert ledger.hash_exists("feedface", run.model, "qdrant://localhost/col_a")
    assert not ledger.hash_exists("feedface", run.model, "qdrant://localhost/col_b")


def test_write_provenance_is_idempotent(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run()
    ledger.write_run(run)
    kwargs = dict(
        embedding_id=str(uuid.uuid4()),
        source_pk="pk",
        source_hash="aabb",
        run_id=run.run_id,
        created_at="2025-01-01T00:00:00+00:00",
    )
    ledger.write_provenance(**kwargs)
    ledger.write_provenance(**kwargs)  # second call must not raise


# ── cost_by_model ─────────────────────────────────────────────────────────────

def test_cost_by_model_empty(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    assert ledger.cost_by_model() == []


def test_cost_by_model_aggregates(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_run(_run(model="openai/text-embedding-3-small", cost=0.001))
    ledger.write_run(_run(model="openai/text-embedding-3-small", cost=0.002))
    ledger.write_run(_run(model="openai/text-embedding-ada-002", cost=0.005))

    by_model = {r["model"]: r["cost_usd"] for r in ledger.cost_by_model()}
    assert by_model["openai/text-embedding-3-small"] == pytest.approx(0.003)
    assert by_model["openai/text-embedding-ada-002"] == pytest.approx(0.005)


# ── provenance ────────────────────────────────────────────────────────────────

def test_provenance_returns_none_for_unknown(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    assert ledger.provenance("does-not-exist") is None


def test_provenance_returns_full_record(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    run = _run(cost=0.042)
    ledger.write_run(run)
    eid = str(uuid.uuid4())
    ledger.write_provenance(
        embedding_id=eid,
        source_pk=eid,
        source_hash="0xdeadbeef",
        run_id=run.run_id,
        created_at="2025-06-01T12:00:00+00:00",
    )

    rec = ledger.provenance(eid)
    assert rec is not None
    assert rec["embedding_id"] == eid
    assert rec["source_hash"] == "0xdeadbeef"
    assert rec["model"] == run.model
    assert rec["sink"] == run.sink
    assert rec["cost_usd"] == pytest.approx(0.042)
    assert rec["run_timestamp"]


# ── write_watch_run / last_watch_version ─────────────────────────────────────

def test_last_watch_version_none_for_first_run(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col") is None


def test_write_watch_run_persists_to_version(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_watch_run(_watch_run(to=7))

    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col") == 7


def test_last_watch_version_returns_most_recent(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_watch_run(_watch_run(to=3))
    ledger.write_watch_run(_watch_run(to=9))
    ledger.write_watch_run(_watch_run(to=6))

    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col") == 6


def test_last_watch_version_is_source_table_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_watch_run(_watch_run(source_table="schema.table_a", to=2))
    ledger.write_watch_run(_watch_run(source_table="schema.table_b", to=8))

    assert ledger.last_watch_version("schema.table_a", "qdrant://localhost/col") == 2
    assert ledger.last_watch_version("schema.table_b", "qdrant://localhost/col") == 8


def test_last_watch_version_is_sink_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "l.db")
    ledger.write_watch_run(_watch_run(sink="qdrant://localhost/col_x", to=4))
    ledger.write_watch_run(_watch_run(sink="qdrant://localhost/col_y", to=11))

    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col_x") == 4
    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col_y") == 11


# ── close / reopen ────────────────────────────────────────────────────────────

def test_close_and_reopen_preserves_data(tmp_path):
    db = tmp_path / "l.db"
    run = _run()

    ledger = Ledger(db_path=db)
    ledger.write_run(run)
    ledger.close()

    reopened = Ledger(db_path=db)
    rows = reopened.recent_runs(sink="qdrant://localhost/col")
    assert len(rows) == 1
    assert rows[0]["run_id"] == run.run_id
    reopened.close()
