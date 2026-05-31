"""Tests for Subsystem 2: watch() — Delta CDF → Qdrant incremental refresh."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drift.ledger import Ledger
from drift.watch import WatchRun, _delete_from_sink, watch


# ── helpers ──────────────────────────────────────────────────────────────────

def _cdf_mock(inserts=None, updates=None, deletes=None, max_version=5):
    """
    Fake Delta CDF DataFrame. Builds a pandas DF with _change_type and body
    columns, wraps it in a chain of Spark mock calls that watch() makes.
    """
    rows = []
    for text in (inserts or []):
        rows.append({"body": text, "_change_type": "insert", "_commit_version": max_version})
    for text in (updates or []):
        rows.append({"body": text, "_change_type": "update_postimage", "_commit_version": max_version})
    for text in (deletes or []):
        rows.append({"body": text, "_change_type": "delete", "_commit_version": max_version})

    full_pdf = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["body", "_change_type", "_commit_version"]
    )

    def make_filter_mock(pdf):
        """Return a mock whose .count() and further .filter() work."""
        m = MagicMock()
        m.count.return_value = len(pdf)
        m.select.return_value.toPandas.return_value = pdf[["body"]] if len(pdf) else pd.DataFrame({"body": []})

        def filter_side(expr):
            if "insert" in expr and "update_postimage" in expr:
                sub = pdf[pdf["_change_type"].isin(["insert", "update_postimage"])]
            elif "delete" in expr:
                sub = pdf[pdf["_change_type"] == "delete"]
            elif "_change_type = 'insert'" in expr or "== 'insert'" in expr:
                sub = pdf[pdf["_change_type"] == "insert"]
            elif "update_postimage" in expr:
                sub = pdf[pdf["_change_type"] == "update_postimage"]
            else:
                sub = pdf
            return make_filter_mock(sub)

        m.filter.side_effect = filter_side

        # selectExpr("max(_commit_version)") — matches watch.py's to_version capture
        m.selectExpr.return_value.collect.return_value = [
            [max_version if len(pdf) else None]
        ]

        return m

    return make_filter_mock(full_pdf)


# ── WatchRun dataclass ───────────────────────────────────────────────────────

def test_watch_run_defaults():
    run = WatchRun(source_table="catalog.docs", sink="qdrant://localhost/col")
    assert run.run_id
    assert run.n_inserted == 0
    assert run.n_deleted == 0
    assert run.duration_s == 0.0


# ── watch() — inserts ────────────────────────────────────────────────────────

def test_watch_embeds_inserts(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    cdf = _cdf_mock(inserts=["new doc A", "new doc B"])

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch.embed") as mock_embed, \
         patch("drift.embed._upsert_qdrant"):

        spark = MagicMock()
        spark.read.format.return_value.option.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost:6333/col",
            shadow_mode=True,
            ledger=ledger,
        )

    assert run.n_inserted == 2
    assert run.n_updated == 0
    assert run.n_deleted == 0
    assert run.to_version == 5
    mock_embed.assert_called_once()


def test_watch_embeds_updates(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    cdf = _cdf_mock(updates=["updated doc"])

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch.embed") as mock_embed, \
         patch("drift.embed._upsert_qdrant"):

        spark = MagicMock()
        spark.read.format.return_value.option.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost:6333/col",
            shadow_mode=True,
            ledger=ledger,
        )

    assert run.n_updated == 1
    assert run.n_inserted == 0
    mock_embed.assert_called_once()


# ── watch() — deletes ────────────────────────────────────────────────────────

def test_watch_deletes_from_sink(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    cdf = _cdf_mock(deletes=["doc to delete"])

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch._delete_from_sink", return_value=1) as mock_delete, \
         patch("drift.watch.embed") as mock_embed:

        spark = MagicMock()
        spark.read.format.return_value.option.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost:6333/col",
            shadow_mode=True,
            ledger=ledger,
        )

    assert run.n_deleted == 1
    assert run.n_inserted == 0
    mock_delete.assert_called_once_with("qdrant://localhost:6333/col", ["doc to delete"])
    mock_embed.assert_not_called()   # no inserts or updates


# ── watch() — empty CDF ──────────────────────────────────────────────────────

def test_watch_empty_cdf_noop(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    cdf = _cdf_mock()   # no changes

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch.embed") as mock_embed, \
         patch("drift.watch._delete_from_sink") as mock_delete:

        spark = MagicMock()
        spark.read.format.return_value.option.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost:6333/col",
            ledger=ledger,
        )

    assert run.n_inserted == 0
    assert run.n_updated == 0
    assert run.n_deleted == 0
    mock_embed.assert_not_called()
    mock_delete.assert_not_called()


# ── checkpoint / since_version ───────────────────────────────────────────────

def test_watch_writes_checkpoint_to_ledger(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    cdf = _cdf_mock(inserts=["doc"], max_version=7)

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch.embed"):

        spark = MagicMock()
        spark.read.format.return_value.option.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost:6333/col",
            since_version=3,
            ledger=ledger,
        )

    assert run.since_version == 3
    assert run.to_version == 7
    # Checkpoint must be persisted so next call can auto-resolve since_version
    assert ledger.last_watch_version("demo.docs", "qdrant://localhost:6333/col") == 7


def test_watch_auto_resolves_since_version_from_ledger(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")

    # Simulate a prior run that checkpointed to_version=4
    prior = WatchRun(source_table="demo.docs", sink="qdrant://localhost/col",
                     since_version=0, to_version=4)
    ledger.write_watch_run(prior)

    cdf = _cdf_mock(inserts=["new doc"], max_version=9)

    with patch("drift.watch._get_spark") as mock_spark, \
         patch("drift.watch.embed"):

        spark = MagicMock()
        # Capture the startingVersion option that watch() passes
        option_calls = []
        def record_option(k, v):
            option_calls.append((k, v))
            return spark.read.format.return_value.option.return_value
        spark.read.format.return_value.option.return_value.option.side_effect = record_option
        # table() is called on the return value of record_option (= option.return_value), not one level deeper
        spark.read.format.return_value.option.return_value.table.return_value = cdf
        mock_spark.return_value = spark

        run = watch(
            source_table="demo.docs",
            text_col="body",
            sink="qdrant://localhost/col",
            since_version=None,   # should auto-resolve to 4
            ledger=ledger,
        )

    # watch() must have used since_version=4 from the ledger
    assert run.since_version == 4


# ── ledger watch API ─────────────────────────────────────────────────────────

def test_last_watch_version_returns_none_for_first_run(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col") is None


def test_last_watch_version_is_sink_scoped(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    r1 = WatchRun(source_table="demo.docs", sink="qdrant://localhost/col_a",
                  to_version=3)
    r2 = WatchRun(source_table="demo.docs", sink="qdrant://localhost/col_b",
                  to_version=9)
    ledger.write_watch_run(r1)
    ledger.write_watch_run(r2)

    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col_a") == 3
    assert ledger.last_watch_version("demo.docs", "qdrant://localhost/col_b") == 9


# ── pgvector CDC stub ────────────────────────────────────────────────────────

def test_delete_from_sink_raises_for_pgvector():
    with pytest.raises(NotImplementedError, match="v0.2"):
        _delete_from_sink("pg://localhost/mydb", ["some text"])


def test_delete_from_sink_raises_for_unknown_scheme():
    with pytest.raises(ValueError, match="scheme"):
        _delete_from_sink("s3://bucket/prefix", ["some text"])
