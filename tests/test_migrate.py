"""Tests for Subsystem 3: migrate() — dual-write and drift-adapter strategies."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drift.ledger import Ledger
from drift.migrate import MigrateRun, migrate

# ── helpers ───────────────────────────────────────────────────────────────────

FAKE_TEXTS = [
    "Login issue after password reset",
    "Invoice shows wrong billing address",
    "Feature request: dark mode for dashboard",
]

SINK_OLD = "qdrant://localhost:6333/my_docs"
SINK_V2  = "qdrant://localhost:6333/my_docs_v2"


def _mock_spark_for_migrate(texts: list[str]):
    """
    Returns a mock SparkSession whose createDataFrame() produces a DataFrame
    that embed() can toPandas() into the expected shape.
    """
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_df.select.return_value.toPandas.return_value = pd.DataFrame(
        {"_migrate_text": texts}
    )
    mock_spark.createDataFrame.return_value = mock_df
    return mock_spark


# ── MigrateRun dataclass ──────────────────────────────────────────────────────

def test_migrate_run_defaults():
    run = MigrateRun(from_model="openai/ada-002", to_model="openai/3-small")
    assert run.run_id
    assert "T" in run.timestamp
    assert run.strategy == "dual-write"
    assert run.n_source == 0
    assert run.n_migrated == 0
    assert run.sink == ""
    assert run.sink_v2 == ""


# ── strategy validation ───────────────────────────────────────────────────────

def test_migrate_rejects_unknown_strategy(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    with pytest.raises(ValueError, match="Unknown strategy"):
        migrate("openai/ada-002", "openai/3-small", SINK_OLD,
                strategy="reindex-everything", ledger=ledger)


def test_migrate_rejects_shadow_eval_strategy(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    with pytest.raises(NotImplementedError, match="shadow-eval"):
        migrate("openai/ada-002", "openai/3-small", SINK_OLD,
                strategy="shadow-eval", ledger=ledger)


def test_migrate_rejects_non_qdrant_sink(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    with pytest.raises(NotImplementedError, match="pgvector"):
        migrate("openai/ada-002", "openai/3-small",
                "pg://localhost/mydb?table=embeddings",
                strategy="dual-write", ledger=ledger)


# ── dual-write happy path ─────────────────────────────────────────────────────

def test_migrate_dual_write_shadow_mode(tmp_path, monkeypatch):
    """Full happy path: scroll 3 texts → re-embed into _v2 collection."""
    pytest.importorskip("pyspark")
    ledger = Ledger(db_path=tmp_path / "test.db")

    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS)

    mock_spark = _mock_spark_for_migrate(FAKE_TEXTS)
    monkeypatch.setattr(
        "pyspark.sql.SparkSession.getActiveSession", lambda: mock_spark
    )

    with patch("drift.embed._upsert_qdrant"):
        run = migrate(
            from_model="openai/text-embedding-ada-002",
            to_model="openai/text-embedding-3-small",
            sink=SINK_OLD,
            strategy="dual-write",
            shadow_mode=True,
            ledger=ledger,
        )

    assert run.sink == SINK_OLD
    assert run.sink_v2 == SINK_V2
    assert run.n_source == 3
    assert run.n_migrated == 3
    assert run.from_model == "openai/text-embedding-ada-002"
    assert run.to_model == "openai/text-embedding-3-small"
    assert run.strategy == "dual-write"
    assert run.duration_s >= 0.0


def test_migrate_sink_v2_uri_derivation(tmp_path, monkeypatch):
    """sink_v2 appends _v2 to the collection name, preserving host/port."""
    pytest.importorskip("pyspark")
    ledger = Ledger(db_path=tmp_path / "test.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS)
    mock_spark = _mock_spark_for_migrate(FAKE_TEXTS)
    monkeypatch.setattr("pyspark.sql.SparkSession.getActiveSession", lambda: mock_spark)

    with patch("drift.embed._upsert_qdrant"):
        run = migrate(
            from_model="openai/ada-002",
            to_model="openai/3-small",
            sink="qdrant://prod-cluster:6333/support_docs",
            shadow_mode=True,
            ledger=ledger,
        )

    assert run.sink_v2 == "qdrant://prod-cluster:6333/support_docs_v2"


def test_migrate_empty_collection(tmp_path, monkeypatch):
    """Empty old collection → n_source=0, n_migrated=0, no crash, no Spark needed."""
    ledger = Ledger(db_path=tmp_path / "test.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: [])

    run = migrate(
        from_model="openai/ada-002",
        to_model="openai/3-small",
        sink=SINK_OLD,
        shadow_mode=True,
        ledger=ledger,
    )

    assert run.n_source == 0
    assert run.n_migrated == 0
    assert run.sink_v2 == SINK_V2
    # Spark must NOT have been called (no texts to embed)


# ── CLI smoke tests ───────────────────────────────────────────────────────────

def test_cli_migrate_stub_unsupported_strategy():
    """CLI exits with code 1 for unrecognised strategy."""
    from typer.testing import CliRunner

    from drift.cli import app

    runner = CliRunner()
    result = runner.invoke(app, [
        "migrate",
        "--from", "openai/ada-002",
        "--to", "openai/3-small",
        "--sink", SINK_OLD,
        "--strategy", "some-future-strategy",
    ])
    assert result.exit_code == 1


def test_cli_migrate_shadow_mode_runs(tmp_path, monkeypatch):
    """CLI migrate with shadow_mode completes and prints next steps."""
    pytest.importorskip("pyspark")
    from typer.testing import CliRunner

    from drift.cli import app

    ledger = Ledger(db_path=tmp_path / "cli.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS)
    mock_spark = _mock_spark_for_migrate(FAKE_TEXTS)
    monkeypatch.setattr("pyspark.sql.SparkSession.getActiveSession", lambda: mock_spark)

    runner = CliRunner()
    with patch("drift.embed._upsert_qdrant"), patch("drift.ledger.Ledger", return_value=ledger):
        result = runner.invoke(app, [
            "migrate",
            "--from", "openai/text-embedding-ada-002",
            "--to", "openai/text-embedding-3-small",
            "--sink", SINK_OLD,
            "--shadow-mode",
        ])

    assert result.exit_code == 0
    assert "3/3" in result.output          # n_migrated/n_source
    assert "my_docs_v2" in result.output   # new collection name
    assert "Next steps" in result.output   # instructional output
    assert "Catch-up" in result.output     # catch-up watch instruction


# ── drift-adapter strategy ────────────────────────────────────────────────────

# 60 fake texts — enough for 90/10 split (54 train, 6 val) and k=5 ARR eval
FAKE_TEXTS_LARGE = [
    f"Support ticket {i}: topic {i % 5}" for i in range(60)
]


def test_migrate_drift_adapter_shadow_mode(tmp_path, monkeypatch):
    """
    drift-adapter happy path: sample → fit → measure_arr → save adapter.

    shadow_mode: old_vecs == new_vecs → R = I → ARR = 1.0.
    Verifies MigrateRun fields, .npy file exists, no new Qdrant collection created.
    """
    ledger = Ledger(db_path=tmp_path / "test.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS_LARGE)
    monkeypatch.chdir(tmp_path)

    run = migrate(
        from_model="openai/text-embedding-ada-002",
        to_model="openai/text-embedding-3-small",
        sink=SINK_OLD,
        strategy="drift-adapter",
        shadow_mode=True,
        ledger=ledger,
    )

    assert run.strategy == "drift-adapter"
    assert run.arr == 1.0                          # shadow: old == new → perfect ARR
    assert run.adapter_path.endswith(".npy")
    assert (tmp_path / run.adapter_path).exists()  # file was written
    assert run.n_source == 60
    assert run.sink_v2 == ""                       # no new collection for drift-adapter
    assert run.n_migrated == 0                     # old index untouched
    assert run.duration_s >= 0.0


def test_migrate_drift_adapter_quality_gate_propagates(tmp_path, monkeypatch):
    """
    AdapterQualityError from measure_arr() propagates out of migrate() unmodified.
    Simulates the GloVe → MPNet failure case (71.5% ARR) via monkeypatch.
    """
    from drift.shadow_eval import AdapterQualityError

    ledger = Ledger(db_path=tmp_path / "test.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS_LARGE)
    monkeypatch.chdir(tmp_path)

    def _bad_arr(*args, **kwargs):
        raise AdapterQualityError(arr=0.71, threshold=0.97)

    monkeypatch.setattr("drift.shadow_eval.measure_arr", _bad_arr)

    with pytest.raises(AdapterQualityError) as exc_info:
        migrate(
            from_model="openai/ada-002",
            to_model="openai/3-small",
            sink=SINK_OLD,
            strategy="drift-adapter",
            shadow_mode=True,
            ledger=ledger,
        )

    assert exc_info.value.arr == 0.71
    assert exc_info.value.threshold == 0.97


def test_cli_migrate_drift_adapter_shadow_mode(tmp_path, monkeypatch):
    """CLI: drift migrate --strategy drift-adapter prints ARR, adapter path, query instructions."""
    from typer.testing import CliRunner

    from drift.cli import app

    ledger = Ledger(db_path=tmp_path / "cli-adapter.db")
    monkeypatch.setattr("drift.migrate._scroll_qdrant_texts", lambda *a, **kw: FAKE_TEXTS_LARGE)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    with patch("drift.ledger.Ledger", return_value=ledger):
        result = runner.invoke(app, [
            "migrate",
            "--from", "openai/text-embedding-ada-002",
            "--to", "openai/text-embedding-3-small",
            "--sink", SINK_OLD,
            "--strategy", "drift-adapter",
            "--shadow-mode",
        ])

    assert result.exit_code == 0, result.output
    assert "ARR" in result.output
    assert ".npy" in result.output
    assert "adapter.predict" in result.output   # shows how to use it at query time
    assert "Catch-up" in result.output          # catch-up watch instruction present
