"""Tests for Subsystem 1: embed() — shadow_mode, dedup, provenance, sinks."""

import math
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from drift.embed import EmbedRun, _mock_embedding, _text_hash, embed
from drift.ledger import Ledger


# ── helpers ──────────────────────────────────────────────────────────────────

def _spark_mock(texts: list[str], col: str = "body") -> MagicMock:
    """Fake PySpark DataFrame whose toPandas() returns a DataFrame of texts."""
    mock = MagicMock()
    mock.select.return_value.toPandas.return_value = pd.DataFrame({col: texts})
    return mock


# ── EmbedRun dataclass ───────────────────────────────────────────────────────

def test_embed_run_defaults():
    run = EmbedRun(model="openai/text-embedding-3-small", sink="qdrant://localhost/test")
    assert run.run_id
    assert "T" in run.timestamp
    assert run.n_rows_processed == 0
    assert run.cost_usd == 0.0


# ── input validation ─────────────────────────────────────────────────────────

def test_embed_rejects_bad_model_format():
    with pytest.raises(ValueError, match="provider/name"):
        embed(df=None, text_col="body",
              model="text-embedding-3-small",   # missing provider prefix
              sink="qdrant://localhost/test")


def test_embed_rejects_no_df_no_table():
    with pytest.raises(ValueError, match="source_table"):
        embed(df=None, text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/test")


def test_embed_rejects_unsupported_sink(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    df = _spark_mock(["hello"])
    with pytest.raises(ValueError, match="scheme"):
        embed(df=df, text_col="body",
              model="openai/text-embedding-3-small",
              sink="s3://bucket/prefix",
              shadow_mode=True, ledger=ledger)


# ── shadow_mode end-to-end ───────────────────────────────────────────────────

def test_embed_shadow_returns_embed_run(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    df = _spark_mock(["hello world", "foo bar"])

    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run = embed(df=df, text_col="body",
                    model="openai/text-embedding-3-small",
                    sink="qdrant://localhost:6333/col",
                    shadow_mode=True, ledger=ledger)

    assert isinstance(run, EmbedRun)
    assert run.n_rows_processed == 2
    assert run.n_rows_deduped == 0
    assert run.cost_usd == 0.0          # shadow is always free
    assert run.duration_s >= 0.0
    mock_upsert.assert_called_once()


def test_embed_shadow_dedup_on_second_run(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    texts = ["hello world", "foo bar"]

    with patch("drift.embed._upsert_qdrant"):
        embed(df=_spark_mock(texts), text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/col",
              shadow_mode=True, ledger=ledger)

    # Same texts, same model, same sink → 100% dedup
    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run2 = embed(df=_spark_mock(texts), text_col="body",
                     model="openai/text-embedding-3-small",
                     sink="qdrant://localhost/col",
                     shadow_mode=True, ledger=ledger)

    assert run2.n_rows_processed == 2
    assert run2.n_rows_deduped == 2
    mock_upsert.assert_not_called()     # nothing new to upsert


def test_embed_no_dedup_flag_re_embeds(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    texts = ["hello world"]

    with patch("drift.embed._upsert_qdrant"):
        embed(df=_spark_mock(texts), text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/col",
              shadow_mode=True, ledger=ledger)

    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run = embed(df=_spark_mock(texts), text_col="body",
                    model="openai/text-embedding-3-small",
                    sink="qdrant://localhost/col",
                    dedup=False,        # force re-embed even though hash exists
                    shadow_mode=True, ledger=ledger)

    assert run.n_rows_deduped == 0
    mock_upsert.assert_called_once()


def test_embed_different_sink_is_not_deduped(tmp_path):
    """Same text + same model, but different sink → not a cache hit."""
    ledger = Ledger(db_path=tmp_path / "test.db")
    texts = ["shared text"]

    with patch("drift.embed._upsert_qdrant"):
        embed(df=_spark_mock(texts), text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/col_a",
              shadow_mode=True, ledger=ledger)

    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run = embed(df=_spark_mock(texts), text_col="body",
                    model="openai/text-embedding-3-small",
                    sink="qdrant://localhost/col_b",   # different sink
                    shadow_mode=True, ledger=ledger)

    assert run.n_rows_deduped == 0      # cache is (model, sink)-scoped
    mock_upsert.assert_called_once()


def test_embed_empty_dataframe(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")

    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run = embed(df=_spark_mock([]), text_col="body",
                    model="openai/text-embedding-3-small",
                    sink="qdrant://localhost/col",
                    shadow_mode=True, ledger=ledger)

    assert run.n_rows_processed == 0
    mock_upsert.assert_not_called()


# ── provenance & ledger wiring ───────────────────────────────────────────────

def test_embed_writes_provenance(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    texts = ["unique provenance text"]

    with patch("drift.embed._upsert_qdrant"):
        embed(df=_spark_mock(texts), text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/col",
              shadow_mode=True, ledger=ledger)

    h = _text_hash(texts[0])
    assert ledger.hash_exists(h, "openai/text-embedding-3-small", "qdrant://localhost/col")
    costs = ledger.cost_by_model()
    assert len(costs) == 1
    assert costs[0]["model"] == "openai/text-embedding-3-small"
    assert costs[0]["cost_usd"] == 0.0  # shadow is free


def test_embed_writes_run_to_ledger(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")

    with patch("drift.embed._upsert_qdrant"):
        run = embed(df=_spark_mock(["a", "b", "c"]), text_col="body",
                    model="openai/text-embedding-3-small",
                    sink="qdrant://localhost/col",
                    shadow_mode=True, ledger=ledger)

    recent = ledger.recent_runs(sink="qdrant://localhost/col")
    assert len(recent) == 1
    assert recent[0]["run_id"] == run.run_id
    assert recent[0]["n_rows"] == 3


# ── upsert payload shape ─────────────────────────────────────────────────────

def test_embed_upsert_payload_shape(tmp_path):
    """_upsert_qdrant receives points with the expected keys and payload fields."""
    ledger = Ledger(db_path=tmp_path / "test.db")

    captured: list = []

    def capture(sink, points):
        captured.extend(points)

    with patch("drift.embed._upsert_qdrant", side_effect=capture):
        embed(df=_spark_mock(["check payload"]), text_col="body",
              model="openai/text-embedding-3-small",
              sink="qdrant://localhost/col",
              shadow_mode=True, ledger=ledger)

    assert len(captured) == 1
    p = captured[0]
    assert "id" in p
    assert "vector" in p
    assert len(p["vector"]) == 1536
    assert p["payload"]["source_text"] == "check payload"
    assert "source_hash" in p["payload"]
    assert "embed_run_id" in p["payload"]
    assert "timestamp" in p["payload"]


# ── mock embedding properties ────────────────────────────────────────────────

def test_mock_embedding_is_deterministic():
    assert _mock_embedding("hello") == _mock_embedding("hello")


def test_mock_embedding_differs_by_text():
    assert _mock_embedding("text A") != _mock_embedding("text B")


def test_mock_embedding_is_unit_vector():
    vec = _mock_embedding("any text")
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6


def test_mock_embedding_default_dim():
    assert len(_mock_embedding("x")) == 1536


# ── other module scaffold tests still pass ───────────────────────────────────

def test_watch_run_defaults():
    from drift.watch import WatchRun
    run = WatchRun(source_table="catalog.docs")
    assert run.run_id
    assert run.n_inserted == 0


def test_migrate_rejects_unknown_strategy():
    from drift.migrate import migrate
    with pytest.raises(ValueError, match="Unknown strategy"):
        migrate(from_model="openai/ada-002", to_model="openai/3-small",
                sink="qdrant://localhost/test", strategy="magic")


def test_ledger_empty_state(tmp_path):
    ledger = Ledger(db_path=tmp_path / "test.db")
    assert ledger.cost_by_model() == []
    assert not ledger.hash_exists("abc123", "openai/text-embedding-3-small", "qdrant://localhost/x")
    ledger.close()
