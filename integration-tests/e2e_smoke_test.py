#!/usr/bin/env python3
"""
e2e_smoke_test.py — Drift end-to-end smoke test.

7 levels of coverage, no Qdrant, no OpenAI key required.
Requires: pip install 'drift-spark[spark,qdrant]' + Java 17.
Runtime: ~30s (Spark JVM startup dominates L3+).

Usage:
    python integration-tests/e2e_smoke_test.py             # all levels
    python integration-tests/e2e_smoke_test.py --level 2   # L0–L2 only (no Spark needed)
    python integration-tests/e2e_smoke_test.py --no-spark  # L0–L2 only, skip Spark levels
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ── constants ────────────────────────────────────────────────────────────────

SINK = "qdrant://localhost:6333/smoke_docs"
MODEL_OLD = "openai/text-embedding-ada-002"
MODEL_NEW = "openai/text-embedding-3-small"
TEXT_COL = "body"
TABLE = "smoke_docs"

FAKE_TEXTS = [f"Support ticket {i}: topic {i % 5}" for i in range(30)]
FAKE_TEXTS_LARGE = FAKE_TEXTS * 3  # 90 docs — enough for the 90/10 adapter train/val split

SPARK_LEVELS = {3, 4, 5, 6}

_results: list[tuple[int, str, float, str]] = []


# ── helpers ──────────────────────────────────────────────────────────────────

def _section(level: int, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  L{level}  {title}")
    print(f"{'─' * 60}")


def _ok(level: int, name: str, elapsed: float) -> None:
    _results.append((level, name, elapsed, "PASS"))
    print(f"  ✓ PASS  ({elapsed:.2f}s)")


def _fail(level: int, name: str, elapsed: float, exc: Exception) -> None:
    _results.append((level, name, elapsed, f"FAIL: {exc}"))
    print(f"  ✗ FAIL  ({elapsed:.2f}s): {exc}")


_spark = None  # module-level singleton so JVM starts only once


def _spark_session():
    global _spark
    if _spark is None:
        from pyspark.sql import SparkSession
        _spark = (
            SparkSession.builder
            .master("local[2]")
            .appName("drift-smoke")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        _spark.sparkContext.setLogLevel("ERROR")
    df = _spark.createDataFrame([(t,) for t in FAKE_TEXTS], [TEXT_COL])
    df.createOrReplaceTempView(TABLE)
    return _spark


# ── L0: imports + version ────────────────────────────────────────────────────

def level_0() -> None:
    _section(0, "Public API — imports + version string")
    t = time.time()
    try:
        import drift
        from drift import DriftAdapter, measure_arr, AdapterQualityError, Ledger, MigrateRun  # noqa: F401

        assert hasattr(drift, "__version__"), "drift.__version__ missing"
        assert drift.__version__ == "0.5.0", f"expected 0.5.0, got {drift.__version__!r}"
        _ok(0, "imports + version", time.time() - t)
    except Exception as e:
        _fail(0, "imports + version", time.time() - t, e)


# ── L1: DriftAdapter pure math ───────────────────────────────────────────────

def level_1() -> None:
    _section(1, "DriftAdapter — pure NumPy math (fit / predict / save / load)")
    t = time.time()
    try:
        from drift import DriftAdapter

        rng = np.random.default_rng(42)
        d, N = 64, 200

        # QR-orthonormal X_old so M = X_old.T @ X_new = R_true exactly (no approximation noise)
        Q_rot, _ = np.linalg.qr(rng.standard_normal((d, d)))
        R_true = Q_rot.astype(np.float32)
        X_old, _ = np.linalg.qr(rng.standard_normal((N, d)).astype(np.float32))
        X_new = X_old @ R_true

        adapter = DriftAdapter().fit(X_old, X_new)

        assert np.allclose(adapter.R, R_true, atol=1e-3), "R does not recover R_true"
        assert np.allclose(adapter.predict(X_new), X_old, atol=1e-3), "predict roundtrip failed"
        assert np.allclose(adapter.R @ adapter.R.T, np.eye(d), atol=1e-5), "R not orthogonal"

        # 1D input stays 1D; 2D stays 2D
        q = rng.standard_normal(d).astype(np.float32)
        assert adapter.predict(q).shape == (d,), "1D predict shape wrong"
        assert adapter.predict(X_new[:5]).shape == (5, d), "2D predict shape wrong"

        # identity case: X_old == X_new → R ≈ I
        id_adapter = DriftAdapter().fit(X_old, X_old)
        assert np.allclose(id_adapter.R, np.eye(d), atol=1e-5), "identity case failed"

        # save / load roundtrip
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            npy_path = f.name
        try:
            adapter.save(npy_path)
            loaded = DriftAdapter.load(npy_path)
            assert np.allclose(loaded.R, adapter.R), "save/load not idempotent"
        finally:
            os.unlink(npy_path)

        _ok(1, "DriftAdapter math", time.time() - t)
    except Exception as e:
        _fail(1, "DriftAdapter math", time.time() - t, e)


# ── L2: measure_arr() + AdapterQualityError ──────────────────────────────────

def level_2() -> None:
    _section(2, "measure_arr() — perfect adapter ARR=1.0; bad adapter raises")
    t = time.time()
    try:
        from drift import DriftAdapter, measure_arr, AdapterQualityError

        rng = np.random.default_rng(7)
        d, N = 32, 100

        X_old = rng.standard_normal((N, d)).astype(np.float32)
        X_old /= np.linalg.norm(X_old, axis=1, keepdims=True)

        # perfect adapter (identity R) → ARR == 1.0
        perfect = DriftAdapter()
        perfect.R = np.eye(d, dtype=np.float32)
        arr = measure_arr(perfect, X_old, X_old, k=5, threshold=None)
        assert arr == 1.0, f"perfect ARR should be 1.0, got {arr}"

        # bad adapter (random orthogonal R on unrelated vecs) → ARR << 0.97, raises
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        bad = DriftAdapter()
        bad.R = Q.astype(np.float32)
        X_unrelated = rng.standard_normal((N, d)).astype(np.float32)
        X_unrelated /= np.linalg.norm(X_unrelated, axis=1, keepdims=True)

        raised = False
        try:
            measure_arr(bad, X_old, X_unrelated, k=5, threshold=0.97)
        except AdapterQualityError as ex:
            raised = True
            assert ex.arr < 0.97, f"ARR should be <0.97, got {ex.arr:.3f}"
            assert ex.threshold == 0.97, f"threshold attr wrong: {ex.threshold}"
        assert raised, "AdapterQualityError not raised for bad adapter"

        # threshold=None returns float, never raises
        arr_float = measure_arr(bad, X_old, X_unrelated, k=5, threshold=None)
        assert isinstance(arr_float, float), "threshold=None should return float"

        _ok(2, "measure_arr + AdapterQualityError", time.time() - t)
    except Exception as e:
        _fail(2, "measure_arr + AdapterQualityError", time.time() - t, e)


# ── L3: embed() shadow_mode ──────────────────────────────────────────────────

def level_3() -> None:
    _section(3, "embed() shadow_mode — run1 embeds 30; run2 dedupes 30")
    t = time.time()
    try:
        from drift import Ledger
        from drift.embed import embed

        print("   (starting local Spark — ~10s on first call)")
        _spark_session()

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(db_path=f"{tmp}/ledger.db")

            with patch("drift.embed._upsert_qdrant"):
                run1 = embed(
                    df=None, source_table=TABLE, text_col=TEXT_COL,
                    model=MODEL_OLD, sink=SINK, dedup=True,
                    shadow_mode=True, ledger=ledger,
                )
                assert run1.n_rows_processed == 30, (
                    f"run1: expected 30 rows, got {run1.n_rows_processed}"
                )
                assert run1.n_rows_deduped == 0, (
                    f"run1: expected 0 deduped, got {run1.n_rows_deduped}"
                )
                assert run1.cost_usd == 0.0, "shadow_mode cost should be 0.0"

                # identical data → 100% dedup, zero writes
                run2 = embed(
                    df=None, source_table=TABLE, text_col=TEXT_COL,
                    model=MODEL_OLD, sink=SINK, dedup=True,
                    shadow_mode=True, ledger=ledger,
                )
                assert run2.n_rows_deduped == 30, (
                    f"run2: expected 30 deduped, got {run2.n_rows_deduped}"
                )

        _ok(3, "embed shadow_mode + dedup", time.time() - t)
    except Exception as e:
        _fail(3, "embed shadow_mode + dedup", time.time() - t, e)


# ── L4: Ledger lineage ───────────────────────────────────────────────────────

def level_4() -> None:
    _section(4, "Ledger — provenance() + cost_by_model() + recent_runs()")
    t = time.time()
    try:
        from drift import Ledger
        from drift.embed import embed

        _spark_session()

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(db_path=f"{tmp}/ledger.db")

            with patch("drift.embed._upsert_qdrant"):
                run = embed(
                    df=None, source_table=TABLE, text_col=TEXT_COL,
                    model=MODEL_OLD, sink=SINK, dedup=True,
                    shadow_mode=True, ledger=ledger,
                )

            # recent_runs
            runs = ledger.recent_runs(sink=SINK, limit=5)
            assert len(runs) >= 1, "no runs recorded in ledger after embed"
            assert runs[0]["model"] == MODEL_OLD, "ledger model field wrong"

            # provenance — fetch a real embedding_id from the ledger
            cur = ledger._conn.execute(
                "SELECT embedding_id FROM embedding_provenance LIMIT 1"
            )
            row = cur.fetchone()
            prov = ledger.provenance(row[0]) if row else None
            assert prov is not None, "provenance() returned None after embed"

            # cost_by_model
            costs = ledger.cost_by_model()
            assert isinstance(costs, list) and len(costs) >= 1, (
                "cost_by_model() should return a non-empty list"
            )

        _ok(4, "Ledger lineage", time.time() - t)
    except Exception as e:
        _fail(4, "Ledger lineage", time.time() - t, e)


# ── L5: migrate(dual-write) ──────────────────────────────────────────────────

def level_5() -> None:
    _section(5, "migrate(dual-write, shadow_mode) — n_migrated == n_source")
    t = time.time()
    try:
        from drift import Ledger
        from drift.migrate import migrate

        _spark_session()

        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(db_path=f"{tmp}/ledger.db")

            with patch("drift.migrate._scroll_qdrant_texts", return_value=FAKE_TEXTS), \
                 patch("drift.embed._upsert_qdrant"):
                run = migrate(
                    from_model=MODEL_OLD, to_model=MODEL_NEW,
                    sink=SINK, strategy="dual-write",
                    shadow_mode=True, ledger=ledger,
                )

            assert run.n_migrated == run.n_source, (
                f"n_migrated {run.n_migrated} != n_source {run.n_source}"
            )
            assert run.n_source == len(FAKE_TEXTS), (
                f"n_source {run.n_source} != {len(FAKE_TEXTS)}"
            )
            assert run.sink_v2.endswith("_v2"), f"sink_v2 should end with _v2, got {run.sink_v2!r}"
            assert run.adapter_path == "", "dual-write should have empty adapter_path"

        _ok(5, "migrate dual-write", time.time() - t)
    except Exception as e:
        _fail(5, "migrate dual-write", time.time() - t, e)


# ── L6: migrate(drift-adapter) ───────────────────────────────────────────────

def level_6() -> None:
    _section(6, "migrate(drift-adapter, shadow_mode) — ARR≥0.97 + .npy on disk")
    t = time.time()
    try:
        from drift import DriftAdapter, Ledger
        from drift.migrate import migrate

        _spark_session()

        with tempfile.TemporaryDirectory() as tmp:
            prev_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                ledger = Ledger(db_path=f"{tmp}/ledger.db")

                with patch("drift.migrate._scroll_qdrant_texts", return_value=FAKE_TEXTS_LARGE):
                    run = migrate(
                        from_model=MODEL_OLD, to_model=MODEL_NEW,
                        sink=SINK, strategy="drift-adapter",
                        shadow_mode=True, ledger=ledger,
                    )

                assert run.adapter_path.endswith(".npy"), (
                    f"expected .npy path, got {run.adapter_path!r}"
                )
                assert Path(run.adapter_path).exists(), (
                    f"adapter file not written to disk: {run.adapter_path}"
                )
                # shadow_mode: old_vecs == new_vecs → R = I → ARR = 1.0
                assert run.arr >= 0.97, f"ARR {run.arr:.3f} below 0.97 threshold"
                assert run.n_source == len(FAKE_TEXTS_LARGE), (
                    f"n_source {run.n_source} != {len(FAKE_TEXTS_LARGE)}"
                )
                assert run.sink_v2 == "", "drift-adapter: old index untouched, no sink_v2"

                # round-trip: load the saved adapter and apply to a synthetic query
                saved = DriftAdapter.load(run.adapter_path)
                q = np.random.default_rng(0).standard_normal(saved.R.shape[0]).astype(np.float32)
                assert saved.predict(q).shape == q.shape, "loaded adapter predict() shape wrong"
            finally:
                os.chdir(prev_cwd)

        _ok(6, "migrate drift-adapter", time.time() - t)
    except Exception as e:
        _fail(6, "migrate drift-adapter", time.time() - t, e)


# ── main ─────────────────────────────────────────────────────────────────────

LEVELS = [level_0, level_1, level_2, level_3, level_4, level_5, level_6]


def main() -> int:
    parser = argparse.ArgumentParser(description="Drift e2e smoke test — 7 levels")
    parser.add_argument(
        "--level", type=int, default=6,
        help="Run levels 0..N (default: 6 = all)",
    )
    parser.add_argument(
        "--no-spark", action="store_true",
        help="Skip Spark levels (L3–L6); useful in CI without Java",
    )
    args = parser.parse_args()

    max_level = min(args.level, len(LEVELS) - 1)

    print(f"\n{'═' * 60}")
    print("  Drift e2e smoke test")
    print(f"  Running L0 – L{max_level}"
          + (" (Spark levels skipped)" if args.no_spark else ""))
    print(f"{'═' * 60}")

    t_total = time.time()
    for i, fn in enumerate(LEVELS[: max_level + 1]):
        if args.no_spark and i in SPARK_LEVELS:
            print(f"\n  L{i}  [skipped — --no-spark]")
            continue
        fn()

    elapsed = time.time() - t_total
    print(f"\n{'═' * 60}")
    print(f"  Summary  ({elapsed:.1f}s total)")
    print(f"{'─' * 60}")
    failed = 0
    for level, name, dur, status in _results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  L{level}  {name:<42} {dur:>5.2f}s  {status}")
        if status != "PASS":
            failed += 1
    print(f"{'═' * 60}")
    if failed:
        print(f"  {failed} level(s) FAILED\n")
        return 1
    print(f"  All levels passed.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
