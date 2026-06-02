"""
DriftAdapter integration test — end-to-end pipeline verification.

Tests the full drift-adapter flow from paired embedding through ARR gate
to adapter serialization and query-time prediction.

Levels:
    1. Pure NumPy (no Qdrant, no Spark) — math pipeline correctness
    2. migrate(strategy="drift-adapter") with real Qdrant in shadow_mode
    3. ARR quality gate fires on a deliberately bad adapter
    4. CLI: drift migrate --strategy drift-adapter --shadow-mode end-to-end

Requirements:
    pip install 'drift-spark[spark,qdrant]'

    Levels 1 + 3: no external dependencies.
    Levels 2 + 4: docker run -p 6333:6333 qdrant/qdrant

Run:
    python integration-tests/it-adapter.py
    python integration-tests/it-adapter.py --skip-qdrant   # run only L1 + L3
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# ── imports ───────────────────────────────────────────────────────────────────

from drift.adapter import DriftAdapter
from drift.shadow_eval import AdapterQualityError, measure_arr

SKIP_QDRANT = "--skip-qdrant" in sys.argv

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_OLD = "openai/text-embedding-3-small"
MODEL_NEW = "openai/text-embedding-3-large"
SINK = f"qdrant://{QDRANT_HOST}:{QDRANT_PORT}/adapter_it_source"

PASS = "✓"
FAIL = "✗"
errors: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}  ← FAILED")
        errors.append(label)


def _unit_norm(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _rotation_matrix(d: int, theta_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    R = np.eye(d, dtype=np.float32)
    R[0, 0] = np.cos(theta)
    R[0, 1] = np.sin(theta)
    R[1, 0] = -np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R


# ── Level 1: pure NumPy pipeline ─────────────────────────────────────────────

print("\n=== Level 1: pure NumPy — fit → measure_arr → save → load → predict ===")

d, N = 32, 300
rng = np.random.default_rng(0)
R_true = _rotation_matrix(d, 30.0)

raw = rng.standard_normal((N, d)).astype(np.float32)
X_old = _unit_norm(raw)
X_new = X_old @ R_true  # perfectly rotated — simulates model upgrade

split = int(N * 0.9)
train_old, val_old = X_old[:split], X_old[split:]
train_new, val_new = X_new[:split], X_new[split:]

adapter = DriftAdapter().fit(train_old, train_new)
check(np.allclose(adapter.R @ adapter.R.T, np.eye(d), atol=1e-3), "R is orthogonal")

arr = measure_arr(adapter, val_old, val_new, k=5, threshold=None)
check(arr == 1.0, f"ARR == 1.0 for perfect rotation  (got {arr:.4f})")

with tempfile.TemporaryDirectory() as tmpdir:
    npy_path = os.path.join(tmpdir, "adapter.npy")
    adapter.save(npy_path)
    check(Path(npy_path).exists(), ".npy file written")
    check(Path(npy_path).stat().st_size >= d * d * 4, ".npy file size ≥ d²×4 bytes")

    loaded = DriftAdapter.load(npy_path)
    check(np.allclose(loaded.R, adapter.R), "loaded.R == original.R")

    q_new = X_new[0]
    adapted = loaded.predict(q_new)
    cos_sim = float(adapted @ X_old[0])
    check(cos_sim > 0.99, f"predict(q_new) ≈ q_old (cos_sim={cos_sim:.4f})")

    adapted_batch = loaded.predict(X_new[:10])
    check(adapted_batch.shape == (10, d), "batch predict shape (10, d)")


# ── Level 2: migrate() drift-adapter with real Qdrant ────────────────────────

if SKIP_QDRANT:
    print("\n=== Level 2: migrate() drift-adapter [SKIPPED — --skip-qdrant] ===")
else:
    print("\n=== Level 2: seed collection → migrate(drift-adapter) → verify adapter ===")

    from pyspark.sql import SparkSession
    from qdrant_client import QdrantClient

    from drift.embed import embed
    from drift.ledger import Ledger
    from drift.migrate import migrate

    spark = (
        SparkSession.builder.master("local[*]")
        .appName("drift-it-adapter")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def _cleanup(*collections: str) -> None:
        for coll in collections:
            try:
                client.delete_collection(coll)
            except Exception:
                pass

    def _seed(n: int, sink: str) -> None:
        rows = [{"id": str(i), "body": f"Support ticket {i}: {['login', 'billing', 'feature', 'outage', 'other'][i % 5]} issue"}
                for i in range(n)]
        df = spark.createDataFrame(rows)
        embed(df, text_col="body", model=MODEL_OLD, sink=sink,
              shadow_mode=True, dedup=False, ledger=Ledger())

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        _cleanup("adapter_it_source")
        _seed(100, SINK)   # 100 docs — enough for 90/10 split

        run = migrate(
            from_model=MODEL_OLD,
            to_model=MODEL_NEW,
            sink=SINK,
            strategy="drift-adapter",
            shadow_mode=True,
            ledger=Ledger(),
        )

        check(run.strategy == "drift-adapter", "run.strategy == 'drift-adapter'")
        check(run.arr == 1.0, f"ARR == 1.0 in shadow_mode  (got {run.arr:.4f})")
        check(run.adapter_path.endswith(".npy"), f"adapter_path ends with .npy  (got {run.adapter_path!r})")
        check(Path(run.adapter_path).exists(), "adapter .npy file exists on disk")
        check(run.n_source == 100, f"n_source == 100  (got {run.n_source})")
        check(run.sink_v2 == "", "sink_v2 is empty (old index untouched)")
        check(run.n_migrated == 0, "n_migrated == 0 (no new collection written)")
        check(run.duration_s >= 0.0, "duration_s is non-negative")

        # Verify the adapter actually works at query time
        loaded_adapter = DriftAdapter.load(run.adapter_path)
        from drift.embed import _mock_embedding
        q_new = np.array(_mock_embedding("test query"), dtype=np.float32)
        q_adapted = loaded_adapter.predict(q_new)
        check(q_adapted.shape == q_new.shape, "predict() output shape matches input")
        check(np.allclose(loaded_adapter.R @ loaded_adapter.R.T, np.eye(loaded_adapter.R.shape[0]), atol=1e-3),
              "loaded adapter R is orthogonal")

        _cleanup("adapter_it_source")


# ── Level 3: ARR quality gate fires on bad adapter ───────────────────────────

print("\n=== Level 3: AdapterQualityError raised when ARR < threshold ===")

d3, N3 = 16, 100
rng3 = np.random.default_rng(7)
raw3 = rng3.standard_normal((N3, d3)).astype(np.float32)
X3 = _unit_norm(raw3)

# Construct a deliberately bad adapter (random orthogonal R — scrambles neighbors)
Q, _ = np.linalg.qr(rng3.standard_normal((d3, d3)))
bad_adapter = DriftAdapter.__new__(DriftAdapter)
bad_adapter.R = Q.astype(np.float32)

caught = False
try:
    measure_arr(bad_adapter, X3, X3, k=5, threshold=0.97)
except AdapterQualityError as e:
    caught = True
    check(e.arr < 0.97, f"e.arr < 0.97  (got {e.arr:.4f})")
    check(e.threshold == 0.97, f"e.threshold == 0.97  (got {e.threshold})")
    check("dual-write" in str(e), "error message mentions dual-write fallback")

check(caught, "AdapterQualityError was raised for bad adapter")

# Verify threshold=None suppresses the raise
arr_unchecked = measure_arr(bad_adapter, X3, X3, k=5, threshold=None)
check(isinstance(arr_unchecked, float), "threshold=None returns float without raising")
check(arr_unchecked < 0.97, f"unchecked ARR is still bad  (got {arr_unchecked:.4f})")


# ── Level 4: CLI end-to-end ───────────────────────────────────────────────────

if SKIP_QDRANT:
    print("\n=== Level 4: CLI drift migrate --strategy drift-adapter [SKIPPED] ===")
else:
    print("\n=== Level 4: CLI end-to-end — drift migrate --strategy drift-adapter ===")
    import subprocess

    with tempfile.TemporaryDirectory() as tmpdir:
        _seed(100, SINK)

        result = subprocess.run(
            [
                "drift", "migrate",
                "--from", MODEL_OLD,
                "--to", MODEL_NEW,
                "--sink", SINK,
                "--strategy", "drift-adapter",
                "--shadow-mode",
            ],
            capture_output=True, text=True, cwd=tmpdir,
        )

        check(result.returncode == 0, f"CLI exit code == 0  (got {result.returncode})")
        check("ARR" in result.stdout, "CLI output contains ARR score")
        check(".npy" in result.stdout, "CLI output contains adapter path")
        check("adapter.predict" in result.stdout, "CLI output shows how to use the adapter")
        check("Catch-up" in result.stdout, "CLI output includes catch-up watch instruction")

        _cleanup("adapter_it_source")


# ── Results ───────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
if errors:
    print(f"FAILED — {len(errors)} assertion(s):")
    for e in errors:
        print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    print("ALL LEVELS PASSED")
    print(f"  Level 1: pure NumPy pipeline              ✓")
    qdrant_note = "[skipped]" if SKIP_QDRANT else "✓"
    print(f"  Level 2: migrate() drift-adapter          {qdrant_note}")
    print(f"  Level 3: ARR quality gate                 ✓")
    print(f"  Level 4: CLI end-to-end                   {qdrant_note}")
    sys.exit(0)
