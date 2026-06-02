"""
Drift end-to-end smoke test — verifies every subsystem after pip install.

No Qdrant, no OpenAI API key, no Delta cluster required.
Qdrant calls are patched; embeddings use shadow_mode deterministic mocks.
Spark runs in local[*] mode — requires Java 17+ on PATH.

Setup:
    pip install 'drift-spark[spark,qdrant]'
    python examples/e2e_smoke_test.py

Expected runtime: ~30 seconds (Spark JVM startup dominates).

Levels:
    0. Imports + version
    1. DriftAdapter — pure NumPy (fit / predict / save / load)
    2. AdapterQualityError — quality gate fires on bad adapter
    3. embed() — shadow_mode, dedup across two runs
    4. Ledger — provenance and cost_by_model queries
    5. migrate(dual-write) — shadow_mode, n_migrated == n_source
    6. migrate(drift-adapter) — shadow_mode, ARR == 1.0, .npy written
    7. CLI — all four commands via typer.testing.CliRunner
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

# ── result tracking ───────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
_errors: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}  ← FAILED")
        _errors.append(label)


# ── test data ─────────────────────────────────────────────────────────────────

SINK = "qdrant://localhost:6333/smoke_test"
TOPICS = ["login fails", "billing error", "dark mode request", "outage report", "password reset"]
TEXTS = [f"Support ticket {i}: {TOPICS[i % 5]}" for i in range(30)]
MANY_TEXTS = TEXTS * 3   # 90 texts — enough for 90/10 adapter split


# ════════════════════════════════════════════════════════════════════════════════
# Level 0 — imports + version
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 0: imports + version ──────────────────────────────────────────")

import drift
check(bool(drift.__version__), f"drift.__version__ = {drift.__version__!r}")

from drift import DriftAdapter, AdapterQualityError, measure_arr
check(True, "DriftAdapter, AdapterQualityError, measure_arr importable from drift")

from drift.embed import EmbedRun, embed
from drift.ledger import Ledger
from drift.migrate import MigrateRun, migrate
from drift.watch import WatchRun
check(True, "embed, watch, migrate, ledger all importable")


# ════════════════════════════════════════════════════════════════════════════════
# Level 1 — DriftAdapter (pure NumPy, zero external deps)
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 1: DriftAdapter — pure NumPy ──────────────────────────────────")

d, N = 64, 200
rng = np.random.default_rng(0)
theta = np.radians(30)
R_true = np.eye(d, dtype=np.float32)
R_true[0, 0] = np.cos(theta);  R_true[0, 1] = np.sin(theta)
R_true[1, 0] = -np.sin(theta); R_true[1, 1] = np.cos(theta)

raw = rng.standard_normal((N, d)).astype(np.float32)
X_old = raw / np.linalg.norm(raw, axis=1, keepdims=True)
X_new = X_old @ R_true

split = int(N * 0.9)
adapter = DriftAdapter().fit(X_old[:split], X_new[:split])

check(adapter.R.shape == (d, d), f"R.shape == ({d}, {d})")
check(np.allclose(adapter.R @ adapter.R.T, np.eye(d), atol=1e-3), "R is orthogonal (R @ R.T ≈ I)")

arr = measure_arr(adapter, X_old[split:], X_new[split:], k=5, threshold=None)
check(arr == 1.0, f"ARR == 1.0 for a perfect rotation  (got {arr:.4f})")

q_adapted = adapter.predict(X_new[0])
check(abs(float(q_adapted @ X_old[0]) - 1.0) < 0.01, "predict() maps new→old space (cos_sim ≈ 1.0)")
check(adapter.predict(X_new[0]).shape == (d,), "predict() 1D input → 1D output")
check(adapter.predict(X_new[:5]).shape == (5, d), "predict() (5,d) input → (5,d) output")

with tempfile.TemporaryDirectory() as tmp:
    npy = os.path.join(tmp, "R.npy")
    adapter.save(npy)
    check(Path(npy).exists(), ".npy file written")
    loaded = DriftAdapter.load(npy)
    check(np.allclose(loaded.R, adapter.R), "save → load round-trip: R identical")
    check(np.array_equal(loaded.predict(X_new[0]), adapter.predict(X_new[0])),
          "loaded adapter predict() bit-identical to original")


# ════════════════════════════════════════════════════════════════════════════════
# Level 2 — AdapterQualityError gate
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 2: AdapterQualityError — quality gate ─────────────────────────")

Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
bad_adapter = DriftAdapter.__new__(DriftAdapter)
bad_adapter.R = Q.astype(np.float32)

raised = False
try:
    measure_arr(bad_adapter, X_old[split:], X_old[split:], k=5, threshold=0.97)
except AdapterQualityError as e:
    raised = True
    check(e.arr < 0.97, f"e.arr={e.arr:.4f} is below threshold")
    check(e.threshold == 0.97, f"e.threshold == 0.97")
    check("dual-write" in str(e), "error message names dual-write as fallback")
check(raised, "AdapterQualityError raised for random (untrained) R")

arr_no_gate = measure_arr(bad_adapter, X_old[split:], X_old[split:], k=5, threshold=None)
check(isinstance(arr_no_gate, float) and 0.0 <= arr_no_gate <= 1.0,
      f"threshold=None returns float without raising  (got {arr_no_gate:.4f})")


# ════════════════════════════════════════════════════════════════════════════════
# Level 3 — embed() shadow_mode
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 3: embed() — shadow_mode + dedup ──────────────────────────────")
print("   Starting local Spark session (takes ~10s on first run) ...")

from pyspark.sql import SparkSession  # noqa: E402

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("drift-smoke-test")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")
print("   Spark ready.")

with tempfile.TemporaryDirectory() as tmp:
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")
    df = spark.createDataFrame([{"id": str(i), "body": t} for i, t in enumerate(TEXTS)])

    with patch("drift.embed._upsert_qdrant") as mock_upsert:
        run1 = embed(df, text_col="body", model="openai/text-embedding-3-small",
                     sink=SINK, shadow_mode=True, ledger=ledger)

    check(isinstance(run1, EmbedRun), "embed() returns EmbedRun")
    check(run1.n_rows_processed == 30, f"run1.n_rows_processed == 30  (got {run1.n_rows_processed})")
    check(run1.n_rows_deduped == 0, "run1: first run has 0 deduped rows")
    check(run1.cost_usd == 0.0, "run1: shadow_mode cost_usd == 0.0")
    check(mock_upsert.call_count == 1, "run1: Qdrant upsert called exactly once")

    # Second run — identical data → 100% dedup, zero upserts
    with patch("drift.embed._upsert_qdrant") as mock_upsert2:
        run2 = embed(df, text_col="body", model="openai/text-embedding-3-small",
                     sink=SINK, shadow_mode=True, ledger=ledger)

    check(run2.n_rows_deduped == 30, "run2: 100% dedup (n_rows_deduped == 30)")
    check(mock_upsert2.call_count == 0, "run2: zero Qdrant upserts (nothing new to write)")


# ════════════════════════════════════════════════════════════════════════════════
# Level 4 — Ledger provenance + cost
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 4: Ledger — provenance + cost_by_model ────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")
    df_small = spark.createDataFrame([{"id": "0", "body": "provenance test doc"}])

    captured_points: list = []

    def _capture(sink, points):
        captured_points.extend(points)

    with patch("drift.embed._upsert_qdrant", side_effect=_capture):
        run = embed(df_small, text_col="body", model="openai/text-embedding-3-small",
                    sink=SINK, shadow_mode=True, ledger=ledger)

    embedding_id = captured_points[0]["id"]
    prov = ledger.provenance(embedding_id)

    check(prov is not None, "ledger.provenance() returns a record")
    check(prov["embedding_id"] == embedding_id, "provenance embedding_id matches")
    check(prov["model"] == "openai/text-embedding-3-small", "provenance model correct")
    check(prov["sink"] == SINK, "provenance sink correct")
    check(prov["cost_usd"] == 0.0, "provenance cost_usd == 0.0 (shadow_mode)")
    check(ledger.provenance("does-not-exist") is None, "provenance(unknown) returns None")

    costs = ledger.cost_by_model()
    check(len(costs) == 1, f"cost_by_model returns 1 entry  (got {len(costs)})")
    check(costs[0]["model"] == "openai/text-embedding-3-small", "cost_by_model model correct")

    recent = ledger.recent_runs(sink=SINK)
    check(len(recent) >= 1, "recent_runs returns at least 1 run")
    check(recent[0]["run_id"] == run.run_id, "most recent run_id matches")


# ════════════════════════════════════════════════════════════════════════════════
# Level 5 — migrate(dual-write) shadow_mode
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 5: migrate(dual-write) — shadow_mode ───────────────────────────")

with tempfile.TemporaryDirectory() as tmp:
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")

    with patch("drift.migrate._scroll_qdrant_texts", return_value=TEXTS), \
         patch("drift.embed._upsert_qdrant"):
        run = migrate(
            from_model="openai/text-embedding-ada-002",
            to_model="openai/text-embedding-3-small",
            sink=SINK, strategy="dual-write",
            shadow_mode=True, ledger=ledger,
        )

    check(isinstance(run, MigrateRun), "migrate() returns MigrateRun")
    check(run.n_source == 30, f"n_source == 30  (got {run.n_source})")
    check(run.n_migrated == 30, f"n_migrated == 30  (got {run.n_migrated})")
    check("smoke_test_v2" in run.sink_v2, f"sink_v2 appends _v2  (got {run.sink_v2!r})")
    check(run.adapter_path == "", "dual-write: adapter_path is empty")
    check(run.arr == 0.0, "dual-write: arr == 0.0 (no adapter trained)")
    check(run.duration_s >= 0.0, "duration_s is non-negative")


# ════════════════════════════════════════════════════════════════════════════════
# Level 6 — migrate(drift-adapter) shadow_mode
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 6: migrate(drift-adapter) — shadow_mode ───────────────────────")

_original_cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")

    with patch("drift.migrate._scroll_qdrant_texts", return_value=MANY_TEXTS):
        run = migrate(
            from_model="openai/text-embedding-ada-002",
            to_model="openai/text-embedding-3-small",
            sink=SINK, strategy="drift-adapter",
            shadow_mode=True, ledger=ledger,
        )

    check(run.strategy == "drift-adapter", "run.strategy == 'drift-adapter'")
    check(run.arr == 1.0, f"ARR == 1.0 in shadow_mode  (got {run.arr:.4f})")
    check(run.adapter_path.endswith(".npy"), f"adapter_path ends with .npy  (got {run.adapter_path!r})")
    check(Path(run.adapter_path).exists(), "adapter .npy file written to disk")
    check(run.n_source == 90, f"n_source == 90  (got {run.n_source})")
    check(run.sink_v2 == "", "drift-adapter: sink_v2 is empty (old index untouched)")
    check(run.n_migrated == 0, "drift-adapter: n_migrated == 0 (no new collection)")

    # Load the saved adapter and verify it works at query time
    saved_adapter = DriftAdapter.load(run.adapter_path)
    from drift.embed import _mock_embedding
    q_new = np.array(_mock_embedding("test query"), dtype=np.float32)
    q_adapted = saved_adapter.predict(q_new)
    check(q_adapted.shape == q_new.shape, "saved adapter predict() shape correct")

os.chdir(_original_cwd)


# ════════════════════════════════════════════════════════════════════════════════
# Level 7 — CLI (typer.testing.CliRunner, no subprocess)
# ════════════════════════════════════════════════════════════════════════════════

print("\n── Level 7: CLI — all four commands ────────────────────────────────────")

from typer.testing import CliRunner  # noqa: E402

from drift.cli import app  # noqa: E402

runner = CliRunner()

# drift --help
result = runner.invoke(app, ["--help"])
check(result.exit_code == 0, "drift --help exits 0")
for cmd in ("embed", "watch", "status", "migrate"):
    check(cmd in result.output, f"  drift --help lists '{cmd}'")

# drift migrate --strategy dual-write --shadow-mode
with tempfile.TemporaryDirectory() as tmp:
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")
    with patch("drift.migrate._scroll_qdrant_texts", return_value=TEXTS), \
         patch("drift.embed._upsert_qdrant"), \
         patch("drift.ledger.Ledger", return_value=ledger):
        result = runner.invoke(app, [
            "migrate",
            "--from", "openai/text-embedding-ada-002",
            "--to", "openai/text-embedding-3-small",
            "--sink", SINK, "--strategy", "dual-write", "--shadow-mode",
        ])
    check(result.exit_code == 0, "drift migrate --strategy dual-write exits 0")
    check("30/30" in result.output, "CLI dual-write output shows 30/30 vectors")
    check("Next steps" in result.output, "CLI dual-write output shows Next steps")
    check("Catch-up" in result.output, "CLI dual-write output shows Catch-up instruction")

# drift migrate --strategy drift-adapter --shadow-mode
with tempfile.TemporaryDirectory() as tmp:
    ledger = Ledger(db_path=Path(tmp) / "ledger.db")
    os.chdir(tmp)
    with patch("drift.migrate._scroll_qdrant_texts", return_value=MANY_TEXTS), \
         patch("drift.ledger.Ledger", return_value=ledger):
        result = runner.invoke(app, [
            "migrate",
            "--from", "openai/text-embedding-ada-002",
            "--to", "openai/text-embedding-3-small",
            "--sink", SINK, "--strategy", "drift-adapter", "--shadow-mode",
        ])
    check(result.exit_code == 0,
          f"drift migrate --strategy drift-adapter exits 0\noutput: {result.output[:200]}")
    check("ARR" in result.output, "CLI drift-adapter output shows ARR score")
    check(".npy" in result.output, "CLI drift-adapter output shows adapter path")
    check("adapter.predict" in result.output, "CLI drift-adapter shows query-time usage")
    check("Catch-up" in result.output, "CLI drift-adapter shows catch-up instruction")

os.chdir(_original_cwd)


# ════════════════════════════════════════════════════════════════════════════════
# Results
# ════════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
if _errors:
    print(f"FAILED — {len(_errors)} assertion(s):")
    for e in _errors:
        print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    total = 7
    print(f"ALL {total} LEVELS PASSED — drift-spark v{drift.__version__} is working correctly")
    print()
    print(f"  Level 0: imports + version                  ✓")
    print(f"  Level 1: DriftAdapter (pure NumPy)          ✓")
    print(f"  Level 2: AdapterQualityError gate           ✓")
    print(f"  Level 3: embed() shadow_mode + dedup        ✓")
    print(f"  Level 4: Ledger provenance + cost           ✓")
    print(f"  Level 5: migrate(dual-write) shadow_mode    ✓")
    print(f"  Level 6: migrate(drift-adapter) shadow_mode ✓")
    print(f"  Level 7: CLI all four commands              ✓")
    print()
    print("Next: run against a real Qdrant instance:")
    print("  docker run -p 6333:6333 qdrant/qdrant")
    print("  python examples/quickstart.py")
    print("  python examples/drift_adapter.py --full")
    print("  python integration-tests/it-migrate.py")
    print("  python integration-tests/it-adapter.py")
    sys.exit(0)
