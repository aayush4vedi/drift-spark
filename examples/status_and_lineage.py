"""
Drift lineage ledger — cost reports, recent runs, per-vector provenance.

The three ledger queries every Drift user should know:
  1. cost_by_model()         — for finance / chargeback / monthly reports
  2. recent_runs(sink, N)    — for `drift status`, debugging, dashboards
  3. provenance(embedding_id) — for GDPR delete-proofs, audit trails

Requires:
    pip install drift-spark
    (Run examples/quickstart.py at least once so ~/.drift/ledger.db has data.)
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

from drift.ledger import Ledger


# ── Query 1: cost_by_model ───────────────────────────────────────────────────
print("=" * 60)
print("1. Cost by model (from ~/.drift/ledger.db)")
print("=" * 60)

ledger = Ledger()
costs = ledger.cost_by_model()

if not costs:
    sys.exit("\nLedger is empty. Run examples/quickstart.py first, then re-run this.")

for row in costs:
    print(f"  {row['model']:40s}  ${row['cost_usd']:.6f}")


# ── Query 2: recent_runs(sink) ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Recent runs (per sink)")
print("=" * 60)

# Pick the first sink that appears in the ledger
sink = ledger._conn.execute("SELECT sink FROM embed_runs LIMIT 1").fetchone()[0]
print(f"\nLast 3 runs for sink={sink}:")
for r in ledger.recent_runs(sink, limit=3):
    print(f"  {r['timestamp'][:19]}  {r['model']}")
    print(f"    rows={r['n_rows']}  deduped={r['n_deduped']}  "
          f"cost=${r['cost_usd']:.6f}  duration={r['duration_s']:.2f}s")


# ── Query 3: provenance(embedding_id) ────────────────────────────────────────
# Self-contained: run an embed against a temp ledger, capture the upserted
# vector's ID, then look it up. The temp ledger keeps your real one clean.
print("\n" + "=" * 60)
print("3. Per-vector provenance — the GDPR / audit answer")
print("=" * 60)

from pyspark.sql import SparkSession

from drift import embed

spark = SparkSession.builder.master("local[*]").appName("drift-lineage").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

demo_db = Path(tempfile.mkdtemp()) / "demo.db"
demo = Ledger(db_path=demo_db)
df = spark.createDataFrame([{"id": "u-123", "body": "User U-123's support ticket"}])

with patch("drift.embed._upsert_qdrant") as mock_upsert:
    embed(df, text_col="body",
          model="openai/text-embedding-3-small",
          sink="qdrant://demo/lineage_demo",
          shadow_mode=True, ledger=demo)
    embedding_id = mock_upsert.call_args[0][1][0]["id"]

print(f"\nLooking up provenance for embedding_id={embedding_id}:")
for k, v in demo.provenance(embedding_id).items():
    print(f"  {k:18s}  {v}")

print("\n→ Given a vector's ID from your sink, you can answer:")
print("    - which source text produced it?       (source_hash)")
print("    - which model + run created it?        (model, run_id)")
print("    - is it still in scope for deletion?   (compare source_hash with hash_exists)")
