"""
Migration integration test — dual-write strategy end-to-end.

Tests the full migrate() flow against a real Qdrant instance.
shadow_mode=True is used throughout so no OPENAI_API_KEY is needed.

Requires:
    pip install 'drift-spark[spark,qdrant]'
    docker run -p 6333:6333 qdrant/qdrant

Run:
    python integration-tests/it-migrate.py

Levels:
    1. Seed docs → migrate() → verify _v2 collection has same count
    2. Empty collection → migrate() → n_source=0, no crash
    3. Idempotent re-migration → running migrate() again on same source is safe
    4. Backfill gap simulation → seed → migrate → add more docs → re-migrate → counts reconcile
"""

import sys
from pathlib import Path

from pyspark.sql import SparkSession
from qdrant_client import QdrantClient

from drift.embed import embed
from drift.ledger import Ledger
from drift.migrate import migrate

# ── Spark setup ───────────────────────────────────────────────────────────────

spark = (
    SparkSession.builder.master("local[*]")
    .appName("drift-it-migrate")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Config ────────────────────────────────────────────────────────────────────

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
MODEL_OLD   = "openai/text-embedding-3-small"
MODEL_NEW   = "openai/text-embedding-3-large"   # simulated via shadow_mode

SINK_OLD = f"qdrant://{QDRANT_HOST}:{QDRANT_PORT}/migrate_it_source"
SINK_V2  = f"qdrant://{QDRANT_HOST}:{QDRANT_PORT}/migrate_it_source_v2"

client  = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
ledger  = Ledger(db_path=Path("/tmp/drift-it-migrate/ledger.db"))

PASS = "✓"
FAIL = "✗"
errors: list[str] = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"  {PASS} {label}")
    else:
        print(f"  {FAIL} {label}  ← FAILED")
        errors.append(label)


def _cleanup(*collections: str) -> None:
    for coll in collections:
        try:
            client.delete_collection(coll)
        except Exception:
            pass


def _seed(n: int, sink: str, start: int = 0) -> None:
    """Seed n docs into sink using shadow_mode embed."""
    rows = [{"id": str(start + i), "body": f"Support ticket {start + i}: topic {(start + i) % 5}"}
            for i in range(n)]
    df = spark.createDataFrame(rows)
    embed(df, text_col="body", model=MODEL_OLD, sink=sink, shadow_mode=True,
          dedup=False, ledger=ledger)


def _count(collection: str) -> int:
    try:
        return client.count(collection).count
    except Exception:
        return 0


# ── Level 1: basic migration ──────────────────────────────────────────────────

print("\n=== Level 1: seed 20 docs → migrate → verify _v2 has 20 vectors ===")
_cleanup("migrate_it_source", "migrate_it_source_v2")

_seed(20, SINK_OLD)
n_old = _count("migrate_it_source")
print(f"  Seeded {n_old} docs into migrate_it_source")
check(n_old == 20, "old collection has 20 vectors after seeding")

r = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
            strategy="dual-write", shadow_mode=True, ledger=ledger)

check(r.n_source == 20,   f"n_source == 20  (got {r.n_source})")
check(r.n_migrated == 20, f"n_migrated == 20  (got {r.n_migrated})")
check(r.n_source == r.n_migrated, "n_source == n_migrated (no silent drops)")
check(r.sink_v2 == SINK_V2, f"sink_v2 == {SINK_V2}")
check(_count("migrate_it_source_v2") == 20, "Qdrant _v2 collection has 20 vectors")
check(r.duration_s > 0, "duration_s is positive")

# ── Level 2: empty old collection ────────────────────────────────────────────

print("\n=== Level 2: empty source collection → migrate → n_source=0, no crash ===")
_cleanup("migrate_it_source", "migrate_it_source_v2")

# Don't seed anything — old collection is empty (or doesn't exist)
r = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
            strategy="dual-write", shadow_mode=True, ledger=ledger)

check(r.n_source == 0,   f"n_source == 0  (got {r.n_source})")
check(r.n_migrated == 0, f"n_migrated == 0  (got {r.n_migrated})")
check(_count("migrate_it_source_v2") == 0, "_v2 collection is empty (no crash)")

# ── Level 3: idempotent re-migration ─────────────────────────────────────────

print("\n=== Level 3: run migrate() twice → _v2 count stable (idempotent) ===")
_cleanup("migrate_it_source", "migrate_it_source_v2")

_seed(10, SINK_OLD)

r1 = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
             strategy="dual-write", shadow_mode=True, ledger=ledger)
count_after_first = _count("migrate_it_source_v2")

# Re-run migrate — embed() uses dedup=False, so all 10 are re-upserted
# Qdrant upsert is idempotent (same point_id → overwrites), count stays 10
r2 = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
             strategy="dual-write", shadow_mode=True, ledger=ledger)
count_after_second = _count("migrate_it_source_v2")

check(count_after_first == 10,  f"_v2 has 10 after first migration  (got {count_after_first})")
check(count_after_second == 10,
      f"_v2 still has 10 after second migration  (got {count_after_second})")
check(r2.n_source == 10,   f"second migrate() n_source == 10  (got {r2.n_source})")

# ── Level 4: backfill gap simulation ─────────────────────────────────────────

print("\n=== Level 4: backfill gap — docs added during migration are not lost ===")
print("    (simulated: seed 20 → migrate → seed 5 more → re-migrate → _v2 has 25)")
_cleanup("migrate_it_source", "migrate_it_source_v2")

# Seed initial 20 docs
_seed(20, SINK_OLD, start=0)

# Migrate the initial 20
r_initial = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
                    strategy="dual-write", shadow_mode=True, ledger=ledger)
check(r_initial.n_migrated == 20, f"initial migration got 20  (got {r_initial.n_migrated})")

# Simulate docs arriving during migration — add 5 more to source
_seed(5, SINK_OLD, start=20)
n_source_total = _count("migrate_it_source")
check(n_source_total == 25, f"source now has 25 docs  (got {n_source_total})")

# Catch-up: re-run migrate (scrolls current state of source — all 25)
# In production you'd use drift watch for this; here we simulate with a re-migrate
r_catchup = migrate(from_model=MODEL_OLD, to_model=MODEL_NEW, sink=SINK_OLD,
                    strategy="dual-write", shadow_mode=True, ledger=ledger)
n_v2_final = _count("migrate_it_source_v2")

check(r_catchup.n_source == 25,  f"catch-up sees 25 source docs  (got {r_catchup.n_source})")
check(r_catchup.n_migrated == 25, f"catch-up migrates 25  (got {r_catchup.n_migrated})")
check(n_v2_final == 25, f"_v2 finally has 25 vectors  (got {n_v2_final})")

# ── Results ───────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
if errors:
    print(f"FAILED — {len(errors)} assertion(s):")
    for e in errors:
        print(f"  {FAIL} {e}")
    sys.exit(1)
else:
    print("ALL LEVELS PASSED")
    print(f"  Level 1: basic migration (20 docs)  ✓")
    print(f"  Level 2: empty collection (0 docs)  ✓")
    print(f"  Level 3: idempotent re-migration     ✓")
    print(f"  Level 4: backfill gap simulation     ✓")
    sys.exit(0)
