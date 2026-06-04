"""
Drift end-to-end: migrate, then query the new model's embeddings.

Walks through the *complete* post-migration story that the other examples
only hint at:

    PHASE 1  Seed a collection with the OLD model.
    PHASE 2  Run BOTH migration strategies side-by-side:
             - drift-adapter  → keeps old collection, saves an adapter .npy
             - dual-write     → creates a sibling <collection>_v2
    PHASE 3  Query each successfully — show the exact code your app must run
             after a migration:
             - adapter flow: embed query with new model → rotate → search old
             - dual-write flow: embed query with new model → search _v2

Shadow-mode caveat: drift's shadow vectors are deterministic per text and
independent of the model, so in this demo old and new vectors are identical
and the adapter rotation is the identity. The plumbing is real; the math is
trivial. In production, run with shadow_mode=False and a real OPENAI_API_KEY
to see the rotation do meaningful work.

Requires:
    pip install 'drift-spark[spark,qdrant]'
    docker run -p 6333:6333 qdrant/qdrant
"""
import os
from urllib.parse import urlparse

from pyspark.sql import SparkSession
from qdrant_client import QdrantClient

from drift import embed
from drift.adapter import DriftAdapter
from drift.embed import _mock_embedding   # stands in for openai.embed() under shadow
from drift.migrate import migrate

SINK       = "qdrant://localhost:6333/migration_demo"
OLD_MODEL  = "openai/text-embedding-ada-002"
NEW_MODEL  = "openai/text-embedding-3-small"
QUERY_TEXT = "password reset is broken"

u = urlparse(SINK)
qdrant = QdrantClient(host=u.hostname, port=u.port)
collection = u.path.strip("/")

# Clean slate so the demo is reproducible.
for name in (collection, f"{collection}_v2"):
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)

spark = SparkSession.builder.master("local[*]").appName("drift-postmigrate").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


# ── PHASE 1: seed the collection with the OLD model ──────────────────────────
print("=" * 64)
print("PHASE 1: seed `migration_demo` with the OLD model")
print("=" * 64)

topics = ["login fails after reset", "invoice shows wrong address",
          "dark mode request", "billing discrepancy", "password reset loop"]
docs = [{"id": str(i), "body": f"Support ticket #{i}: {topics[i % 5]}"}
        for i in range(50)]
df = spark.createDataFrame(docs)

seed_run = embed(df, text_col="body", model=OLD_MODEL, sink=SINK, shadow_mode=True)
print(f"  seeded {seed_run.n_rows_processed} docs into `{collection}` "
      f"using {OLD_MODEL}")


# ── PHASE 2a: drift-adapter migration ────────────────────────────────────────
print("\n" + "=" * 64)
print("PHASE 2a: migrate(strategy='drift-adapter')")
print("=" * 64)

adapter_run = migrate(
    from_model=OLD_MODEL,
    to_model=NEW_MODEL,
    sink=SINK,
    strategy="drift-adapter",
    shadow_mode=True,
)
print(f"  ARR:          {adapter_run.arr:.4f}  (gate ≥ 0.97)")
print(f"  adapter:      {adapter_run.adapter_path}")
print(f"  collection `{collection}` is UNTOUCHED — no re-indexing happened.")


# ── PHASE 2b: dual-write migration ───────────────────────────────────────────
print("\n" + "=" * 64)
print("PHASE 2b: migrate(strategy='dual-write')")
print("=" * 64)

dw_run = migrate(
    from_model=OLD_MODEL,
    to_model=NEW_MODEL,
    sink=SINK,
    strategy="dual-write",
    shadow_mode=True,
)
print(f"  n_source:     {dw_run.n_source}")
print(f"  n_migrated:   {dw_run.n_migrated}")
print(f"  sink_v2:      {dw_run.sink_v2}")
print(f"  new collection `{collection}_v2` now holds new-model vectors.")


# ── PHASE 3a: query via the ADAPTER (collection unchanged) ───────────────────
print("\n" + "=" * 64)
print("PHASE 3a: query through the ADAPTER")
print("=" * 64)
print(f"  user query: {QUERY_TEXT!r}")

# In production, this line is: openai.embeddings.create(model=NEW_MODEL, ...)
q_new = _mock_embedding(QUERY_TEXT)

adapter = DriftAdapter.load(adapter_run.adapter_path)
q_rotated = adapter.predict(q_new).tolist()         # ← rotate into old model's space

hits = qdrant.query_points(
    collection_name=collection,                     # ← original collection
    query=q_rotated,
    limit=3,
).points
print(f"\n  top-3 from `{collection}` (rotated query):")
for h in hits:
    print(f"    score={h.score:.4f}  {h.payload['source_text']}")


# ── PHASE 3b: query DIRECTLY against the _v2 collection ──────────────────────
print("\n" + "=" * 64)
print("PHASE 3b: query the new collection DIRECTLY (dual-write cutover)")
print("=" * 64)
print(f"  user query: {QUERY_TEXT!r}")

q_new_v2 = _mock_embedding(QUERY_TEXT)              # same call as 3a

hits = qdrant.query_points(
    collection_name=f"{collection}_v2",             # ← new collection, no rotation
    query=q_new_v2,
    limit=3,
).points
print(f"\n  top-3 from `{collection}_v2`:")
for h in hits:
    print(f"    score={h.score:.4f}  {h.payload['source_text']}")


# ── Recap ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 64)
print("Picking between the two in production:")
print("=" * 64)
print(f"""
  drift-adapter  → cheap (one .npy file), instant rollback (delete the file),
                   old collection stays canonical. Use when ARR ≥ 0.97.

  dual-write     → full reindex into `<collection>_v2`. Costs another
                   round-trip of embedding API spend. Use when adapter ARR
                   falls below threshold, or when you want the new model's
                   raw geometry (not a rotation of it) for downstream use.

  Both: rerun watch() against the chosen sink to catch any docs added
  during migration, then flip your app's collection name (or load the
  adapter at startup).
""")

# Cleanup left-behind adapter file from the demo
if os.path.exists(adapter_run.adapter_path):
    os.remove(adapter_run.adapter_path)
