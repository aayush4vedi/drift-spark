import os

_JAVA17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JAVA17):
    os.environ["JAVA_HOME"] = _JAVA17

from pyspark.sql import SparkSession
import pandas as pd
from qdrant_client import QdrantClient

from drift.embed import embed
from drift.ledger import Ledger
from drift.watch import WatchRun, _delete_from_sink

spark = SparkSession.builder.appName("drift-watch-it").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
ledger = Ledger()
qdrant = QdrantClient("localhost", port=6333)
SINK = "qdrant://localhost:6333/watch_it_col"

# Step 1: embed 3 docs (what watch() calls for inserts)
print("\n=== Step 1: embed 3 docs ===")
df = spark.createDataFrame(pd.DataFrame({"body": ["doc A", "doc B", "doc C"]}))
run = embed(df=df, text_col="body", model="openai/text-embedding-3-small",
            sink=SINK, shadow_mode=True, ledger=ledger)
count = qdrant.count("watch_it_col").count
print(f"  embedded: {run.n_rows_processed}  deduped: {run.n_rows_deduped}")
print(f"  Qdrant count: {count}")
assert count == 3, f"Expected 3 vectors, got {count}"

# Step 2: delete one doc (what watch() calls for deletes)
print("\n=== Step 2: delete 'doc A' ===")
n_deleted = _delete_from_sink(SINK, ["doc A"])
count = qdrant.count("watch_it_col").count
print(f"  deleted: {n_deleted}  Qdrant count now: {count}")
assert count == 2, f"Expected 2 vectors after delete, got {count}"

# Step 3: checkpoint write + auto-resolve
print("\n=== Step 3: checkpoint ===")
w = WatchRun(source_table="demo.docs", sink=SINK,
             since_version=0, to_version=7, n_inserted=3, n_deleted=1)
ledger.write_watch_run(w)
resolved = ledger.last_watch_version("demo.docs", SINK)
print(f"  last_watch_version: {resolved}")
assert resolved == 7, f"Expected checkpoint 7, got {resolved}"

print("\n✓ All watch() components verified end-to-end")