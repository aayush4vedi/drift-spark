import os

_JAVA17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JAVA17):
    os.environ["JAVA_HOME"] = _JAVA17

from pyspark.sql import SparkSession
from qdrant_client import QdrantClient
from drift.ledger import Ledger
from drift.watch import watch

# Spark must be configured with Delta extensions — required for delta format + CDF
spark = (
    SparkSession.builder
    .appName("drift-watch-delta-it")
    .master("local[*]")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.0.0")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

ledger = Ledger()
qdrant = QdrantClient("localhost", port=6333)
SINK = "qdrant://localhost:6333/watch_delta_col"

# Step 1: create Delta table with CDF enabled
print("\n=== Step 1: create Delta table with CDF ===")
spark.sql("DROP TABLE IF EXISTS drift_it_docs")
spark.sql("""
    CREATE TABLE drift_it_docs (doc_id INT, body STRING)
    USING delta
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
spark.sql("""
    INSERT INTO drift_it_docs VALUES
    (1, 'reset password'),
    (2, 'billing issue'),
    (3, 'export to CSV')
""")
print("  Table created at version 1 with 3 rows")

# Step 2: first watch() — should embed all 3 inserts
print("\n=== Step 2: first watch() — since_version=0 ===")
run1 = watch(source_table="drift_it_docs", text_col="body",
             sink=SINK, shadow_mode=True, ledger=ledger, since_version=0)
count = qdrant.count("watch_delta_col").count
print(f"  n_inserted={run1.n_inserted}  n_updated={run1.n_updated}  n_deleted={run1.n_deleted}")
print(f"  to_version={run1.to_version}  Qdrant count={count}")
assert run1.n_inserted == 3
assert count == 3

# Step 3: update + delete, then second watch() with auto-checkpoint
print("\n=== Step 3: update doc 1, delete doc 3, second watch() ===")
spark.sql("UPDATE drift_it_docs SET body='reset my password' WHERE doc_id=1")
spark.sql("DELETE FROM drift_it_docs WHERE doc_id=3")

run2 = watch(source_table="drift_it_docs", text_col="body",
             sink=SINK, shadow_mode=True, ledger=ledger)   # since_version=None → auto
count = qdrant.count("watch_delta_col").count
print(f"  n_inserted={run2.n_inserted}  n_updated={run2.n_updated}  n_deleted={run2.n_deleted}")
print(f"  since_version={run2.since_version}  to_version={run2.to_version}  Qdrant count={count}")
assert run2.since_version == run1.to_version   # auto-resolved from checkpoint
assert run2.n_updated == 1
assert run2.n_deleted == 1

print("\n✓ Full watch() Delta CDF loop verified")

spark.sql("DROP TABLE IF EXISTS drift_it_docs")