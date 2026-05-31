"""
Drift CDC example — Delta CDF → incremental embedding refresh.

Shows: only changed rows are re-embedded. A table with 15 docs where
5 rows are updated and 1 deleted triggers 6 embedding ops, not 15.

Requires:
    pip install 'drift-spark[spark,qdrant,delta]'
    docker run -p 6333:6333 qdrant/qdrant   # local Qdrant
"""
from pyspark.sql import SparkSession
from drift import watch

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.2.0")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

TABLE = "drift_demo_docs"
SINK  = "qdrant://localhost:6333/cdc_demo"
MODEL = "openai/text-embedding-3-small"

# ── seed 15 rows ──────────────────────────────────────────────────────────────
rows = [{"id": str(i), "body": f"Document {i} about topic {i % 5}"} for i in range(15)]
(spark.createDataFrame(rows).write.format("delta")
     .mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE))
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")

# ── first watch: embeds all 15 ─────────────────────────────────────────────
print("--- Watch 1: initial embed (15 docs) ---")
r1 = watch(TABLE, text_col="body", sink=SINK, model=MODEL, shadow_mode=True)
print(f"  inserted={r1.n_inserted}  updated={r1.n_updated}  deleted={r1.n_deleted}")

# ── mutate: update 5 rows, delete 1 ──────────────────────────────────────────
spark.sql(f"UPDATE {TABLE} SET body = 'UPDATED: ' || body WHERE id IN ('0','1','2','3','4')")
spark.sql(f"DELETE FROM {TABLE} WHERE id = '14'")

# ── second watch: only 6 ops (5 updates + 1 delete) ─────────────────────────
print("\n--- Watch 2: only changed rows re-embedded ---")
r2 = watch(TABLE, text_col="body", sink=SINK, model=MODEL, shadow_mode=True)
print(f"  inserted={r2.n_inserted}  updated={r2.n_updated}  deleted={r2.n_deleted}")
print(f"  (9 unchanged rows: never touched)")
