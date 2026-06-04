"""
Drift quickstart — no API key needed (shadow_mode=True).

shadow_mode skips the OpenAI call (zero cost) but still writes real vectors
to the sink, so Qdrant is required either way.

Requires:
    pip install 'drift-spark[spark,qdrant]'
    docker run -p 6333:6333 qdrant/qdrant   # local Qdrant
"""
from pyspark.sql import SparkSession
from drift import embed

spark = SparkSession.builder.master("local[*]").appName("drift-quickstart").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

SINK  = "qdrant://localhost:6333/quickstart_demo"
MODEL = "openai/text-embedding-3-small"

topics = ["login fails after reset", "invoice shows wrong address",
          "dark mode request", "billing discrepancy", "password reset loop"]
docs = [{"id": str(i), "body": f"Support ticket #{i}: {topics[i % 5]}"}
        for i in range(50)]
df = spark.createDataFrame(docs)

print("--- Run 1: embed all 50 docs ---")
r1 = embed(df, text_col="body", model=MODEL, sink=SINK, shadow_mode=True)
print(f"  processed={r1.n_rows_processed}  deduped={r1.n_rows_deduped}  cost=${r1.cost_usd:.4f}")

print("\n--- Run 2: same data → all 50 deduped, zero API calls ---")
r2 = embed(df, text_col="body", model=MODEL, sink=SINK, shadow_mode=True)
print(f"  processed={r2.n_rows_processed}  deduped={r2.n_rows_deduped}  cost=${r2.cost_usd:.4f}")
