"""
Drift embed with a REAL OpenAI key — see actual cost_usd.

The only example that calls the OpenAI Embeddings API and spends real money.
Useful to confirm: (1) what an EmbedRun looks like in production, (2) that
cost_usd is non-zero, (3) that the dedup short-circuits a second run to $0.

Cost: text-embedding-3-small is $0.02 per 1M tokens. 3 short docs ≈ 4 cents
× one-millionth, i.e. effectively free. Re-running the same docs costs $0.

Requires:
    export OPENAI_API_KEY=sk-...
    pip install 'drift-spark[spark,qdrant]'
    docker run -p 6333:6333 qdrant/qdrant
"""
import os
import sys

from pyspark.sql import SparkSession

from drift import embed

if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("Set OPENAI_API_KEY first, or run quickstart.py (shadow mode, no key needed).")

spark = SparkSession.builder.master("local[*]").appName("drift-real-api").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

SINK  = "qdrant://localhost:6333/real_api_demo"
MODEL = "openai/text-embedding-3-small"

docs = [
    {"id": "1", "body": "Customer reports login fails after password reset."},
    {"id": "2", "body": "Invoice for Q1 shows wrong billing address."},
    {"id": "3", "body": "Feature request: dark mode for the dashboard."},
]
df = spark.createDataFrame(docs)

print("--- Run 1: real OpenAI call ---")
r1 = embed(df, text_col="body", model=MODEL, sink=SINK)
print(f"  processed={r1.n_rows_processed}  deduped={r1.n_rows_deduped}")
print(f"  cost=${r1.cost_usd:.8f}  duration={r1.duration_s:.2f}s")

print("\n--- Run 2: same data → 100% dedup, $0 spend, no API call ---")
r2 = embed(df, text_col="body", model=MODEL, sink=SINK)
print(f"  processed={r2.n_rows_processed}  deduped={r2.n_rows_deduped}")
print(f"  cost=${r2.cost_usd:.8f}  duration={r2.duration_s:.2f}s")
