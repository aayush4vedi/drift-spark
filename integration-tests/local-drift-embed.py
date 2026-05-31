import os

_JAVA17 = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
if os.path.isdir(_JAVA17):
    os.environ["JAVA_HOME"] = _JAVA17

from pyspark.sql import SparkSession
import pandas as pd
from drift.embed import embed
from drift.ledger import Ledger

spark = SparkSession.builder.appName("drift-test").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.createDataFrame(
    pd.DataFrame({"doc_id": [0, 1, 2],
                  "body": ["reset password", "billing issue", "export CSV"]})
)

run = embed(
    df=df,                                        # pass df directly — no table lookup
    text_col="body",
    model="openai/text-embedding-3-small",
    sink="qdrant://localhost:6333/demo_col",
    shadow_mode=True,
    ledger=Ledger(),
)
print(run)