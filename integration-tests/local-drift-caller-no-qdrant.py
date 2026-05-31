# run this from a Python REPL or a .py script inside drift-spark/
from unittest.mock import MagicMock, patch
import pandas as pd

from drift.embed import embed
from drift.ledger import Ledger

# fake Spark DataFrame — wraps a pandas DataFrame
def spark_mock(texts):
    m = MagicMock()
    m.select.return_value.toPandas.return_value = pd.DataFrame({"body": texts})
    return m

ledger = Ledger()   # writes to ~/.drift/ledger.db

with patch("drift.embed._upsert_qdrant") as mock_sink:
    run = embed(
        df=spark_mock(["How do I reset my password?", "My invoice is wrong."]),
        text_col="body",
        model="openai/text-embedding-3-small",
        sink="qdrant://localhost:6333/test_col",
        shadow_mode=True,   # no API key, no cost
        ledger=ledger,
    )

print(f"run_id:    {run.run_id}")
print(f"processed: {run.n_rows_processed}")
print(f"deduped:   {run.n_rows_deduped}")
print(f"cost:      ${run.cost_usd:.4f}")
print(f"duration:  {run.duration_s:.3f}s")
print(f"upsert called: {mock_sink.call_count} time(s)")