from drift.embed import embed
from drift.ledger import Ledger
import pandas as pd
from unittest.mock import MagicMock

def spark_mock(texts):
    m = MagicMock()
    m.select.return_value.toPandas.return_value = pd.DataFrame({"body": texts})
    return m

run = embed(
    df=spark_mock(["reset password", "billing issue", "export to CSV"]),
    text_col="body",
    model="openai/text-embedding-3-small",
    sink="qdrant://localhost:6333/demo_col",
    shadow_mode=True,
    ledger=Ledger(),
)
print(run)