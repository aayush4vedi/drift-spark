import pytest
from drift.embed import EmbedRun, embed


def test_embed_run_defaults():
    run = EmbedRun(model="openai/text-embedding-3-small", sink="qdrant://localhost/test")
    assert run.run_id  # non-empty UUID
    assert run.timestamp  # non-empty ISO string
    assert run.n_rows_processed == 0


def test_embed_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        embed(df=None, text_col="body", model="openai/text-embedding-3-small",
              sink="qdrant://localhost/test")