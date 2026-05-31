# Drift — Spark-native embedding lifecycle

> **dbt for embeddings. Terraform for vector indexes.**

`pip install drift-spark` · MIT · [PyPI](https://pypi.org/project/drift-spark/)

Drift is a Python library that turns the standard 300-line PySpark embedding pipeline
into three declarative commands — and adds the CDC refresh, cost ledger, and model
migration tooling the artisanal script never had.

---

## Install

```bash
pip install drift-spark

# with Qdrant sink
pip install 'drift-spark[qdrant]'

# with pgvector sink
pip install 'drift-spark[pgvector]'

# full install (Spark + all sinks)
pip install 'drift-spark[spark,qdrant,pgvector]'
```

---

## Quickstart

No `OPENAI_API_KEY` needed for local development — `shadow_mode=True` uses deterministic
mock vectors at zero cost. Dedup and provenance work identically.

```python
from pyspark.sql import SparkSession
from drift import embed, watch
from drift.ledger import Ledger

spark = SparkSession.builder.master("local[*]").getOrCreate()

df = spark.createDataFrame([
    {"id": "1", "body": "Customer reports login issue after password reset."},
    {"id": "2", "body": "Invoice for Q1 shows wrong billing address."},
    {"id": "3", "body": "Feature request: dark mode for the dashboard."},
])

# --- Run 1: embed all 3 rows ---
run = embed(
    df=df,
    text_col="body",
    model="openai/text-embedding-3-small",
    sink="qdrant://localhost:6333/demo",
    shadow_mode=True,          # no API key needed
)
print(run)
# EmbedRun(n_rows_processed=3, n_rows_deduped=0, cost_usd=0.0, duration_s=0.14)

# --- Run 2: same data, everything deduped ---
run2 = embed(df=df, text_col="body", model="openai/text-embedding-3-small",
             sink="qdrant://localhost:6333/demo", shadow_mode=True)
print(run2)
# EmbedRun(n_rows_processed=3, n_rows_deduped=3, cost_usd=0.0, duration_s=0.03)

# --- CDC refresh: only changed rows ---
watch_run = watch(
    source_table="catalog.support_docs",   # Delta table with CDF enabled
    text_col="body",
    sink="qdrant://localhost:6333/demo",
    model="openai/text-embedding-3-small",
    shadow_mode=True,
)
print(watch_run)
# WatchRun(n_inserted=31200, n_updated=18800, n_deleted=412, duration_s=4.1)
```

Or via CLI:

```bash
drift embed --table catalog.support_docs --text-col body \
            --model openai/text-embedding-3-small \
            --sink qdrant://localhost:6333/support_docs --shadow-mode

drift watch --table catalog.support_docs --text-col body \
            --sink qdrant://localhost:6333/support_docs

drift status --sink qdrant://localhost:6333/support_docs
```

---

## Why Drift exists

Every data team building RAG has a script like this:

```python
df = spark.read.table("catalog.support_docs")   # reads ALL 10M rows
rows = df.select("doc_id", "body").toPandas()

for batch in chunked(rows["body"].tolist(), 512):
    resp = openai.embeddings.create(model="text-embedding-3-small", input=batch)
    qdrant.upsert(collection_name="support_docs", points=[...])
```

It was written by someone who has since left. It re-embeds all 10M rows every night even
though 95% are unchanged — wasting ~$190/run on `text-embedding-3-small`. Nobody can
tell finance which table cost how much last week. OpenAI deprecated `text-embedding-ada-002`
six months ago and the migration still hasn't happened because nobody wants to own the
weekend risk. A GDPR delete request came in last month and the team still cannot prove the
vector was removed.

Drift fixes all six of these problems with three functions.

---

## What's in the box

### Subsystem 1 — `embed()`: the runtime

Replaces the hand-rolled `pandas_udf` with a declarative call. Handles cross-run dedup
(MD5 hash per text scoped to `(model, sink)` — if the text was already embedded in a
prior run, the API call is skipped), configurable batching, exponential backoff on 429s,
idempotent point IDs (deterministic UUID from text hash — retry-safe), and per-run cost
tracking. `shadow_mode=True` runs without any API key using deterministic mock vectors —
identical texts produce identical vectors, so dedup and provenance work correctly in CI.

### Subsystem 2 — `watch()`: incremental CDC refresh

Reads Delta Change Data Feed from the last committed checkpoint and embeds only the rows
that changed. A 10M-row table with 5% daily churn: `embed()` costs ~$40/run, `watch()`
costs ~$2/run. Handles `insert`, `update_postimage`, and `delete` — deleted source rows
have their Qdrant vectors removed via the same deterministic point ID formula. The
checkpoint (Delta version watermark) is written to the lineage ledger so each run picks
up exactly where the last one left off.

### Subsystem 3 — `migrate()`: model upgrade plane

When the embedding model changes, Drift knows which vectors need re-embedding (from the
lineage ledger) and provides migration strategies. **v0.3: dual-write strategy is
documented but exits with `NotImplementedError` — shipping in v0.4.** v2 target:
shadow eval (run queries against both collections, report recall@k delta) and
Drift-Adapter projection (train a lightweight MLP to map new model queries into old
model space — 95–99% recall at 1/100th the reindex cost, per EMNLP 2025 paper 2509.23471).

### Subsystem 4 — Lineage ledger: cost, provenance, compliance

Every `embed()` and `watch()` run writes to a local SQLite ledger at `~/.drift/ledger.db`.
Queryable from Python:

```python
from drift.ledger import Ledger
ledger = Ledger()

# cost by model
ledger.cost_by_model()
# [{'model': 'openai/text-embedding-3-small', 'cost_usd': 4.27}]

# full lineage for a single vector (GDPR audit)
ledger.provenance("3f2a1b8c-...")
# {'embedding_id': '3f2a1b...', 'source_hash': 'abc...', 'model': '...', 'cost_usd': 0.0038}

# last 5 runs for a sink
ledger.recent_runs("qdrant://localhost:6333/support_docs")
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Sources                                │
│      Delta table (CDF)      Iceberg       Postgres            │
│         [live]            [planned]      [planned]            │
└──────────────────────────────┬───────────────────────────────┘
                               │
               ┌───────────────▼──────────────────┐
               │          drift watch()             │  Subsystem 2
               │  · reads Delta CDF                 │  CDC refresh plane
               │  · filters insert / update /       │
               │    update_postimage / delete        │
               │  · auto-checkpoints version         │
               │    in lineage ledger                │
               └───────────────┬──────────────────┘
                               │  changed rows only (~5%)
               ┌───────────────▼──────────────────┐
               │          drift embed()             │  Subsystem 1
               │  · MD5 cross-run dedup             │  Embedding runtime
               │  · batched API calls (128/req)     │
               │  · exponential backoff on 429      │
               │  · shadow_mode for $0 local dev    │
               └────────┬────────────┬─────────────┘
                        │            │
             ┌──────────▼──┐  ┌──────▼──────────┐
             │   Qdrant     │  │    pgvector      │   Sinks (v0.3)
             │  (live v0.3) │  │   (live v0.3)   │
             └──────────────┘  └─────────────────┘
                        │            │
               ┌────────▼────────────▼────────────┐
               │         Lineage Ledger             │  Subsystem 4
               │  embed_runs  · run_id, cost_usd    │  SQLite
               │  provenance  · embedding → source  │  ~/.drift/ledger.db
               │  watch_runs  · checkpoint version  │  queryable via Python
               └──────────────────────────────────┘

               ┌──────────────────────────────────┐
               │         drift migrate()            │  Subsystem 3
               │  · reads ledger: which vectors     │  Migration plane
               │    need re-embedding               │
               │  · dual-write strategy  [v0.4]     │
               │  · shadow eval + Drift-Adapter [v2]│
               └──────────────────────────────────┘
```

---

## API reference

### `embed(df, text_col, model, sink, *, dedup, batch_size, shadow_mode, source_table, ledger) → EmbedRun`

| Parameter | Default | Description |
|---|---|---|
| `df` | — | PySpark DataFrame (or `None` when `source_table` is given) |
| `text_col` | — | Column name containing the text to embed |
| `model` | — | `"provider/model-name"` e.g. `"openai/text-embedding-3-small"` |
| `sink` | — | `"qdrant://host:port/collection"` or `"pg://..."` |
| `dedup` | `True` | Skip rows already embedded with this `(model, sink)` pair |
| `batch_size` | `128` | Texts per API call (OpenAI max: 2048) |
| `shadow_mode` | `False` | Deterministic mock vectors — no API key, zero cost |

Returns `EmbedRun(run_id, n_rows_processed, n_rows_deduped, cost_usd, duration_s)`.

### `watch(source_table, text_col, sink, *, model, since_version, shadow_mode, ledger) → WatchRun`

| Parameter | Default | Description |
|---|---|---|
| `source_table` | — | Delta table name (must have CDF enabled) |
| `text_col` | — | Column to embed |
| `sink` | — | Sink URI |
| `model` | `"openai/text-embedding-3-small"` | Embedding model |
| `since_version` | `None` | Delta version to start from (auto-resolved from ledger) |

Returns `WatchRun(n_inserted, n_updated, n_deleted, since_version, to_version, duration_s)`.

### CLI

```
drift embed   --table TABLE --text-col COL --model MODEL --sink URI [--shadow-mode]
drift watch   --table TABLE --text-col COL --sink URI [--since-version N] [--shadow-mode]
drift status  --sink URI
drift migrate --from MODEL --to MODEL --sink URI --strategy dual-write  # stub in v0.3
```

---

## Roadmap

| Version | What ships |
|---|---|
| **v0.3** (current) | `embed()` with cross-run dedup, batching, shadow_mode · `watch()` Delta CDF → Qdrant/pgvector · Lineage ledger (SQLite) · CLI (`embed`, `watch`, `status`, `migrate` stub) |
| **v0.4** | `migrate --strategy dual-write` implemented · pgvector CDC · chunk-level delta planning |
| **v2** | Shadow eval (recall@k delta before cutover) · Drift-Adapter projection (95–99% recall at 1/100× reindex cost, [EMNLP 2025](https://arxiv.org/abs/2509.23471)) · LanceDB sink |
| **v3+** | Hosted lineage dashboard · cost alerts · Iceberg + Postgres CDC sources |

---

## How Drift is different

| | Drift | Mosaic AI VS | qdrant-spark | Daft |
|---|---|---|---|---|
| Embedding generation + dedup | ✅ | ❌ | ❌ | ✅ (faster) |
| CDC refresh | ✅ triggered | ✅ continuous | ❌ | ❌ |
| Model migration | ⚠️ stub→v0.4 | ❌ full reindex | ❌ | ❌ |
| Per-embedding lineage + cost | ✅ | ❌ | ❌ | ❌ |
| Runs outside Databricks | ✅ | ❌ | ✅ | ✅ |

Full adversarial breakdown: [docs/competitors.md](docs/competitors.md)

---

## Contributing

Drift is MIT-licensed. Issues and PRs welcome.

```bash
git clone https://github.com/aayush4vedi/drift-spark
cd drift-spark
pip install -e '.[spark,qdrant,pgvector]'
pytest tests/          # unit tests (no Docker, no API key)
```

Integration tests (requires local Qdrant + Delta table):

```bash
python integration-tests/it-embed-components.py
python integration-tests/it-watch-delta.py
```
