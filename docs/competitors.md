# Competitor Landscape

How Drift compares to adjacent tools in the embedding-pipeline space. Each entry states what the tool ships, where it stops, and how Drift covers the gap — including the cases where a "competitor" is better understood as a complementary component.

<sub>Last reviewed: 2026-05-31 against Mosaic AI Vector Search docs, Daft docs, and the qdrant-spark connector. Reflects `drift-spark` v0.5.0.</sub>

> [!NOTE]
> This is a deliberately critical comparison. Drift's honest gaps are documented in [Limitations](#limitations-in-v050) at the bottom — please read them before adopting.

---

## At a glance

| Capability | Drift | Mosaic AI VS | Daft | qdrant-spark | LanceDB |
|---|:---:|:---:|:---:|:---:|:---:|
| Embedding generation | Yes | No | Yes | No | No |
| Cross-run dedup | Yes | No | Partial | No | No |
| Per-embedding cost ledger | Yes | No | No | No | No |
| CDC / incremental refresh | Yes (triggered) | Yes (continuous) | No | No | Partial |
| Model migration | Yes | No | No | No | No |
| Row-level lineage / audit | Yes | No | No | No | Partial |
| Runs outside Databricks | Yes | No | Yes | Yes | Yes |

Legend: **Yes** — first-class support · **Partial** — possible but user-orchestrated · **No** — not supported.

---

## Detailed comparison

### Mosaic AI Vector Search (Databricks)

- **Runtime.** No ingest-time dedup — deduplication is manual and post-query only. Batching is capped at a 1,024-item filter clause, so users split batches by hand. Cost reporting is limited to daily index-level aggregates. Switching embedding models requires a full rebuild.
- **CDC.** Strong. Delta CDF is mandatory for standard endpoints, with continuous sync (seconds latency) plus triggered sync; insert, update, and delete are all handled. Caveats: storage-optimised endpoints do a partial rebuild on every sync rather than a true incremental update, and there is no CDF backfill for pre-CDF history.
- **Migration.** Full reindex required on model change. The docs are explicit that a self-managed embedding index cannot be converted to a Databricks-managed index — a new index must be created and embeddings recomputed. No dual-write, shadow eval, or gradual rollout.
- **Lineage.** Unity Catalog provides source-table-to-index lineage at the index level only, not per row. No per-embedding cost tracking. GDPR deletion is fully user-orchestrated with no generated compliance proof.
- **Where Drift differs.** Cross-run dedup with per-embedding cost; model migration via dual-write and the Drift-Adapter (Orthogonal Procrustes, [arXiv:2509.23471](https://arxiv.org/abs/2509.23471)); row-level lineage with a compliance audit trail; and it runs on any Spark cluster rather than being Databricks-locked.

### Daft (`getdaft.io`)

- **Runtime.** Strong. Native `embed_text()`, stateful `@daft.cls` UDFs (model loaded once and reused across rows), GPU-aware batching via the Swordfish engine (materially faster than Spark on GPU workloads), and native MinHash dedup. No per-embedding cost tracking or cost ledger. Rate limiting and retries are handled internally.
- **CDC.** Batch dataframe engine, not a streaming runtime — there is no incremental-refresh API. Delta CDF must be handled externally and piped in by the user.
- **Migration.** No model versioning, shadow eval, A/B comparison, or zero-downtime cutover. Switching models means re-running the UDF over all data manually.
- **Lineage.** No embedding lineage or audit trail linking embeddings to source rows, model versions, and cost. External observability (Grafana, Datadog) is required.
- **Where Drift differs.** CDC refresh via Delta CDF (avoiding re-embedding the ~95% of rows that did not change), a queryable cost ledger, model-migration strategies, and embedding provenance for audit and GDPR. Daft could serve as a future batch-runtime substrate for Drift — the two are complementary rather than competing.

### qdrant-spark connector (Qdrant)

- **Runtime.** Not a runtime — a Qdrant sink driver. It accepts a DataFrame that already has an embedding column and writes vectors to a pre-created collection. No embedding generation, dedup, batching control, or cost tracking.
- **CDC.** Stateless across runs, with no source awareness; it cannot detect which rows changed.
- **Migration.** Single-vendor (Qdrant only); out of scope by construction.
- **Lineage.** No memory of past upserts; cannot trace a vector back to its source row, model, cost, or version.
- **Where Drift differs.** Everything above the write call. Drift's Qdrant sink can wrap qdrant-spark internally — they are complementary, not competitors. qdrant-spark is one box; Drift is the surrounding control plane.

### LanceDB (raw)

- **Runtime.** Stores vectors but does not generate them. No embedding UDF, dedup, batching, or cost tracking.
- **CDC.** The versioned table format enables manual snapshot diffing, but there is no first-class CDC API — the user computes the diff.
- **Migration.** No model-upgrade path; changing models requires re-embedding all documents and rebuilding the table.
- **Lineage.** Table versioning gives implicit, snapshot-level lineage only. No per-embedding cost, row-to-vector traceability, or compliance proof.
- **Where Drift differs.** Embedding generation, dedup, and CDC; migration strategies; and per-embedding cost with a compliance trail. LanceDB is a valid future Drift sink target, not a workflow competitor.

### Drift (`drift-spark` v0.5.0)

- **Runtime.** `embed(df, …)` — cross-run dedup (MD5 hash scoped to `(model, sink)`), configurable batching, exponential backoff, and `shadow_mode` for zero-cost local development. OpenAI provider; pgvector and Qdrant sinks.
- **CDC.** `watch(source_table, …)` — Delta CDF to Qdrant incremental refresh with auto-checkpoint via the lineage ledger; insert, update_postimage, and delete handled. CDC delete is not yet supported on the pgvector sink.
- **Migration.** `migrate(strategy="dual-write")` performs a full reindex into a new collection; `migrate(strategy="drift-adapter")` uses Orthogonal Procrustes (95–99% recall, ~15s on CPU, no reindex). A `shadow-eval` strategy is planned for v0.6.
- **Lineage.** SQLite lineage ledger with `embed_runs` and `embedding_provenance` tables; helpers `cost_by_model()`, `provenance(embedding_id)`, and `recent_runs(sink)`; plus a `watch_runs` table holding the checkpoint per `(source_table, sink)`.

---

## Strongest competitor per dimension

| Dimension | Strongest competitor | Drift's position |
|---|---|---|
| Runtime (batch embedding) | Daft — faster than Spark, native MinHash dedup | Drift owns the control layer on top of whatever runtime executes; Daft could be a future substrate. |
| CDC | Mosaic AI VS — genuine continuous sync, seconds latency | Drift's `watch()` is triggered rather than continuous, but is open source and cross-vendor — any Spark plus Qdrant/pgvector, not Databricks-only. |
| Migration | None — every competitor leaves this to the user | `drift migrate` is the only OSS library wrapping dual-write and Procrustes-based space alignment; shadow eval is planned for v0.6. |
| Lineage | None — every competitor leaves this to the user | Drift records a per-embedding audit trail (cost, provenance, model version) in the SQLite lineage ledger — no competitor offers an equivalent. |

---

## Limitations in v0.5.0

These are the trade-offs to weigh before adopting Drift today:

- **CDC is triggered, not continuous.** For Databricks shops that need seconds-fresh indexes, Mosaic AI Vector Search wins on freshness.
- **pgvector is partial.** CDC delete and `migrate()` are Qdrant-only for now; pgvector is write-only.
- **Batch throughput.** Daft is materially faster than Spark on GPU workloads.

---

<sub><a href="../README.md">← Back to README</a></sub>
