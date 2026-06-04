# Drift examples

Work through these in order. Each example is a single Python file you can run end-to-end.

## Start here — no API key needed

1. **`quickstart.py`** — first `embed()` call + dedup on the second run, in shadow mode (zero OpenAI cost).
2. **`embed_with_real_api.py`** — same flow with a real `OPENAI_API_KEY` so you see actual `cost_usd`.
3. **`status_and_lineage.py`** — the three ledger queries every user should know: `cost_by_model()`, `recent_runs()`, `provenance()`.

## Incremental refresh

4. **`delta_cdc.py`** — Delta CDF + `watch()`: only changed rows get re-embedded. Run twice to see the diff.

## Model upgrades

5. **`drift_adapter.py`** — rotation-based adapter (Procrustes) for migrating between embedding models without re-indexing your store. Part A is pure NumPy; Part B drives the full pipeline through `migrate()`.
6. **`query_after_migration.py`** — the missing post-migration story. Runs both `drift-adapter` and `dual-write` against the same collection, then shows the exact query-time code each strategy requires.

## Prerequisites at a glance

| Example                       | Qdrant | OpenAI key | Java 17 / Spark | Ledger pre-populated |
|-------------------------------|:------:|:----------:|:---------------:|:--------------------:|
| `quickstart.py`               |   yes  |     no     |       yes       |          no          |
| `embed_with_real_api.py`      |   yes  |     yes    |       yes       |          no          |
| `status_and_lineage.py`       |   no*  |     no     |       yes       |   yes (run #1 first) |
| `delta_cdc.py`                |   yes  |     no     |       yes       |          no          |
| `drift_adapter.py` Part A     |   no   |     no     |       no        |          no          |
| `drift_adapter.py` Part B     |   yes  |     no     |       no        | yes (existing collection) |
| `query_after_migration.py`    |   yes  |     no     |       yes       |          no          |

\* `status_and_lineage.py` reads `~/.drift/ledger.db`; the provenance demo runs in shadow mode against a temp ledger.

## Spinning up Qdrant locally

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Installing extras

```bash
pip install 'drift-spark[spark,qdrant]'        # everything most examples need
pip install 'drift-spark[spark,qdrant,delta]'  # for delta_cdc.py
```

## Where the integration tests live

The full-stack smoke test (`e2e_smoke_test.py`) and the `it-*.py` scripts now live in `integration-tests/`, not here. Those are for maintainers; the files in this folder are for users learning the library.
