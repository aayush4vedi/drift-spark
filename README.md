# drift-spark

> **Spark-native embedding lifecycle** — produce, CDC refresh, model-migrate, audit.

`pip install drift-spark` · `import drift` · MIT

---

**Status: pre-alpha (v0.0.1 placeholder).** 

Drift is a Python library that turns the standard 300-line PySpark embedding pipeline into three declarative commands:

```bash
drift embed --table my_catalog.docs --text-col body --sink qdrant://localhost:6333/docs
drift watch --table my_catalog.docs --text-col body --sink qdrant://localhost:6333/docs
drift status --sink qdrant://localhost:6333/docs
```

**What it does:**
- `embed()` — Spark-native embedding with dedup, batching, multi-model, Qdrant + pgvector sinks
- `watch()` — incremental CDC refresh via Delta Change Data Feed → only changed rows re-embedded
- `migrate()` — dual-write model migration with lineage tracking (v1.0); adapter projection (v2.0)
- Lineage ledger — per-embedding cost, source tracing, GDPR-delete proof (SQLite, queryable)

**GitHub:** https://github.com/aayush4vedi/drift-spark
