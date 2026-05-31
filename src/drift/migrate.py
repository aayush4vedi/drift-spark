"""Subsystem 3: model-upgrade migration — dual-write strategy."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

STRATEGIES = ("dual-write", "shadow-eval", "drift-adapter")


@dataclass
class MigrateRun:
    """Returned by migrate() — the record of a model-upgrade run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    from_model: str = ""
    to_model: str = ""
    strategy: str = "dual-write"
    sink: str = ""       # original collection URI
    sink_v2: str = ""    # new collection URI (<collection>_v2)
    n_source: int = 0    # vectors scrolled from old collection
    n_migrated: int = 0  # vectors written to new collection (should == n_source)
    duration_s: float = 0.0


# ── sink helpers ──────────────────────────────────────────────────────────────

def _scroll_qdrant_texts(sink: str, collection: str) -> list[str]:
    """
    Page through every point in a Qdrant collection via scroll API and extract
    source_text from each point's payload.

    source_text is stored by embed() at upsert time — see embed.py _upsert_qdrant().
    Points embedded by other tools (qdrant-spark etc.) won't have source_text and
    are silently skipped; the n_source vs n_migrated mismatch surfaces this to the user.

    For 10M docs at batch=100: ~100K scroll calls, ~100 sec network I/O at 1ms/call.
    Batch size can be raised to 1000 if needed (Qdrant max: 100MB payload per response).
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError("pip install 'drift-spark[qdrant]' to use migrate()")

    u = urlparse(sink)
    client = QdrantClient(host=u.hostname or "localhost", port=u.port or 6333)

    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
    except ImportError:
        UnexpectedResponse = Exception  # fallback; scroll will still raise

    texts: list[str] = []
    offset = None
    while True:
        try:
            results, offset = client.scroll(
                collection_name=collection,
                limit=100,
                with_payload=True,    # need source_text from payload
                with_vectors=False,   # vectors not needed — saves bandwidth
                offset=offset,
            )
        except UnexpectedResponse as exc:
            if "404" in str(exc) or "Not found" in str(exc):
                # Collection doesn't exist yet — treat as empty
                return []
            raise
        for point in results:
            text = point.payload.get("source_text", "")
            if text:
                texts.append(text)
        if offset is None:
            break  # scroll exhausted
    return texts


# ── Spark session helper (same pattern as embed.py / watch.py) ────────────────

def _get_spark():
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        raise ImportError("pip install 'drift-spark[spark]' to use migrate()")

    spark = SparkSession.getActiveSession()
    if spark is None:
        for _path in (
            "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
            "/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
        ):
            if os.path.isdir(_path):
                os.environ.setdefault("JAVA_HOME", _path)
                break
        spark = (
            SparkSession.builder
            .appName("drift-migrate")
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")
    return spark


# ── public API ────────────────────────────────────────────────────────────────

def migrate(
    from_model: str,
    to_model: str,
    sink: str,
    *,
    strategy: str = "dual-write",
    shadow_mode: bool = False,
    ledger=None,
) -> MigrateRun:
    """
    Migrate embeddings from one model to another using the dual-write strategy.

    Dual-write (v0.3):
        1. Scroll all vectors from the original collection, extract source_text payloads.
        2. Re-embed all texts with to_model → write to <collection>_v2.
        3. Return MigrateRun with n_source and n_migrated for verification.
        4. User manually validates and flips app config to the new collection.

    WHY THERE IS NO AUTOMATIC CUTOVER:

        1. Recall regression is invisible without measurement.
           Cosine similarity scores are model-specific — 0.87 from ada-002 and
           0.87 from text-embedding-3-small are not comparable. Without running
           your real queries against the new collection, you don't know if recall
           improved or degraded for YOUR data.

        2. MTEB benchmarks measure average quality across generic datasets.
           Your domain (medical records, legal docs, code) may rank models
           differently from the benchmark average.
           Evidence: Drift-Adapter paper (EMNLP 2025) — GloVe → MPNet gets only
           71.5% ARR, showing architectural mismatch causes silent quality loss.

        3. Rollback after cutover is expensive.
           Re-embedding everything again with the old model (possibly deprecated)
           is the only recovery path. The manual gate is your last cheap chance to
           validate before committing.

    WHAT TO DO BEFORE CUTTING OVER:
        1. Run 20+ representative queries against sink_v2
        2. Compare top-10 results for queries you know the right answer to
        3. Run catch-up: drift watch --table <table> --sink <sink_v2> --model <to_model>
           (catches docs added during migration — see build guide 6-migrate-dual-write.md)
        4. If sink_v2 looks correct: update your app config to query sink_v2
        5. Monitor 24–48 hours, then delete the old collection

    v2 will add shadow-eval (auto route N% of traffic to both, measure recall@k delta)
    and Drift-Adapter (keep old index, train a projection — 95-99% recall at 1/2000th cost).

    Args:
        from_model:   current model, e.g. 'openai/text-embedding-ada-002'
        to_model:     target model, e.g. 'openai/text-embedding-3-small'
        sink:         URI of the existing collection, e.g. 'qdrant://localhost:6333/docs'
        strategy:     'dual-write' (v0.3). 'shadow-eval' and 'drift-adapter' in v2.
        shadow_mode:  use mock vectors — no API calls, no cost. Safe for testing.
        ledger:       Ledger instance; creates ~/.drift/ledger.db if None.

    Returns:
        MigrateRun — check n_migrated == n_source before cutting over.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}."
        )
    if strategy != "dual-write":
        raise NotImplementedError(
            f"strategy={strategy!r} is planned for v2. "
            "Only 'dual-write' is available in v0.3. "
            "See docs/competitors.md and build guide 6-migrate-dual-write.md "
            "for the v2 Drift-Adapter implementation plan."
        )

    u = urlparse(sink)
    if u.scheme != "qdrant":
        raise NotImplementedError(
            f"migrate() only supports qdrant:// sinks in v0.3. Got: {u.scheme!r}. "
            "pgvector migration coming in v0.4."
        )

    from .ledger import Ledger as _Ledger
    if ledger is None:
        ledger = _Ledger()

    # Derive new collection URI by appending _v2 to the collection name
    old_collection = u.path.strip("/")
    new_collection = f"{old_collection}_v2"
    sink_v2 = urlunparse(u._replace(path=f"/{new_collection}"))

    run = MigrateRun(
        from_model=from_model,
        to_model=to_model,
        strategy=strategy,
        sink=sink,
        sink_v2=sink_v2,
    )
    t0 = time.monotonic()

    # Phase 1: scroll all source_text values from the old collection
    texts = _scroll_qdrant_texts(sink, old_collection)
    run.n_source = len(texts)

    if not texts:
        run.duration_s = time.monotonic() - t0
        return run

    # Phase 2: build a DataFrame and re-embed with new model → write to sink_v2
    # We reuse embed() so batching, backoff, shadow_mode, and ledger writes
    # are all inherited for free. dedup=False: always re-embed everything on
    # migration regardless of prior ledger state.
    spark = _get_spark()
    df = spark.createDataFrame(
        [{"_migrate_idx": str(i), "_migrate_text": t} for i, t in enumerate(texts)]
    )

    from .embed import embed as _embed
    embed_run = _embed(
        df=df,
        text_col="_migrate_text",
        model=to_model,
        sink=sink_v2,
        dedup=False,
        shadow_mode=shadow_mode,
        ledger=ledger,
    )

    run.n_migrated = embed_run.n_rows_processed
    run.duration_s = time.monotonic() - t0
    return run
