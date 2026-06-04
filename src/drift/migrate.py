"""Subsystem 3: model-upgrade migration — dual-write strategy."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse, urlunparse

from ._utils import _get_spark

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
    sink_v2: str = ""    # new collection URI (<collection>_v2) — dual-write only
    n_source: int = 0    # vectors scrolled / pairs sampled
    n_migrated: int = 0  # vectors written to new collection (dual-write only)
    duration_s: float = 0.0
    adapter_path: str = ""  # path to saved .npy file (drift-adapter only)
    arr: float = 0.0        # ARR score from measure_arr() (drift-adapter only)


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
    except ImportError as err:
        raise ImportError("pip install 'drift-spark[qdrant]' to use migrate()") from err

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

    Dual-write:
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
        strategy:     'dual-write' or 'drift-adapter'. 'shadow-eval' is planned.
        shadow_mode:  use mock vectors — no API calls, no cost. Safe for testing.
        ledger:       Ledger instance; creates ~/.drift/ledger.db if None.

    Returns:
        MigrateRun — check n_migrated == n_source before cutting over.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}."
        )
    if strategy == "shadow-eval":
        raise NotImplementedError(
            "strategy='shadow-eval' is planned for v2. "
            "Use 'dual-write' or 'drift-adapter'."
        )

    u = urlparse(sink)
    if u.scheme != "qdrant":
        raise NotImplementedError(
            f"migrate() currently supports only qdrant:// sinks. Got: {u.scheme!r}. "
            "pgvector migration is planned."
        )

    from .ledger import Ledger as _Ledger
    if ledger is None:
        ledger = _Ledger()

    run = MigrateRun(
        from_model=from_model,
        to_model=to_model,
        strategy=strategy,
        sink=sink,
    )
    t0 = time.monotonic()
    old_collection = u.path.strip("/")

    # ── drift-adapter strategy ────────────────────────────────────────────────
    if strategy == "drift-adapter":
        from .adapter import DriftAdapter
        from .shadow_eval import measure_arr

        N_PAIRS = 5000
        all_old, all_new = DriftAdapter._sample_paired_texts(
            sink=sink,
            n_pairs=N_PAIRS,
            from_model=from_model,
            to_model=to_model,
            shadow_mode=shadow_mode,
        )

        N = len(all_old)
        run.n_source = N

        if N < 20:
            raise ValueError(
                f"Only {N} texts found in collection — need at least 20 for a "
                "meaningful 90/10 train/val split. Add more documents first."
            )

        split = int(N * 0.9)
        train_old, val_old = all_old[:split], all_old[split:]
        train_new, val_new = all_new[:split], all_new[split:]

        adapter = DriftAdapter().fit(train_old, train_new)

        k = min(10, len(val_old) - 1)
        arr = measure_arr(adapter, val_old, val_new, k=k)  # raises AdapterQualityError if < 0.97

        adapter_path = f"drift_adapter_{run.run_id[:8]}.npy"
        adapter.save(adapter_path)

        run.adapter_path = adapter_path
        run.arr = arr
        run.duration_s = time.monotonic() - t0
        return run

    # ── dual-write strategy ───────────────────────────────────────────────────
    new_collection = f"{old_collection}_v2"
    sink_v2 = urlunparse(u._replace(path=f"/{new_collection}"))
    run.sink_v2 = sink_v2

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
    spark = _get_spark("drift-migrate")
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
