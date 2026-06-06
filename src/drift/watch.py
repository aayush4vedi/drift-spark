"""Subsystem 2: incremental CDC refresh via Delta Change Data Feed."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

from ._utils import _get_spark
from .embed import _text_hash, embed


@dataclass
class WatchRun:
    """Returned by watch() — the record of what changed in this refresh cycle."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_table: str = ""
    sink: str = ""
    since_version: int = 0
    to_version: int = 0
    n_inserted: int = 0
    n_updated: int = 0
    n_deleted: int = 0
    duration_s: float = 0.0


# ── sink delete helpers ───────────────────────────────────────────────────────

def _delete_qdrant(sink: str, texts: list[str]) -> int:
    """Delete vectors for the given texts from a Qdrant collection."""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointIdsList
    except ImportError as err:
        raise ImportError("pip install 'drift-spark[qdrant]' to use the Qdrant sink") from err

    u = urlparse(sink)
    collection = u.path.strip("/")
    point_ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, _text_hash(t))) for t in texts]
    client = QdrantClient(host=u.hostname or "localhost", port=u.port or 6333)
    client.delete(
        collection_name=collection,
        points_selector=PointIdsList(points=point_ids),  # type: ignore[arg-type]
    )
    return len(point_ids)


def _delete_from_sink(sink: str, texts: list[str]) -> int:
    """Dispatch delete to the correct sink backend."""
    u = urlparse(sink)
    if u.scheme == "qdrant":
        return _delete_qdrant(sink, texts)
    elif u.scheme in ("pg", "postgresql"):
        raise NotImplementedError(
            "pgvector CDC delete is not yet supported (planned for v0.2). "
            "Use Qdrant sink for CDC workflows."
        )
    else:
        raise ValueError(f"Unsupported sink scheme: {u.scheme!r}. Use 'qdrant://'.")


# ── public API ────────────────────────────────────────────────────────────────

def watch(
    source_table: str,
    text_col: str,
    sink: str,
    *,
    model: str = "openai/text-embedding-3-small",
    since_version: int | None = None,
    shadow_mode: bool = False,
    ledger=None,
) -> WatchRun:
    """
    Incrementally refresh embeddings from a Delta table via Change Data Feed.

    Reads only rows that changed since `since_version` (or the last committed
    version in the ledger for this source_table+sink pair). Embeds inserts and
    updates; deletes vectors for deleted source rows.

    Args:
        source_table:   Delta table — catalog.schema.table or path
        text_col:       column to embed
        sink:           sink URI, e.g. 'qdrant://localhost:6333/my_collection'
        model:          embedding model string
        since_version:  Delta version to start from. None = use last ledger
                        checkpoint, or 0 (full history) if first run.
        shadow_mode:    mock embeddings — no API calls, no cost
        ledger:         Ledger instance (creates ~/.drift/ledger.db if None)

    Returns:
        WatchRun with insert/update/delete counts and to_version.
        Pass run.to_version as since_version on the next call.
    """
    from .ledger import Ledger as _Ledger
    if ledger is None:
        ledger = _Ledger()

    # Resolve since_version — checkpoint lookup if not provided
    if since_version is None:
        since_version = ledger.last_watch_version(source_table, sink) or 0

    run = WatchRun(
        source_table=source_table,
        sink=sink,
        since_version=since_version,
    )
    t0 = time.monotonic()

    spark = _get_spark("drift-watch")

    # Read Delta CDF from since_version onward
    cdf = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", since_version)
        .table(source_table)
    )

    # Capture to_version from the CDF metadata column.
    # selectExpr avoids importing pyspark.sql.functions (lighter + more mockable).
    version_row = cdf.selectExpr("max(_commit_version)").collect()[0][0]
    run.to_version = int(version_row) if version_row is not None else since_version

    # Split by change type
    # Delta CDF change types: 'insert', 'update_preimage', 'update_postimage', 'delete'
    # update_preimage = row before the update (we skip it — we embed postimage)
    inserts_updates = cdf.filter(
        "_change_type IN ('insert', 'update_postimage')"
    )
    deletes = cdf.filter("_change_type = 'delete'")

    # Embed inserts + updates (dedup handles unchanged rows)
    n_new = inserts_updates.count()
    if n_new > 0:
        embed(
            df=inserts_updates,
            text_col=text_col,
            model=model,
            sink=sink,
            dedup=True,
            shadow_mode=shadow_mode,
            ledger=ledger,
        )
        run.n_inserted = inserts_updates.filter("_change_type = 'insert'").count()
        run.n_updated = inserts_updates.filter(
            "_change_type = 'update_postimage'"
        ).count()

    # Delete vectors for deleted source rows
    # Point ID = uuid5(NAMESPACE_OID, text_hash) — same formula as embed()
    # Delta CDF 'delete' rows include the full deleted row, so we have the text.
    n_del = deletes.count()
    if n_del > 0:
        deleted_texts = (
            deletes.select(text_col).toPandas()[text_col].tolist()
        )
        run.n_deleted = _delete_from_sink(sink, deleted_texts)

    run.duration_s = time.monotonic() - t0
    ledger.write_watch_run(run)
    return run
