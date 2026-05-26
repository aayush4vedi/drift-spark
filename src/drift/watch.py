"""Subsystem 2: incremental CDC refresh via Delta Change Data Feed."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class WatchRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    source_table: str = ""
    since_version: int = 0
    to_version: int = 0
    n_inserted: int = 0
    n_updated: int = 0
    n_deleted: int = 0


def watch(source_table: str, text_col: str, sink: str, *,
          model: str = "openai/text-embedding-3-small",
          since_version: int | None = None) -> WatchRun:
    """
    Incrementally refresh embeddings from a Delta table via CDF.

    Args:
        source_table:   Delta table path or catalog.schema.table
        text_col:       column to embed
        sink:           sink URI
        model:          embedding model string
        since_version:  Delta version to read from (None = last committed)
    """
    raise NotImplementedError("watch() — implementation in next iteration")