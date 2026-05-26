"""Subsystem 3: model-upgrade migration strategies."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import uuid

STRATEGIES = ("dual-write", "shadow-eval", "drift-adapter")


@dataclass
class MigrateRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    from_model: str = ""
    to_model: str = ""
    strategy: str = "dual-write"
    n_migrated: int = 0
    collection_old: str = ""
    collection_new: str = ""


def migrate(from_model: str, to_model: str, sink: str, *,
            strategy: str = "dual-write") -> MigrateRun:
    """
    Migrate embeddings from one model to another.

    Strategies:
        dual-write    — embed into new collection, user flips config (v1.0)
        shadow-eval   — dual-write + shadow query stream with recall@k report (v2.0)
        drift-adapter — 95-99% recall via projection matrix, 100x cheaper (v2.0)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}")
    if strategy != "dual-write":
        raise NotImplementedError(
            f"strategy={strategy!r} coming in v2.0. "
            "Use strategy='dual-write' for v1.0. See docs/migration.md."
        )
    raise NotImplementedError("migrate() — implementation in next iteration")