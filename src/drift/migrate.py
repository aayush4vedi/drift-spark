"""Subsystem 3: model-upgrade migration strategies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

STRATEGIES = ("dual-write", "shadow-eval", "drift-adapter")


@dataclass
class MigrateRun:
    """Returned by migrate() — the record of a model-upgrade run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    from_model: str = ""
    to_model: str = ""
    strategy: str = "dual-write"
    n_migrated: int = 0
    collection_old: str = ""
    collection_new: str = ""


def migrate(
    from_model: str,
    to_model: str,
    sink: str,
    *,
    strategy: str = "dual-write",
    ledger=None,
) -> MigrateRun:
    """
    Migrate embeddings from one model to another.

    Strategies:
        dual-write     — embed all docs with to_model into <collection>_v2;
                         user flips app config when ready. Safe, manual cutover. (v1.0)
        shadow-eval    — dual-write + live shadow query stream with recall@k report. (v2.0)
        drift-adapter  — train a projection matrix old→new space; 95-99% recall
                         at 1/100th the reindex cost (Drift-Adapter paper). (v2.0)

    Args:
        from_model: current model string, e.g. 'openai/text-embedding-ada-002'
        to_model:   target model string, e.g. 'openai/text-embedding-3-small'
        sink:       sink URI pointing to the existing collection
        strategy:   one of STRATEGIES (default 'dual-write')
        ledger:     Ledger instance

    Returns:
        MigrateRun with migration metadata.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose from {STRATEGIES}"
        )
    if strategy != "dual-write":
        raise NotImplementedError(
            f"strategy={strategy!r} is coming in v2.0. "
            "Use strategy='dual-write' for v1.0. "
            "See https://github.com/aayush4vedi/drift/blob/main/docs/migration.md"
        )

    run = MigrateRun(from_model=from_model, to_model=to_model, strategy=strategy)

    # --- implementation lands Wed 6/3 ---
    # dual-write steps:
    #   1. Determine collection_old from sink URI
    #   2. collection_new = collection_old + "_v2"
    #   3. Call embed(df, model=to_model, sink=<new collection URI>)
    #   4. Verify len(collection_old) == len(collection_new)
    #   5. Return MigrateRun — user manually flips app config to _v2
    raise NotImplementedError(
        "migrate() dual-write — full implementation in Wed 6/3 build session. "
        "Scaffold confirmed wired correctly."
    )

    return run
