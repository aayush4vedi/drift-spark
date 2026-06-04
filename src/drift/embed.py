"""Subsystem 1: batch embedding with dedup, multi-model, multi-sink."""

from __future__ import annotations

import hashlib
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterator
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from ._utils import _get_spark


# ── cost table (USD per token) ───────────────────────────────────────────────
# NOTE: prices hardcoded as of June 2025 (OpenAI). Verify at
# platform.openai.com/pricing before using for budget decisions.
# Configurable override is planned for v1.0.
_COST_PER_TOKEN: dict[str, float] = {
    "text-embedding-3-small": 0.02 / 1_000_000,
    "text-embedding-3-large": 0.13 / 1_000_000,
    "text-embedding-ada-002": 0.10 / 1_000_000,
}

# shadow mode produces 1536-dim vectors — matches text-embedding-3-small
_SHADOW_DIM = 1536


@dataclass
class EmbedRun:
    """Returned by embed() — the record of what happened in this run."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model: str = ""
    sink: str = ""
    n_rows_processed: int = 0
    n_rows_deduped: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0


# ── internal helpers ─────────────────────────────────────────────────────────

def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _parse_model(model: str) -> tuple[str, str]:
    """'openai/text-embedding-3-small' → ('openai', 'text-embedding-3-small')"""
    if "/" not in model:
        raise ValueError(
            f"model must be 'provider/name', e.g. 'openai/text-embedding-3-small'. Got: {model!r}"
        )
    provider, name = model.split("/", 1)
    return provider, name


def _mock_embedding(text: str, dim: int = _SHADOW_DIM) -> list[float]:
    """
    Deterministic unit vector derived from text — no API, zero cost.

    Same text always produces the same vector, so dedup and provenance work
    correctly in shadow mode. Useful for CI, local dev, and cost-free demos.
    The vector is on the unit hypersphere (cosine similarity is well-defined).
    """
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _chunked(lst: list, size: int) -> Iterator[list]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


# ── embedding backends ───────────────────────────────────────────────────────

def _embed_openai(
    texts: list[str], model_name: str, batch_size: int
) -> tuple[list[list[float]], float]:
    """Call OpenAI Embeddings API in batches with exponential backoff on 429."""
    try:
        from openai import OpenAI, RateLimitError
    except ImportError:
        raise ImportError("pip install openai to use OpenAI models")

    import os
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Set it in your environment, or pass shadow_mode=True for cost-free testing."
        )

    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []
    total_tokens = 0

    for batch in _chunked(texts, batch_size):
        for attempt in range(5):
            try:
                resp = client.embeddings.create(model=model_name, input=batch)
                vectors.extend(e.embedding for e in resp.data)
                total_tokens += resp.usage.total_tokens
                break
            except RateLimitError:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)

    cost = total_tokens * _COST_PER_TOKEN.get(model_name, 0.02 / 1_000_000)
    return vectors, cost


# ── sink writers ─────────────────────────────────────────────────────────────

def _upsert_qdrant(sink: str, points: list[dict]) -> None:
    """
    Upsert to Qdrant. points = [{id, vector, payload}].
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
    except ImportError:
        raise ImportError("pip install 'drift-spark[qdrant]' to use the Qdrant sink")

    u = urlparse(sink)
    collection = u.path.strip("/")
    if not collection:
        raise ValueError(f"Qdrant sink URI must include a collection name: {sink!r}")

    dim = len(points[0]["vector"])
    host = u.hostname or "localhost"
    port = u.port or 6333
    client = QdrantClient(host=host, port=port)

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ],
    )


def _upsert_pgvector(sink: str, points: list[dict]) -> None:
    """
    Upsert to pgvector. Stub — basic INSERT, no CDC yet.

    URI format: pg://user:pass@host:5432/dbname?table=collection_name
    The table= query param names the target table (default: 'embeddings').
    """
    try:
        import psycopg2
        from psycopg2.extras import execute_values
    except ImportError:
        raise ImportError("pip install 'drift-spark[pgvector]' to use the pgvector sink")

    u = urlparse(sink)
    params = parse_qs(u.query)
    table = params.get("table", ["embeddings"])[0]
    dim = len(points[0]["vector"])

    # Strip the custom table= param before building the psycopg2 DSN
    clean_query = urlencode({k: v[0] for k, v in params.items() if k != "table"})
    dsn = urlunparse(u._replace(scheme="postgresql", query=clean_query))

    conn = psycopg2.connect(dsn)
    with conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id           TEXT PRIMARY KEY,
                    embedding    vector({dim}),
                    source_hash  TEXT,
                    embed_run_id TEXT,
                    created_at   TEXT
                )
            """)
            execute_values(
                cur,
                f"INSERT INTO {table} (id, embedding, source_hash, embed_run_id, created_at)"
                " VALUES %s ON CONFLICT DO NOTHING",
                [
                    (
                        p["id"],
                        str(p["vector"]),
                        p["payload"].get("source_hash", ""),
                        p["payload"].get("embed_run_id", ""),
                        p["payload"].get("timestamp", ""),
                    )
                    for p in points
                ],
            )
    conn.close()


# ── public API ───────────────────────────────────────────────────────────────

def embed(
    df,
    text_col: str,
    model: str,
    sink: str,
    *,
    source_table: str | None = None,
    dedup: bool = True,
    batch_size: int = 128,
    shadow_mode: bool = False,
    ledger=None,
) -> EmbedRun:
    """
    Embed a Spark DataFrame column and upsert vectors to a sink.

    Args:
        df:           PySpark DataFrame (or None when source_table is given)
        text_col:     column name containing the text to embed
        model:        'provider/model-name', e.g. 'openai/text-embedding-3-small'
        sink:         sink URI — 'qdrant://host:port/collection' or 'pg://...'
        source_table: Delta/Iceberg table to load when df=None (CLI path)
        dedup:        skip rows whose text hash was already embedded with this
                      (model, sink) pair in a prior run — the core cost-saving feature
        batch_size:   texts per API call (OpenAI max: 2048)
        shadow_mode:  use deterministic mock embeddings — no API calls, zero cost.
                      Identical texts produce identical vectors, so dedup and
                      provenance both work correctly. Safe for CI and local dev.
        ledger:       Ledger instance; creates ~/.drift/ledger.db if None

    Returns:
        EmbedRun — run_id, row counts, cost_usd, duration_s.
    """
    provider, model_name = _parse_model(model)

    from .ledger import Ledger as _Ledger
    if ledger is None:
        ledger = _Ledger()

    run = EmbedRun(model=model, sink=sink)
    t0 = time.monotonic()

    # 1. Load DataFrame -------------------------------------------------------
    if df is None:
        if source_table is None:
            raise ValueError("Provide either df or source_table.")
        spark = _get_spark("drift-cli")
        df = spark.table(source_table)

    # 2. Collect texts to driver ----------------------------------------------
    # Driver-side collect; works up to ~10M rows on a cluster driver.
    # A distributed path via broadcast of known hashes is planned.
    pdf = df.select(text_col).toPandas()
    all_texts: list[str] = pdf[text_col].tolist()
    run.n_rows_processed = len(all_texts)

    if not all_texts:
        run.duration_s = time.monotonic() - t0
        ledger.write_run(run)
        return run

    # 3. Dedup — filter to texts not yet embedded with this (model, sink) -----
    all_hashes = [_text_hash(t) for t in all_texts]

    if dedup:
        new_indices = [
            i for i, h in enumerate(all_hashes)
            if not ledger.hash_exists(h, model, sink)
        ]
    else:
        new_indices = list(range(len(all_texts)))

    run.n_rows_deduped = len(all_texts) - len(new_indices)

    new_texts = [all_texts[i] for i in new_indices]
    new_hashes = [all_hashes[i] for i in new_indices]

    if not new_texts:
        run.duration_s = time.monotonic() - t0
        ledger.write_run(run)
        return run

    # 4. Embed ----------------------------------------------------------------
    if shadow_mode:
        vectors: list[list[float]] = [_mock_embedding(t) for t in new_texts]
        cost = 0.0
    elif provider == "openai":
        vectors, cost = _embed_openai(new_texts, model_name, batch_size)
    else:
        raise ValueError(
            f"Unsupported provider: {provider!r}. "
            "Supported: 'openai'. Use shadow_mode=True for cost-free local testing."
        )

    run.cost_usd = cost

    # 5. Upsert to sink -------------------------------------------------------
    now = datetime.now(timezone.utc).isoformat()
    # ID = deterministic UUID from text hash → idempotent upserts on retry
    points = [
        {
            "id": str(uuid.uuid5(uuid.NAMESPACE_OID, h)),
            "vector": vec,
            "payload": {
                "source_text": text,
                "source_hash": h,
                "embed_run_id": run.run_id,
                "timestamp": now,
            },
        }
        for text, h, vec in zip(new_texts, new_hashes, vectors)
    ]

    u = urlparse(sink)
    if u.scheme == "qdrant":
        _upsert_qdrant(sink, points)
    elif u.scheme in ("pg", "postgresql"):
        _upsert_pgvector(sink, points)
    else:
        raise ValueError(
            f"Unsupported sink scheme: {u.scheme!r}. Use 'qdrant://' or 'pg://'."
        )

    # 6. Write provenance -----------------------------------------------------
    for p in points:
        ledger.write_provenance(
            embedding_id=p["id"],
            source_pk=p["id"],
            source_hash=p["payload"]["source_hash"],
            run_id=run.run_id,
            created_at=now,
        )

    run.duration_s = time.monotonic() - t0
    ledger.write_run(run)
    return run
