"""Shadow evaluation: Adapted Recall Rate (ARR) gate for DriftAdapter quality."""

from __future__ import annotations

import numpy as np

from .adapter import DriftAdapter

ARR_THRESHOLD = 0.97


class AdapterQualityError(Exception):
    """
    Raised by measure_arr() when ARR falls below the quality threshold.

    ARR below 0.97 means the embedding spaces are too architecturally different
    for Orthogonal Procrustes to bridge reliably. The known failure case is
    GloVe → MPNet (71.5% ARR): token-level vs. full-sequence attention produce
    irreconcilable geometric spaces.

    When this is raised, fall back to migrate(strategy='dual-write') for a
    full reindex — the cost is higher but recall is guaranteed.
    """

    def __init__(self, arr: float, threshold: float) -> None:
        self.arr = arr
        self.threshold = threshold
        super().__init__(
            f"Adapter ARR {arr:.3f} is below threshold {threshold:.2f}. "
            "The embedding spaces are too architecturally different for the "
            "Procrustes adapter. "
            "Recommendation: use migrate(strategy='dual-write') for a full reindex."
        )


def measure_arr(
    adapter: DriftAdapter,
    old_vecs: np.ndarray,
    new_vecs: np.ndarray,
    *,
    k: int = 10,
    threshold: float | None = ARR_THRESHOLD,
) -> float:
    """
    Compute Adapted Recall Rate (ARR) — self-supervised adapter quality metric.

        ARR = mean_i( |oracle_top_k(i) ∩ adapted_top_k(i)| / k )

    Oracle path:  find top-k neighbors for old_vecs[i] in the old_vecs corpus.
    Adapted path: find top-k neighbors for adapter.predict(new_vecs[i]) in the
                  same corpus.

    Self-supervised: no human annotation needed. Quality is measured relative
    to what the old model would retrieve. ARR=0.97 means for every 100 documents
    the oracle returns, the adapter recovers 97 of them.

    Self-retrieval (query == doc at same index) is excluded from neighbor sets
    to avoid trivially inflating the score.

    Memory: builds two (N, N) similarity matrices in float32.
    For N > 5000, pass a held-out subset rather than the full corpus.

    Args:
        adapter:   fitted DriftAdapter
        old_vecs:  (N, d) — evaluation vectors from the OLD model (the corpus)
        new_vecs:  (N, d) — same texts from the NEW model, paired row-for-row
        k:         neighbors to retrieve per query (default 10)
        threshold: raise AdapterQualityError if ARR < threshold.
                   Pass None to always return the float without raising.

    Returns:
        float ARR in [0.0, 1.0]

    Raises:
        AdapterQualityError: if ARR < threshold (and threshold is not None)
        ValueError: if shapes mismatch or k >= N
    """
    old_vecs = np.asarray(old_vecs, dtype=np.float32)
    new_vecs = np.asarray(new_vecs, dtype=np.float32)

    if old_vecs.shape != new_vecs.shape:
        raise ValueError(
            f"old_vecs and new_vecs must have the same shape. "
            f"Got {old_vecs.shape} and {new_vecs.shape}."
        )

    N = old_vecs.shape[0]
    if k >= N:
        raise ValueError(
            f"k={k} must be less than the number of vectors N={N}."
        )

    adapted_vecs = adapter.predict(new_vecs)  # (N, d) — rotated into old model's space

    # Cosine similarity matrices — unit-norm embeddings: cos_sim = dot product
    oracle_sims = old_vecs @ old_vecs.T       # (N, N)
    adapted_sims = adapted_vecs @ old_vecs.T  # (N, N)

    recalls = np.empty(N, dtype=np.float32)
    for i in range(N):
        oracle_top = _top_k(oracle_sims[i], k, exclude=i)
        adapted_top = _top_k(adapted_sims[i], k, exclude=i)
        recalls[i] = len(oracle_top & adapted_top) / k

    arr = float(np.mean(recalls))

    if threshold is not None and arr < threshold:
        raise AdapterQualityError(arr=arr, threshold=threshold)

    return arr


def _top_k(sims: np.ndarray, k: int, exclude: int) -> set[int]:
    """Top-k indices by similarity, excluding the query's own index."""
    sims = sims.copy()
    sims[exclude] = -np.inf
    return set(np.argpartition(sims, -k)[-k:].tolist())
