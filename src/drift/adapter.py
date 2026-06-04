"""DriftAdapter: Orthogonal Procrustes rotation for embedding space alignment."""

from __future__ import annotations

import numpy as np


class DriftAdapter:
    """
    Orthogonal Procrustes adapter for near-zero-downtime embedding model upgrades.

    Learns a rotation matrix R from paired (old-model, new-model) embeddings.
    At query time: adapted = new_vec @ R.T

    Based on: Drift-Adapter (EMNLP 2025, arXiv 2509.23471), Phase 1.
    """

    R: np.ndarray  # shape (d, d), orthogonal: R @ R.T ≈ I

    def fit(self, old_vecs: np.ndarray, new_vecs: np.ndarray) -> DriftAdapter:
        """
        Solve Orthogonal Procrustes: find R = argmin ||X_old - X_new @ R.T||_F
        subject to R orthogonal.

        Closed-form: M = X_old.T @ X_new; U, _, Vt = svd(M); R = U @ Vt

        Args:
            old_vecs: (N, d) float32 — embeddings from the OLD model
            new_vecs: (N, d) float32 — embeddings from the NEW model
                      Row i of old_vecs and new_vecs must encode the same text.
        """
        old_vecs = np.asarray(old_vecs, dtype=np.float64)
        new_vecs = np.asarray(new_vecs, dtype=np.float64)

        if old_vecs.ndim != 2 or new_vecs.ndim != 2:
            raise ValueError(
                "old_vecs and new_vecs must be 2-D arrays of shape (N, d). "
                f"Got shapes {old_vecs.shape} and {new_vecs.shape}."
            )
        if old_vecs.shape != new_vecs.shape:
            raise ValueError(
                f"old_vecs and new_vecs must have the same shape. "
                f"Got {old_vecs.shape} and {new_vecs.shape}."
            )

        M = old_vecs.T @ new_vecs          # cross-covariance: (d, d)
        U, _, Vt = np.linalg.svd(M)        # full SVD; singular values discarded
        self.R = (U @ Vt).astype(np.float32)  # rotation matrix: (d, d)
        return self

    def predict(self, new_vecs: np.ndarray) -> np.ndarray:
        """
        Apply the rotation: adapted = new_vecs @ R.T
        Output lives in the old model's geometric space — send to Qdrant as-is.

        Args:
            new_vecs: (N, d) or (d,) — vectors from the new model
        Returns:
            (N, d) or (d,) — rotated vectors in old model's space
        """
        new_vecs = np.asarray(new_vecs, dtype=np.float32)
        return new_vecs @ self.R.T

    def save(self, path: str) -> None:
        """Serialise R as a .npy file. Path should end in .npy."""
        np.save(path, self.R)

    @classmethod
    def load(cls, path: str) -> DriftAdapter:
        """Deserialise from a .npy file."""
        adapter = cls.__new__(cls)
        adapter.R = np.load(path)
        return adapter

    @classmethod
    def _sample_paired_texts(
        cls,
        sink: str,
        n_pairs: int,
        from_model: str,
        to_model: str,
        shadow_mode: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Embeds n_pairs documents from the Qdrant collection with both models.
        Returns (old_vecs, new_vecs) — paired, same order.

        Used by migrate(strategy="drift-adapter") to build training data.
        Not called during normal adapter.fit() — that takes pre-built arrays.

        In shadow_mode both models use the same deterministic mock vectors,
        so old_vecs == new_vecs → adapter.R = I → ARR = 1.0. This is correct
        for shadow_mode: you're testing the pipeline plumbing, not the math.
        """
        import random
        from urllib.parse import urlparse

        from .embed import _embed_openai, _mock_embedding, _parse_model
        from .migrate import _scroll_qdrant_texts

        u = urlparse(sink)
        collection = u.path.strip("/")
        all_texts = _scroll_qdrant_texts(sink, collection)

        if not all_texts:
            raise ValueError(
                f"No source_text payloads found in collection {collection!r}. "
                "Populate the collection with drift embed() before running "
                "migrate(strategy='drift-adapter')."
            )

        texts = (
            random.sample(all_texts, n_pairs)
            if len(all_texts) > n_pairs
            else all_texts
        )

        if shadow_mode:
            old_vecs = np.array([_mock_embedding(t) for t in texts], dtype=np.float32)
            new_vecs = np.array([_mock_embedding(t) for t in texts], dtype=np.float32)
        else:
            _, old_name = _parse_model(from_model)
            _, new_name = _parse_model(to_model)
            old_raw, _ = _embed_openai(texts, old_name, batch_size=128)
            new_raw, _ = _embed_openai(texts, new_name, batch_size=128)
            old_vecs = np.array(old_raw, dtype=np.float32)
            new_vecs = np.array(new_raw, dtype=np.float32)

        return old_vecs, new_vecs
