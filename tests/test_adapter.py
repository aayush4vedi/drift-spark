"""Tests for DriftAdapter — Orthogonal Procrustes, Phase 1."""

import numpy as np
import pytest

from drift.adapter import DriftAdapter

# ── helpers ───────────────────────────────────────────────────────────────────

def _rotation_matrix(d: int, theta_deg: float = 30.0) -> np.ndarray:
    """Givens rotation in the first two dimensions of a d-dimensional space."""
    theta = np.radians(theta_deg)
    R = np.eye(d, dtype=np.float32)
    R[0, 0] = np.cos(theta)
    R[0, 1] = np.sin(theta)
    R[1, 0] = -np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R


def _orthonormal_rows(N: int, d: int, seed: int = 0) -> np.ndarray:
    """
    Returns (N, d) matrix with orthonormal *columns* (X.T @ X = I).
    Constructed via QR decomposition so X_old.T @ X_new = R_true exactly,
    making the Procrustes solution exact rather than approximate.
    """
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((N, d)))
    return Q.astype(np.float32)


# ── test_fit_recovers_known_rotation ─────────────────────────────────────────

def test_fit_recovers_known_rotation():
    """
    With X_old having orthonormal columns:
        M = X_old.T @ X_new = X_old.T @ (X_old @ R_true) = I @ R_true = R_true
    So SVD(M) yields R_fitted = R_true exactly (up to float precision).
    """
    d = 4
    N = 32                           # N > d — needed for QR orthonormal columns
    R_true = _rotation_matrix(d, 30.0)
    X_old = _orthonormal_rows(N, d)  # X_old.T @ X_old = I
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)

    assert np.allclose(adapter.R, R_true, atol=1e-4), (
        f"R_fitted should recover R_true.\nDiff: {np.abs(adapter.R - R_true).max():.6f}"
    )


# ── test_predict_roundtrip ────────────────────────────────────────────────────

def test_predict_roundtrip():
    """
    After fitting on (X_old, X_new = X_old @ R_true), predict(X_new) should
    recover X_old: each row's cosine similarity to the original must exceed 0.99.
    """
    d = 8
    N = 100
    rng = np.random.default_rng(1)
    R_true = _rotation_matrix(d, 45.0)

    # Random unit-norm X_old (mimics real normalized embeddings)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    adapted = adapter.predict(X_new)

    # cosine similarity = dot product for unit vectors
    # after fit, adapted rows should be very close to X_old rows
    cos_sims = np.sum(adapted * X_old, axis=1) / (
        np.linalg.norm(adapted, axis=1) * np.linalg.norm(X_old, axis=1)
    )
    assert cos_sims.min() > 0.99, (
        f"All rows should have cosine similarity > 0.99. Min: {cos_sims.min():.4f}"
    )


# ── test_save_load_idempotency ────────────────────────────────────────────────

def test_save_load_idempotency(tmp_path):
    """save() → load() preserves R exactly; predict() gives identical output."""
    d = 4
    N = 32
    R_true = _rotation_matrix(d, 20.0)
    X_old = _orthonormal_rows(N, d)
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    npy_path = str(tmp_path / "R.npy")
    adapter.save(npy_path)

    loaded = DriftAdapter.load(npy_path)

    assert np.allclose(loaded.R, adapter.R), "Loaded R must match original R exactly."

    # predict output must be bit-identical
    q = X_new[:3]
    assert np.array_equal(adapter.predict(q), loaded.predict(q))

    # file should be ~d*d*4 bytes (float32)
    import os
    file_size = os.path.getsize(npy_path)
    expected_min = d * d * 4  # at least d² float32 values
    assert file_size >= expected_min


# ── test_shape_assertions ─────────────────────────────────────────────────────

def test_shape_assertions_mismatched_dim():
    """fit() with mismatched d raises ValueError."""
    old = np.ones((50, 4), dtype=np.float32)
    new = np.ones((50, 5), dtype=np.float32)   # wrong d
    with pytest.raises(ValueError, match="same shape"):
        DriftAdapter().fit(old, new)


def test_shape_assertions_1d_predict():
    """predict() with 1D input (d,) returns 1D output (d,)."""
    d = 4
    N = 32
    R_true = _rotation_matrix(d, 10.0)
    X_old = _orthonormal_rows(N, d)
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    q = X_new[0]                     # shape (d,)
    out = adapter.predict(q)
    assert out.shape == (d,), f"Expected shape ({d},), got {out.shape}"


def test_shape_assertions_2d_predict():
    """predict() with (N, d) input returns (N, d) output."""
    d = 4
    N = 32
    R_true = _rotation_matrix(d, 10.0)
    X_old = _orthonormal_rows(N, d)
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    out = adapter.predict(X_new)
    assert out.shape == (N, d), f"Expected shape ({N}, {d}), got {out.shape}"


# ── test_orthogonality_of_R ───────────────────────────────────────────────────

def test_orthogonality_of_R():
    """R @ R.T must be close to the identity matrix (float32 tolerance)."""
    d = 16
    N = 64
    rng = np.random.default_rng(7)
    R_true = _rotation_matrix(d, 15.0)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    X_new = X_old @ R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    RRt = adapter.R @ adapter.R.T

    assert np.allclose(RRt, np.eye(d, dtype=np.float32), atol=1e-4), (
        f"R @ R.T should be identity. Max deviation: {np.abs(RRt - np.eye(d)).max():.6f}"
    )


# ── test_fit_identity_case ────────────────────────────────────────────────────

def test_fit_identity_case():
    """
    When old_vecs == new_vecs (same model, same vectors), R should be ≈ identity.
    No transformation needed — the adapter is a no-op.
    """
    d = 8
    N = 50
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X = raw / np.linalg.norm(raw, axis=1, keepdims=True)

    adapter = DriftAdapter().fit(X, X)

    assert np.allclose(adapter.R, np.eye(d, dtype=np.float32), atol=1e-4), (
        f"R should be identity when old_vecs == new_vecs. "
        f"Max deviation: {np.abs(adapter.R - np.eye(d)).max():.6f}"
    )
