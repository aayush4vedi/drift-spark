"""Tests for shadow_eval.py — measure_arr() and AdapterQualityError."""

import numpy as np
import pytest

from drift.adapter import DriftAdapter
from drift.shadow_eval import AdapterQualityError, measure_arr


# ── helpers ───────────────────────────────────────────────────────────────────

def _unit_norm(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _rotation_matrix(d: int, theta_deg: float) -> np.ndarray:
    theta = np.radians(theta_deg)
    R = np.eye(d, dtype=np.float32)
    R[0, 0] = np.cos(theta)
    R[0, 1] = np.sin(theta)
    R[1, 0] = -np.sin(theta)
    R[1, 1] = np.cos(theta)
    return R


def _bad_adapter(d: int, seed: int = 99) -> DriftAdapter:
    """An adapter with a random (untrained) orthogonal R — scrambles neighbor structure."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    adapter = DriftAdapter.__new__(DriftAdapter)
    adapter.R = Q.astype(np.float32)
    return adapter


# ── test_arr_perfect_adapter ─────────────────────────────────────────────────

def test_arr_perfect_adapter():
    """
    A mathematically perfect rotation: adapter.predict(X_new) == X_old exactly.
    Oracle top-k and adapted top-k are identical for every query → ARR = 1.0.
    """
    d = 8
    N = 50
    rng = np.random.default_rng(0)
    R_true = _rotation_matrix(d, 30.0)

    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = _unit_norm(raw)
    X_new = X_old @ R_true  # new = old rotated by R_true

    adapter = DriftAdapter().fit(X_old, X_new)
    arr = measure_arr(adapter, X_old, X_new, k=5, threshold=None)

    assert arr == 1.0, f"Perfect adapter should give ARR=1.0, got {arr:.4f}"


# ── test_arr_identity_case ────────────────────────────────────────────────────

def test_arr_identity_case():
    """
    When old_vecs == new_vecs (same model), R = I and predict is a no-op.
    Oracle and adapted retrieve the same neighbors → ARR = 1.0.
    """
    d = 8
    N = 50
    rng = np.random.default_rng(1)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X = _unit_norm(raw)

    adapter = DriftAdapter().fit(X, X)
    arr = measure_arr(adapter, X, X, k=5, threshold=None)

    assert arr == 1.0, f"Identity case should give ARR=1.0, got {arr:.4f}"


# ── test_arr_random_rotation_is_low ──────────────────────────────────────────

def test_arr_random_rotation_is_low():
    """
    A random (untrained) orthogonal R scrambles neighbor structure.
    Expected ARR ≈ k/(N-1) by random chance; with k=5, N=100 → ≈0.05.
    Must be well below the 0.97 threshold.
    """
    d = 16
    N = 100
    rng = np.random.default_rng(2)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = _unit_norm(raw)

    bad_adapter = _bad_adapter(d, seed=99)
    arr = measure_arr(bad_adapter, X_old, X_old, k=5, threshold=None)

    assert arr < 0.5, (
        f"Random rotation should give ARR << 0.97. Got {arr:.4f}. "
        f"(Expected ≈ k/(N-1) ≈ {5/99:.3f} by chance.)"
    )


# ── test_measure_arr_raises_quality_error ─────────────────────────────────────

def test_measure_arr_raises_quality_error():
    """ARR below threshold raises AdapterQualityError with arr and threshold set."""
    d = 8
    N = 50
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = _unit_norm(raw)

    bad_adapter = _bad_adapter(d, seed=7)

    with pytest.raises(AdapterQualityError) as exc_info:
        measure_arr(bad_adapter, X_old, X_old, k=5, threshold=0.97)

    err = exc_info.value
    assert err.arr < 0.97
    assert err.threshold == 0.97
    assert "dual-write" in str(err)  # error message names the fallback strategy


# ── test_measure_arr_threshold_none ──────────────────────────────────────────

def test_measure_arr_threshold_none_never_raises():
    """threshold=None returns the float regardless of how low ARR is."""
    d = 8
    N = 50
    rng = np.random.default_rng(4)
    raw = rng.standard_normal((N, d)).astype(np.float32)
    X_old = _unit_norm(raw)

    bad_adapter = _bad_adapter(d, seed=8)
    arr = measure_arr(bad_adapter, X_old, X_old, k=5, threshold=None)

    assert isinstance(arr, float)
    assert 0.0 <= arr <= 1.0


# ── test_arr_k_validation ─────────────────────────────────────────────────────

def test_arr_k_too_large_raises():
    """k >= N raises ValueError — can't retrieve k neighbors from N vectors."""
    N, d = 20, 4
    rng = np.random.default_rng(5)
    X = _unit_norm(rng.standard_normal((N, d)).astype(np.float32))
    adapter = DriftAdapter().fit(X, X)

    with pytest.raises(ValueError, match="k="):
        measure_arr(adapter, X, X, k=20, threshold=None)  # k == N

    with pytest.raises(ValueError, match="k="):
        measure_arr(adapter, X, X, k=25, threshold=None)  # k > N


# ── test_arr_shape_mismatch ───────────────────────────────────────────────────

def test_arr_shape_mismatch_raises():
    """old_vecs and new_vecs with different shapes raise ValueError."""
    N, d = 30, 8
    rng = np.random.default_rng(6)
    X_old = _unit_norm(rng.standard_normal((N, d)).astype(np.float32))
    X_new_bad = _unit_norm(rng.standard_normal((N, d + 1)).astype(np.float32))
    adapter = DriftAdapter().fit(X_old, _unit_norm(rng.standard_normal((N, d)).astype(np.float32)))

    with pytest.raises(ValueError, match="same shape"):
        measure_arr(adapter, X_old, X_new_bad, k=5, threshold=None)
