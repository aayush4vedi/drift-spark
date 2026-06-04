# Drift-Adapter: Orthogonal Procrustes for Model Migration

How `migrate(strategy="drift-adapter")` upgrades an embedding model without re-indexing the vector store — the intuition, the closed-form solution, the quality gate, and what is intentionally out of scope.

<sub>Method based on <em>Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases</em>, EMNLP 2025 — <a href="https://arxiv.org/abs/2509.23471">arXiv:2509.23471</a>.</sub>

---

## The problem

Two embedding models produce vectors in geometrically different coordinate systems. The same sentence yields one 1536-dimensional point in one model's space and a completely different 1536-dimensional point in another's — even when both models come from the same vendor.

The naive migration is to re-embed all N documents with the new model. For a 1B-vector store at $0.02 per million tokens, that is roughly $6,000 and ~28 hours of wall-clock time — and the index is stale for every document that changes during the reindex window.

> [!TIP]
> **Key insight:** the two spaces are related by a rotation. Find the rotation, apply it at query time, and leave the index untouched.

---

## What is implemented (Phase 1 — Orthogonal Procrustes)

### Inputs

- `X_old` — `(N, d)`: N sample texts embedded with the **old** model.
- `X_new` — `(N, d)`: the **same** N texts embedded with the **new** model (row `i` is the same text in both).

### The optimization problem

Find an orthogonal matrix `R` (`R @ R.T = I` — a pure rotation, no stretching) that minimizes:

```
minimize  ‖X_old − X_new @ R‖²_F      (Frobenius norm — sum of squared element differences)
subject to  R @ R.T = I
```

### Closed-form solution

**Step 1 — Expand the Frobenius norm.**

```
‖X_old − X_new @ R‖²_F
= tr(X_old.T X_old)  −  2·tr(Rᵀ X_new.T X_old)  +  tr(Rᵀ X_new.T X_new R)
```

The first and last terms contain no `R` (or cancel under the orthogonality constraint), so minimizing the whole expression is equivalent to **maximizing**:

```
tr(Rᵀ M)    where  M = X_old.T @ X_new      (shape d × d — the cross-covariance matrix)
```

**Step 2 — Apply SVD to M.**

```
M = U @ Σ @ Vt          (Singular Value Decomposition)
```

For any orthogonal `R`, `tr(Rᵀ M) ≤ tr(Σ)`, and the upper bound is achieved exactly when:

```
R = U @ Vt
```

**Step 3 — Fit (~15–25 seconds on CPU).**

```python
M = old_vecs.T @ new_vecs        # (d, d) cross-covariance
U, _, Vt = np.linalg.svd(M)      # LAPACK DGESDD — O(d³), ~10–20s for d=1536
R = (U @ Vt).astype(np.float32)  # optimal rotation matrix
```

SVD of a `d × d` matrix costs `O(d³)`. For `d = 1536` that is ~3.6B operations; NumPy delegates to LAPACK's divide-and-conquer DGESDD, which completes in 10–20 seconds on a modern CPU. No GPU, PyTorch, or gradient descent required.

**Step 4 — Apply at query time.**

```python
adapted_vec = new_query_vec @ R.T     # rotate into the old model's geometric space
hits = qdrant.query("my_collection", query=adapted_vec, limit=10)
# the old index is searched, unchanged — no reindex needed
```

Orthogonal matrices preserve cosine similarity exactly (`cos(u, v) = cos(Ru, Rv)`), which is what makes the rotation safe: relative distances are preserved.

---

## Quality gate: Adapted Recall Rate (ARR)

ARR measures how well the adapter recovers the old model's nearest-neighbour structure, with no human annotation required:

```
ARR = mean_i( |oracle_top_k(i) ∩ adapted_top_k(i)| / k )

oracle_top_k(i)  = top-k neighbours of X_old[i] in the old-model corpus
adapted_top_k(i) = top-k neighbours of adapter.predict(X_new[i]) in the same corpus
```

- **ARR = 1.0** — the adapter perfectly recovers every neighbour the old model would return.
- **ARR = 0.97** (default threshold) — for every 100 documents the oracle returns, 97 are recovered. `migrate(strategy="drift-adapter")` raises `AdapterQualityError` below this.
- **Known failure: GloVe → MPNet at 71.5% ARR.** Token-level and full-sequence-attention models produce irreconcilable spaces. The 0.97 threshold is calibrated to catch this case and fall back automatically to `migrate(strategy="dual-write")`.

Self-retrieval (query equals document at the same index) is excluded from both sets to avoid trivially inflating the score.

---

## Out of scope (deferred)

| Variant | ARR gain over Procrustes | Why deferred |
|---|---|---|
| Low-rank affine (rotation + scale + bias) | ~+0.5% | Marginal gain for added complexity |
| Residual MLP (2-layer) | ~+2% | Requires PyTorch, a training loop, and a GPU — conflicts with the NumPy-only core |
| `DriftQueryClient` (query-time interception) | architecture | Breaking API change — planned for v1.0 |
| Chained adapters (A → B → C) | unknown | Mathematically sound, but recall properties are unstudied in the paper |

---

## Implementation map

| File | Role |
|---|---|
| `src/drift/adapter.py` | `DriftAdapter` — `fit()`, `predict()`, `save()`, `load()`, `_sample_paired_texts()` |
| `src/drift/shadow_eval.py` | `measure_arr()`, `AdapterQualityError` |
| `src/drift/migrate.py` | `migrate(strategy="drift-adapter")` — full pipeline: sample → 90/10 split → fit → ARR gate → save |

Test coverage: `tests/test_adapter.py` (8 cases), `tests/test_shadow_eval.py` (6 cases), and `tests/test_migrate.py` (drift-adapter happy path, quality-gate propagation, CLI smoke test).

---

## Reference

Drift-Adapter: A Practical Approach to Near Zero-Downtime Embedding Model Upgrades in Vector Databases. EMNLP 2025 — [arXiv:2509.23471](https://arxiv.org/abs/2509.23471).

Phase 1 (Orthogonal Procrustes) is what ships in `drift-spark` v0.5.0. Phase 2 (shadow-eval traffic split) and Phase 4 (query-time interception) are roadmap items.

---

<sub><a href="../README.md">← Back to README</a></sub>
