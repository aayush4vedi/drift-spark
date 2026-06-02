"""
drift-adapter example — near-zero-cost embedding model upgrade.

Demonstrates upgrading from text-embedding-ada-002 to text-embedding-3-small
without re-indexing your document store. The adapter is a 9MB .npy file;
your Qdrant collection stays untouched.

Part A: Manual API (runs without Qdrant — pure NumPy)
Part B: Full pipeline via migrate() (requires Qdrant + existing collection)

Run Part A:
    pip install drift-spark
    python examples/drift_adapter.py

Run Part B:
    docker run -p 6333:6333 qdrant/qdrant
    python examples/drift_adapter.py --full
"""

import sys
import numpy as np

from drift.adapter import DriftAdapter
from drift.shadow_eval import AdapterQualityError, measure_arr

# ── Part A: Manual API ────────────────────────────────────────────────────────
# Shows the three steps any adapter workflow follows:
#   1. Collect paired embeddings (same texts, both models)
#   2. Fit the adapter on 90% of pairs; evaluate ARR on the remaining 10%
#   3. Save the adapter; load and apply at query time

print("=" * 60)
print("Part A: Manual DriftAdapter API (no Qdrant needed)")
print("=" * 60)

# Simulate paired embeddings.
# In production these come from adapter._sample_paired_texts() or from
# embedding the same texts with both models yourself.
N, d = 500, 64     # 500 paired samples, 64-dim (use 1536 in production)
rng = np.random.default_rng(42)

# Old model vectors — already in your Qdrant collection
X_old = rng.standard_normal((N, d)).astype(np.float32)
X_old /= np.linalg.norm(X_old, axis=1, keepdims=True)

# Simulate new model: same semantic content, different coordinate system.
# Real new-model vectors are just X_new = embed_new(same_texts).
# Here we simulate with a small rotation so the example is self-contained.
theta = np.radians(30)
R_sim = np.eye(d, dtype=np.float32)
R_sim[0, 0] = np.cos(theta)
R_sim[0, 1] = np.sin(theta)
R_sim[1, 0] = -np.sin(theta)
R_sim[1, 1] = np.cos(theta)
X_new = X_old @ R_sim

# Step 1: 90/10 train/val split
split = int(N * 0.9)
train_old, val_old = X_old[:split], X_old[split:]
train_new, val_new = X_new[:split], X_new[split:]

# Step 2: Fit (Orthogonal Procrustes — closed-form SVD, no training loop)
print(f"\nFitting adapter on {split} pairs...")
adapter = DriftAdapter().fit(train_old, train_new)
print(f"  R shape:      {adapter.R.shape}")
print(f"  Orthogonal:   {np.allclose(adapter.R @ adapter.R.T, np.eye(d), atol=1e-3)}")

# Step 3: Measure ARR — the self-supervised quality gate
print(f"\nMeasuring ARR on {len(val_old)} held-out pairs (k=10)...")
try:
    arr = measure_arr(adapter, val_old, val_new, k=10, threshold=0.97)
    print(f"  ARR: {arr:.4f}  ✓ PASS (≥ 0.97)")
except AdapterQualityError as e:
    print(f"  ARR: {e.arr:.4f}  ✗ FAIL — {e}")
    print("  Recommendation: use migrate(strategy='dual-write') for a full reindex.")
    sys.exit(1)

# Step 4: Save
adapter.save("drift_adapter_example.npy")
print(f"\nAdapter saved → drift_adapter_example.npy")

# Step 5: Load and apply at query time
loaded = DriftAdapter.load("drift_adapter_example.npy")

query_new = X_new[0]                    # new-model query vector (from user input)
query_adapted = loaded.predict(query_new)  # maps into old model's space

cos_sim = float(query_adapted @ X_old[0])  # should be ≈ 1.0 for a perfect rotation
print(f"\nQuery-time adapter:")
print(f"  cos_sim(adapted_query, oracle_old_vec) = {cos_sim:.4f}  (1.0 = perfect)")
print(f"  → send query_adapted to Qdrant as your search vector (no reindex needed)")

# ── Part B: Full pipeline via migrate() ───────────────────────────────────────
if "--full" not in sys.argv:
    print("\n" + "=" * 60)
    print("Part B skipped (pass --full to run against a live Qdrant instance)")
    print("Requires: docker run -p 6333:6333 qdrant/qdrant")
    print("          + an existing collection populated with drift embed()")
    print("=" * 60)
    sys.exit(0)

print("\n" + "=" * 60)
print("Part B: Full pipeline via migrate(strategy='drift-adapter')")
print("=" * 60)

from drift.migrate import migrate

SINK = "qdrant://localhost:6333/my_docs"   # must already exist, populated by drift embed()

print(f"\nRunning migrate(strategy='drift-adapter') on {SINK} ...")
print("(shadow_mode=True — no real API calls)")

try:
    run = migrate(
        from_model="openai/text-embedding-ada-002",
        to_model="openai/text-embedding-3-small",
        sink=SINK,
        strategy="drift-adapter",
        shadow_mode=True,       # remove for production (will call OpenAI API)
    )
    print(f"\n  ✓ ARR:          {run.arr:.4f}")
    print(f"  ✓ Adapter path: {run.adapter_path}")
    print(f"  ✓ Trained on:   {run.n_source} paired samples")
    print(f"  ✓ Duration:     {run.duration_s:.1f}s")
    print(f"\n  Your {SINK} collection is untouched.")
    print(f"  Load the adapter at query time:")
    print(f"    adapter = DriftAdapter.load('{run.adapter_path}')")
    print(f"    adapted_vec = adapter.predict(new_model_query_vec)")
    print(f"    qdrant.search('{SINK.split('/')[-1]}', adapted_vec, limit=10)")
except AdapterQualityError as e:
    print(f"\n  ✗ Adapter quality too low: ARR={e.arr:.3f} < {e.threshold}")
    print(f"  Recommendation: use migrate(strategy='dual-write') for a full reindex.")
    sys.exit(1)
