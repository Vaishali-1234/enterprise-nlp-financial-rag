"""
build_ivf_index.py
------------------
Run this ONCE from your project root:
    python build_ivf_index.py

It loads your existing chunk_embeddings.npy and rebuilds faiss_index.bin
as an IVF index — no re-encoding needed. Takes ~2-3 minutes.
"""

import numpy as np
import faiss
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunk_embeddings.npy")
FAISS_OUT_PATH  = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")

# ── 1. Load existing embeddings ──────────────────────────────────────────────
print("Loading embeddings from disk...")
embeddings = np.load(EMBEDDINGS_PATH)                  # shape: (N, 384)
embeddings = embeddings.astype(np.float32)             # FAISS requires float32
num_chunks, embedding_dim = embeddings.shape
print(f"Loaded {num_chunks:,} vectors of dim {embedding_dim}")

# ── 2. IVF parameters ────────────────────────────────────────────────────────
# NLIST: number of clusters. Rule of thumb = 4 * sqrt(N)
# For 1.2M vectors → ~4376 clusters (same as what printed before)
NLIST  = max(64, int(4 * np.sqrt(num_chunks)))
NPROBE = 10   # clusters searched per query — higher = more accurate, slightly slower
print(f"IVF config — clusters: {NLIST}, nprobe: {NPROBE}")

# ── 3. Build IVF index ───────────────────────────────────────────────────────
print("\nBuilding IVF index (this takes ~2-3 mins for 1.2M vectors)...")
quantizer = faiss.IndexFlatIP(embedding_dim)           # inner product on normalized vectors
index     = faiss.IndexIVFFlat(quantizer, embedding_dim, NLIST, faiss.METRIC_INNER_PRODUCT)

print("Training clusters...")
index.train(embeddings)                                # learns cluster centroids
print("Training done! Adding vectors...")
index.add(embeddings)                                  # adds all vectors to index
index.nprobe = NPROBE

print(f"Index built — {index.ntotal:,} vectors stored.")

# ── 4. Save new index (overwrites old flat index) ────────────────────────────
faiss.write_index(index, FAISS_OUT_PATH)
print(f"\nIVF index saved to: {FAISS_OUT_PATH}")
print("Done! You can now run: streamlit run app.py")
