import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import os

# ============================================
# PROJECT ROOT
# ============================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

chunks_path = os.path.join(BASE_DIR, "data", "processed", "chunks.pkl")

print("Loading chunk dataset...\n")
print("Looking for file at:", chunks_path)

chunk_df = pd.read_pickle(chunks_path)

print("Chunks loaded!")
print("Total chunks:", len(chunk_df))

texts = chunk_df["chunk_text"].tolist()

# ============================================
# LOAD MODEL
# ============================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256

print("Model loaded!")

# ============================================
# PARAMETERS
# ============================================

BATCH_SIZE = 128
embedding_dim = 384
num_chunks = len(texts)

# IVF index parameters:
# - NLIST: number of clusters (Voronoi cells). More clusters = faster search but
#   needs more vectors to train. Rule of thumb: sqrt(num_chunks) to 4*sqrt(num_chunks)
# - We use IndexIVFFlat which clusters first, then does exact search within clusters.
#   This gives ~60-80% memory reduction vs IndexFlatL2 with minimal accuracy loss.
NLIST = max(64, int(4 * np.sqrt(num_chunks)))  # Adaptive cluster count
NPROBE = 10  # How many clusters to search at query time (higher = more accurate, slower)

print(f"\nIVF config — clusters: {NLIST}, probe at query: {NPROBE}")

# ============================================
# PREALLOCATE EMBEDDING MATRIX
# ============================================

print("\nAllocating embedding matrix...")

embeddings = np.zeros((num_chunks, embedding_dim), dtype=np.float32)

print("Matrix shape:", embeddings.shape)

# ============================================
# GENERATE EMBEDDINGS
# ============================================

print("\nGenerating embeddings...\n")

for start in tqdm(range(0, num_chunks, BATCH_SIZE)):

    end = start + BATCH_SIZE
    batch_texts = texts[start:end]

    batch_embeddings = model.encode(
        batch_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize so cosine similarity == dot product
        show_progress_bar=False
    )

    embeddings[start:end] = batch_embeddings

# ============================================
# BUILD IVF FAISS INDEX
# ============================================
# Previously: IndexFlatL2 — brute-force, O(n) search, high memory
# Now: IndexIVFFlat — clusters vectors into NLIST buckets, only searches
# the nearest NPROBE buckets at query time. Much faster and lighter.

print("\nBuilding IVF FAISS index...")

# Quantizer decides how to compare cluster centroids
quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product (works with normalized vectors)

# IVF index: clusters vectors, then does flat search within each cluster
index = faiss.IndexIVFFlat(quantizer, embedding_dim, NLIST, faiss.METRIC_INNER_PRODUCT)

# IVF index MUST be trained on representative data before adding vectors
print(f"Training IVF index on {num_chunks} vectors...")
index.train(embeddings)
print("Training complete!")

# Add all vectors to the index
index.add(embeddings)
print(f"Added {index.ntotal} vectors to index.")

# Set nprobe — stored inside the index so retrieve_chunks.py picks it up automatically
index.nprobe = NPROBE

# ============================================
# SAVE EMBEDDINGS + INDEX
# ============================================

processed_dir = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# Save raw embeddings (useful for re-indexing or experimenting later)
embeddings_path = os.path.join(processed_dir, "chunk_embeddings.npy")
np.save(embeddings_path, embeddings)
print("\nEmbeddings saved to:", embeddings_path)

# Save the IVF FAISS index
faiss_path = os.path.join(processed_dir, "faiss_index.bin")
faiss.write_index(index, faiss_path)
print("IVF FAISS index saved to:", faiss_path)

print(f"\nIndex stats — type: IVFFlat | clusters: {NLIST} | vectors: {index.ntotal}")
print("\nEmbedding generation complete!")
