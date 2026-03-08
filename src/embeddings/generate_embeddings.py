import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings[start:end] = batch_embeddings

# ============================================
# SAVE EMBEDDINGS
# ============================================

processed_dir = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(processed_dir, exist_ok=True)

save_path = os.path.join(processed_dir, "chunk_embeddings.npy")

np.save(save_path, embeddings)

print("\nEmbeddings saved to:", save_path)

print("\nEmbedding generation complete!")