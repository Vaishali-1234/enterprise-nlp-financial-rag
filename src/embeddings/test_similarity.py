import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
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
embeddings_path = os.path.join(BASE_DIR, "data", "processed", "test_embeddings.npy")

# ============================================
# LOAD DATA
# ============================================

print("Loading chunks...")
chunk_df = pd.read_pickle(chunks_path)

print("Loading embeddings...")
embeddings = np.load(embeddings_path)

print("Embedding shape:", embeddings.shape)

# Only use first 2000 chunks (because test embeddings used them)
chunk_df = chunk_df.iloc[:2000]

# ============================================
# LOAD MODEL
# ============================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================
# QUERY
# ============================================

query = "AI investment strategy and machine learning infrastructure"

print("\nQuery:", query)

query_embedding = model.encode(query)

# ============================================
# SIMILARITY
# ============================================

similarities = cosine_similarity(
    [query_embedding],
    embeddings
)[0]

# ============================================
# TOP RESULTS
# ============================================

top_k = 5
top_indices = similarities.argsort()[-top_k:][::-1]

print("\nTop Similar Chunks:\n")

for i in top_indices:
    
    print("Similarity Score:", similarities[i])
    print("Chunk:\n")
    
    print(chunk_df.iloc[i]["chunk_text"][:500])
    
    print("\n" + "-"*80 + "\n")