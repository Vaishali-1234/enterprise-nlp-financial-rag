import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.csv")


# ============================================
# LAZY LOADING WITH CACHE
# ============================================

@st.cache_resource
def load_retrieval_resources():

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)
    index.nprobe = 10

    print("Loading metadata...")
    chunks_df = pd.read_csv(CHUNKS_PATH)

    print("Retrieval system ready.")
    return model, index, chunks_df


model, index, chunks_df = load_retrieval_resources()


# ============================================
# QUERY EMBEDDING CACHE
# ============================================

@lru_cache(maxsize=256)
def encode_query(query: str) -> np.ndarray:
    """Encode and cache query embeddings to avoid redundant inference."""
    return model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )


# ============================================
# RETRIEVAL
# ============================================

def retrieve_top_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top_k most relevant prepared-section chunks for a query.

    Each result includes a relevance_score (0-1) which is the cosine
    similarity between the query embedding and the chunk embedding.
    Since both are L2-normalized, the FAISS inner product distance
    IS the cosine similarity — no extra computation needed.
    """

    query_embedding = encode_query(query)
    fetch_k = top_k * 5

    # distances = inner product = cosine similarity (since normalized)
    distances, indices = index.search(np.array(query_embedding), fetch_k)

    results = []
    seen = set()

    skip_phrases = [
        "without limitation",
        "forward-looking statements",
        "operator instructions",
        "our next question",
        "please refer to",
        "form 10-k",
        "form 10-q",
    ]

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        row = chunks_df.iloc[idx]
        text = row["chunk_text"].lower()

        if "section_type" in chunks_df.columns and row.get("section_type") == "qa":
            continue

        if any(phrase in text for phrase in skip_phrases):
            continue

        key = row["chunk_text"][:100]
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "ticker": row["ticker"],
            "year": row["year"],
            "quarter": row["quarter"],
            "text": row["chunk_text"],
            # Cosine similarity between query and chunk (0 to 1).
            # Clipped to [0,1] since inner product on normalized vectors
            # can occasionally return tiny negative values.
            "relevance_score": round(float(np.clip(score, 0, 1)), 3)
        })

        if len(results) >= top_k:
            break

    return results
