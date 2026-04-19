import os
import re
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from functools import lru_cache
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.parquet")

# ============================================
# FILTERING CONSTANTS
# ============================================

MIN_RELEVANCE_THRESHOLD = 0.50
MAX_CHUNKS_PER_TICKER = 2

# Ticker aliases — maps company name mentions to ticker symbols
TICKER_ALIASES = {
    "amazon": "AMZN", "aws": "AMZN",
    "apple": "AAPL", "iphone": "AAPL",
    "microsoft": "MSFT", "azure": "MSFT",
    "google": "GOOGL", "alphabet": "GOOGL",
    "meta": "META", "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "salesforce": "CRM",
    "oracle": "ORCL",
    "intel": "INTC",
}


# ============================================
# LAZY LOADING WITH CACHE
# ============================================

@st.cache_resource
def load_retrieval_resources():
    print("⚠️ LOADING RESOURCES — this should only print ONCE")

    print("Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    print("Loading cross-encoder reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)
    index.nprobe = 10

    print("Loading metadata...")
    chunks_df = pd.read_parquet(CHUNKS_PATH)

    print("Retrieval system ready.")
    return model, reranker, index, chunks_df


model, reranker, index, chunks_df = load_retrieval_resources()


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
# METADATA EXTRACTION
# ============================================

def extract_years(query: str) -> list[int]:
    """Extract year mentions from query e.g. '2022' -> [2022]"""
    return [int(y) for y in re.findall(r'\b(20\d{2})\b', query)]


def extract_ticker(query: str) -> str | None:
    """Extract ticker from query via alias map."""
    q = query.lower()
    for alias, ticker in TICKER_ALIASES.items():
        if alias in q:
            return ticker
    tickers = re.findall(r'\b([A-Z]{2,5})\b', query)
    for t in tickers:
        if t in chunks_df["ticker"].values:
            return t
    return None


# ============================================
# RETRIEVAL
# ============================================

def retrieve_top_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve top_k most relevant chunks for a query.

    Applies staged filtering:
    Stage 1 - Metadata pre-filter (year + ticker from query)
    Stage 2 - FAISS vector search on filtered subset
    Stage 3 - Post-filter (boilerplate, relevance threshold, diversity)
    Stage 4 - Cross-encoder reranking on top-25 candidates
    """

    query_embedding = encode_query(query)
    fetch_k = top_k * 10

    # ── Stage 1: Metadata pre-filter ──────────────────────────────────────
    filtered_df = chunks_df.copy()

    years = extract_years(query)
    if years:
        filtered_df = filtered_df[filtered_df["year"].isin(years)]

    ticker = extract_ticker(query)
    if ticker:
        ticker_df = filtered_df[filtered_df["ticker"] == ticker]
        if len(ticker_df) >= top_k * 2:
            filtered_df = ticker_df

    if len(filtered_df) < top_k * 2:
        filtered_df = chunks_df.copy()

    # ── Stage 2: FAISS search ──────────────────────────────────────────────
    filtered_indices = filtered_df.index.tolist()
    distances, indices = index.search(np.array(query_embedding), fetch_k)

    # ── Stage 3: Post-filter ───────────────────────────────────────────────
    candidates = []
    seen = set()
    ticker_counts = {}

    skip_phrases = [
        "without limitation",
        "forward-looking statements",
        "operator instructions",
        "our next question",
        "please refer to",
        "form 10-k",
        "form 10-q",
        "— analyst",
        "— operator",
        "thank you. operator",
        "our next question comes",
        "cowen and company",
        "bank of america",
        "goldman sachs",
        "morgan stanley",
        "jp morgan",
        "oppenheimer",
        "jefferies",
        "citigroup",
        "barclays",
    ]

    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue

        if idx not in filtered_indices:
            continue

        if score < MIN_RELEVANCE_THRESHOLD:
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

        ticker_val = row["ticker"]
        ticker_counts[ticker_val] = ticker_counts.get(ticker_val, 0)
        if ticker_counts[ticker_val] >= MAX_CHUNKS_PER_TICKER:
            continue
        ticker_counts[ticker_val] += 1

        candidates.append({
            "ticker": ticker_val,
            "year": row["year"],
            "quarter": row["quarter"],
            "text": row["chunk_text"],
            "faiss_score": round(float(np.clip(score, 0, 1)), 3)
        })

        if len(candidates) >= top_k * 5:  # collect up to 25 for reranking
            break

    # ── Stage 4: Cross-encoder reranking ──────────────────────────────────
    if candidates:
        pairs = [[query, c["text"]] for c in candidates]
        rerank_scores = reranker.predict(pairs)

        for i, c in enumerate(candidates):
            c["relevance_score"] = round(float(rerank_scores[i]), 3)

        # Sort by reranker score descending
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Normalize reranker scores to 0-1 range for display
        scores = [c["relevance_score"] for c in candidates]
        min_s, max_s = min(scores), max(scores)
        for c in candidates:
            if max_s > min_s:
                c["relevance_score"] = round((c["relevance_score"] - min_s) / (max_s - min_s), 3)
            else:
                c["relevance_score"] = 1.0

    return candidates[:top_k]
