import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index("data/processed/faiss_index.bin")

print("Loading chunk metadata...")
chunks_df = pd.read_parquet("data/processed/chunks.parquet")

print("System ready!")


def retrieve_top_chunks(query, top_k=5):

    # Convert query to embedding
    query_embedding = model.encode([query])

    # Search FAISS
    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for idx in indices[0]:
        row = chunks_df.iloc[idx]

        results.append({
            "ticker": row["ticker"],
            "quarter": row["quarter"],
            "text": row["chunk_text"]
        })

    return results


if __name__ == "__main__":

    query = "AI investment strategy and machine learning infrastructure"

    results = retrieve_top_chunks(query)

    print("\nTop Results:\n")

    for r in results:
        print("Ticker:", r["ticker"])
        print("Quarter:", r["quarter"])
        print(r["text"])
        print("\n--------------------------\n")