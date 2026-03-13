import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAISS_PATH = os.path.join(BASE_DIR, "data", "processed", "faiss_index.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "data", "processed", "chunks.csv")


@st.cache_resource
def load_retrieval_system():

    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)

    print("Loading metadata...")
    chunks_df = pd.read_csv(CHUNKS_PATH)

    print("System Ready!")

    return model, index, chunks_df


model, index, chunks_df = load_retrieval_system()


def retrieve_top_chunks(query, top_k=3):

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), top_k)

    results = []

    for idx in indices[0]:
        row = chunks_df.iloc[idx]

        results.append({
            "ticker": row["ticker"],
            "year": row["year"],
            "quarter": row["quarter"],
            "text": row["chunk_text"]
        })

    return results