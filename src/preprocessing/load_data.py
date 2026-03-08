import pandas as pd
import re
import os
from sentence_transformers import SentenceTransformer
import random

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

file_path = "data/raw/motley-fool-data.pkl"

print("Loading dataset...\n")
df = pd.read_pickle(file_path)
print("Dataset loaded!\n")

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# =====================================================
# 2️⃣ DATE + STRUCTURE POLISH
# =====================================================

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["q"].str.extract(r"(\d{4})").astype(int)
df["quarter_num"] = df["q"].str.extract(r"Q(\d)").astype(int)

print("\nDate column type:", df["date"].dtype)

duplicate_count = df.duplicated(subset=["transcript"]).sum()
print("Duplicate transcripts:", duplicate_count)

# =====================================================
# 3️⃣ BASIC CLEANING FUNCTION
# =====================================================

def basic_clean(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = text.strip()
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = text.replace("�", "")
    return text

# =====================================================
# 4️⃣ METADATA VALIDATION
# =====================================================

df["length"] = df["transcript"].apply(len)

print("\nTranscript Length Statistics:")
print("Min:", df["length"].min())
print("Max:", df["length"].max())
print("Avg:", df["length"].mean())

print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique tickers:", df["ticker"].nunique())

# =====================================================
# 5️⃣ SECTION SPLITTING
# =====================================================

def split_sections(text):
    qa_markers = [
        "Question-and-Answer Session",
        "Our next question",
        "Operator\nOur next question"
    ]
    
    for marker in qa_markers:
        if marker in text:
            split_index = text.find(marker)
            return text[:split_index], text[split_index:]
    
    return text, ""

# =====================================================
# 6️⃣ TOKEN-BASED CHUNKING (WITH OVERLAP)
# =====================================================

def chunk_text(text, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =====================================================
# 7️⃣ BUILD CHUNK DATASET
# =====================================================

chunk_records = []

for idx in range(len(df)):

    row = df.loc[idx]
    cleaned_text = basic_clean(row["transcript"])

    prepared_text, qa_text = split_sections(cleaned_text)

    # -------- PREPARED --------
    prepared_chunks = chunk_text(prepared_text)

    for chunk_index, chunk in enumerate(prepared_chunks):
        chunk_records.append({
            "transcript_index": idx,
            "ticker": row["ticker"],
            "year": row["year"],
            "quarter": row["q"],
            "quarter_num": row["quarter_num"],
            "date": row["date"],
            "exchange": row["exchange"],
            "section_type": "prepared",
            "chunk_id": f"{idx}_prep_{chunk_index}",
            "chunk_text": chunk
        })

    # -------- Q&A --------
    if qa_text.strip() != "":
        qa_chunks = chunk_text(qa_text)

        for chunk_index, chunk in enumerate(qa_chunks):
            chunk_records.append({
                "transcript_index": idx,
                "ticker": row["ticker"],
                "year": row["year"],
                "quarter": row["q"],
                "quarter_num": row["quarter_num"],
                "date": row["date"],
                "exchange": row["exchange"],
                "section_type": "qa",
                "chunk_id": f"{idx}_qa_{chunk_index}",
                "chunk_text": chunk
            })

# =====================================================
# 8️⃣ CREATE FINAL CHUNK DATAFRAME
# =====================================================

chunk_df = pd.DataFrame(chunk_records)

print("\nChunk Dataset Shape:")
print(chunk_df.shape)

print("\nSection distribution:")
print(chunk_df["section_type"].value_counts())

print("\nFirst 5 rows:")
print(chunk_df.head())

# =====================================================
# 9️⃣ SAVE PROCESSED DATA
# =====================================================

os.makedirs("data/processed", exist_ok=True)
chunk_df.to_pickle("data/processed/chunks.pkl")

print("\nSaved chunk dataset to data/processed/chunks.pkl")

# =====================================================
# 🔟 TOKEN LENGTH AUDIT (SAMPLE)
# =====================================================

print("\nRunning Token Length Audit...\n")

model = SentenceTransformer("all-MiniLM-L6-v2")
model.max_seq_length = 256   # 🔥 enforce safety

sample_chunks = chunk_df.sample(1000, random_state=42)["chunk_text"].tolist()

token_lengths = []

for text in sample_chunks:
    tokens = model.tokenizer.encode(text)
    token_lengths.append(len(tokens))

print("Token Length Statistics (Sample of 1000 chunks):")
print("Min tokens:", min(token_lengths))
print("Max tokens:", max(token_lengths))
print("Average tokens:", sum(token_lengths) / len(token_lengths))