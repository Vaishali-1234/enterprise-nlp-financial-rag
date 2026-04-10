import pandas as pd

print("Loading parquet file...")

df = pd.read_parquet("data/processed/chunks.parquet")

print("Saving CSV...")

df.to_csv("data/processed/chunks.csv", index=False)

print("✅ CSV created successfully!")