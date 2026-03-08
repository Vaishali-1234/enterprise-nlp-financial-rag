import pandas as pd

print("Loading pickle...")

df = pd.read_pickle("data/processed/chunks.pkl")

print("Saving as parquet...")

df.to_parquet("data/processed/chunks.parquet")

print("Done.")