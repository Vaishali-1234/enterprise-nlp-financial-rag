import pandas as pd

print("START", flush=True)

# df = pd.read_parquet("data/processed/chunks.parquet")
df = pd.read_parquet(r"C:/Users/vvais/enterprise-nlp/data/processed/chunks.parquet")
print("Columns:", flush=True)
print(list(df.columns), flush=True)

print("\nFirst row:", flush=True)
print(df.head(1), flush=True)

print("END", flush=True)

