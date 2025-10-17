import pandas as pd

df = pd.read_csv("data/cleaned/kidney_clean.csv")

# Check columns with object (string) types
print("🔍 Columns with non-numeric data:")
print(df.select_dtypes(include=['object']).columns.tolist())

# Show a few unique values for these columns
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}: {df[col].unique()[:5]}")
