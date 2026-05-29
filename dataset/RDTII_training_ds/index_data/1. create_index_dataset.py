import pandas as pd
import os

# === [1] PATH CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current folder (index_data)
INPUT_FILE = os.path.join(BASE_DIR, "0. index_consolidated.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "1. index_dataset_index.csv")  # save in same folder

# === [2] LOAD DATA (robust handling) ===
try:
    df = pd.read_csv(INPUT_FILE, sep=None, engine='python')
except Exception:
    print("⚠️ Default read failed, retrying with semicolon separator...")
    df = pd.read_csv(INPUT_FILE, sep=';', engine='python')

# === [3] CLEAN COLUMN NAMES ===
df.columns = [c.strip() for c in df.columns]

# === [4] SELECT RELEVANT COLUMNS ===
columns_needed = [
    'Economy',
    'Pillar_ID',
    'Indicator_ID',
    'Act and/or practice',
    'Impact or comments on Acts or practices'
]

missing_cols = [col for col in columns_needed if col not in df.columns]
if missing_cols:
    raise ValueError(f"❌ Missing columns: {missing_cols}\nAvailable columns: {list(df.columns)}")

df = df[columns_needed].copy()

# === [5] RENAME AND CLEAN ===
df.rename(columns={
    'Economy': 'economy',
    'Pillar_ID': 'pillar_id',
    'Indicator_ID': 'label',
    'Impact or comments on Acts or practices': 'text',
    'Act and/or practice': 'act_or_principle'
}, inplace=True)

# === [6] ADD NEW COLUMNS ===
df['doc_id'] = df['economy'].astype(str) + "_" + df['label'].astype(str)
df['source'] = 'index'

# === [7] REORDER COLUMNS ===
df = df[['text', 'act_or_principle', 'label', 'pillar_id', 'economy', 'doc_id', 'source']]

# === [8] DROP EMPTY ROWS ===
df.dropna(subset=['text', 'label', 'pillar_id', 'economy'], inplace=True)

# === [9] SAVE OUTPUT IN SAME FOLDER ===
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

print("✅ BERT dataset created successfully!")
print(f"Saved to: {OUTPUT_FILE}")
print(df.head(5))
