import pandas as pd
from pathlib import Path

# ==========================
# PATH CONFIG
# ==========================
BASE_DIR = Path(__file__).resolve().parent

UNIFIED_PATH = BASE_DIR / "0. unified_semantic_dataset_extended.csv"
KEYWORD_PATH = BASE_DIR / "1. keywords_list_processed.csv"
OUTPUT_PATH = BASE_DIR / "0. unified_ds_plus_keywords.csv"

# ==========================
# LOAD DATASETS
# ==========================
df_unified = pd.read_csv(UNIFIED_PATH)
df_keywords = pd.read_csv(KEYWORD_PATH)

print("Unified dataset size:", df_unified.shape)
print("Keyword dataset size:", df_keywords.shape)

# ==========================
# ENSURE SAME COLUMNS
# ==========================
expected_cols = ["label", "text", "source"]

df_unified = df_unified[expected_cols]
df_keywords = df_keywords[expected_cols]

# ==========================
# CONCATENATE
# ==========================
df_combined = pd.concat([df_unified, df_keywords], ignore_index=True)

print("Final combined size:", df_combined.shape)

# ==========================
# SAVE RESULT
# ==========================
df_combined.to_csv(OUTPUT_PATH, index=False)

print("Saved combined dataset to:")
print(OUTPUT_PATH)
