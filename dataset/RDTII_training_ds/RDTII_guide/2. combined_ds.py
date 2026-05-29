import pandas as pd
from pathlib import Path

# ==========================
# PATH CONFIG
# ==========================
BASE_DIR = Path(__file__).resolve().parent

UNIFIED_PATH = BASE_DIR / "0. unified_ds_plus_keywords.csv"
GUIDE_PATH = BASE_DIR / "1. RDTII_guide.csv"
OUTPUT_PATH = BASE_DIR / "unified_ds_full.csv"

# ==========================
# LOAD DATASETS
# ==========================
df_unified = pd.read_csv(UNIFIED_PATH)

# Auto-detect separator for RDTII guide file
df_guide = pd.read_csv(
    GUIDE_PATH,
    sep=None,
    engine="python",
    encoding="utf-8-sig"
)

print("Unified dataset:", df_unified.shape)
print("Guide dataset (raw):", df_guide.shape)
print("Guide columns:", df_guide.columns.tolist())

# ==========================
# CLEAN GUIDE COLUMN NAMES
# ==========================
# Handle BOM also
df_guide = df_guide.rename(columns={
    "Pillar/indicator_ID": "label",
    "﻿Pillar/indicator_ID": "label",  # BOM version
    "Guide_desc": "text",
    "﻿Guide_desc": "text"
})

# Add source column
df_guide["source"] = "pillar_guide"

# Keep correct columns only
df_guide = df_guide[["label", "text", "source"]]

# ==========================
# CONCATENATE
# ==========================
df_combined = pd.concat([df_unified, df_guide], ignore_index=True)

print("FINAL combined dataset:", df_combined.shape)

# ==========================
# SAVE
# ==========================
df_combined.to_csv(OUTPUT_PATH, index=False)

print("Saved final dataset to:")
print(OUTPUT_PATH)
