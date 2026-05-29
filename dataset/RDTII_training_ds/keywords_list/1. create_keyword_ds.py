import pandas as pd
import re

# ================================
# LOAD ORIGINAL DATASET
# ================================
df = pd.read_csv(
    "0. keywords_list_original.csv",
    sep=None,
    engine="python",
    encoding="utf-8-sig"   # <-- removes BOM (\ufeff)
)

# Rename BOM column to a normal name
df = df.rename(columns={"﻿Indicator_index": "Indicator_index"})

print("Loaded columns:", df.columns.tolist())

# -----------------------------------------------------
# 1. CLEAN LABEL (Indicator_index -> label)
# -----------------------------------------------------
def clean_label(label):
    if pd.isna(label):
        return label

    label = str(label)
    label = label.replace(",", "_").replace(".", "_")

    parts = label.split("_")
    cleaned = []

    for p in parts:
        if p.isdigit():
            cleaned.append(str(int(p)))  # removes leading zeros
        else:
            cleaned.append(p)

    return "_".join(cleaned)

df["label"] = df["Indicator_index"].apply(clean_label)

# -----------------------------------------------------
# 2. BUILD FINAL ROWS (sector, subject, measure)
# -----------------------------------------------------
rows = []

def add_row(label, text, source):
    if pd.notna(text) and str(text).strip() != "":
        rows.append({
            "label": label,
            "text": str(text).strip(),
            "source": source
        })

for _, row in df.iterrows():
    label = row["label"]
    add_row(label, row["Tag_Sector"], "sector")
    add_row(label, row["Tag_Subject"], "subject")
    add_row(label, row["Tag_Measure"], "measure")

final_df = pd.DataFrame(rows)

# ================================
# SAVE OUTPUT
# ================================
final_df.to_csv("1. keywords_list_processed.csv", index=False)

print("Done! Saved as 1. keywords_list_processed.csv")
print(final_df.head())
