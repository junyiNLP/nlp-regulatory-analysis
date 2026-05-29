import pandas as pd
import re

# Load dataset
df = pd.read_csv("1. index_dataset_index.csv")

# ==== NEW: rename text → index ====
if "text" in df.columns:
    df = df.rename(columns={"text": "index"})


def clean_label(label):
    if pd.isna(label):
        return label

    # Convert to string
    label = str(label)

    # Replace ',' and '.' with '_'
    label = label.replace(",", "_").replace(".", "_")

    # Remove leading zeros in each segment: 4_01 → 4_1
    parts = label.split("_")
    cleaned_parts = [str(int(p)) if p.isdigit() else p for p in parts]

    # Rejoin
    return "_".join(cleaned_parts)

# Apply cleaning
df["label"] = df["label"].apply(clean_label)

# Save output
df.to_csv("index_dataset.csv", index=False)

print("Done! Labels cleaned, normalized, and 'text' renamed to 'index'.")
