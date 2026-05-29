"""
==========================================================
 RDTII Website Discovery Crawler – Pillar 4 (Test: Thailand Only)
==========================================================
Purpose:
----------------------------------------------------------
Reads dataset/pillar_dictionary/pillar_4.json,
expands pillar_dictionary with SentenceTransformer,
runs open Google searches (Thailand only),
and saves discovered URLs to web_crawler/rdtii_pillar4_open_sites_test.csv
----------------------------------------------------------
"""

import os
import json
import pandas as pd
from tqdm import tqdm
from googlesearch import search
from sentence_transformers import SentenceTransformer, util



# === [1] PATH CONFIGURATION ===
from pathlib import Path

# Root folder of rdtii/
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to pillar dictionary
PILLAR_DICT_DIR = BASE_DIR / "BERT_training" / "dataset" / "pillar_dictionary"

# Specify which pillar to load
PILLAR_ID = 4
KEYWORD_PATH = PILLAR_DICT_DIR / f"pillar_{PILLAR_ID}.json"

# Output file
OUTPUT_FILE = BASE_DIR / "web_crawler" / f"rdtii_pillar{PILLAR_ID}_open_sites_test.csv"

# Debug prints
print("📂 BASE_DIR:", BASE_DIR)
print("📂 PILLAR_DICT_DIR:", PILLAR_DICT_DIR)
print("📄 KEYWORD_PATH:", KEYWORD_PATH)
print("Exists:", KEYWORD_PATH.exists())



# === [2] LOAD JSON ===
with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
    pillar_data = json.load(f)

pillar_name = pillar_data.get("pillar_name", "Pillar 4")
print(f"📘 Loaded pillar: {pillar_name}")

# Collect all pillar_dictionary
all_keywords = set(pillar_data.get("pillar_dictionary", []))
for _, indicator in pillar_data.get("indicators", {}).items():
    all_keywords.update(indicator.get("pillar_dictionary", []))

print(f"✅ {len(all_keywords)} total base pillar_dictionary loaded")

# === [3] INITIALIZE MODEL & EXPAND KEYWORDS ===
model = SentenceTransformer("all-MiniLM-L6-v2")

vocab = [
    "law", "regulation", "act", "policy", "framework", "authority",
    "intellectual property office", "patent office", "copyright act",
    "WIPO treaty", "innovation protection", "trade secret act",
    "industrial design law", "IP enforcement"
]

base_vecs = model.encode(list(all_keywords), convert_to_tensor=True)
vocab_vecs = model.encode(vocab, convert_to_tensor=True)
cos_scores = util.cos_sim(base_vecs, vocab_vecs)

expansions = set()
for i, base in enumerate(all_keywords):
    for j, vocab_term in enumerate(vocab):
        if cos_scores[i][j] > 0.75:
            expansions.add(vocab_term)

expanded_keywords = all_keywords.union(expansions)
print(f"🧠 {len(expanded_keywords)} total expanded pillar_dictionary")

# === [4] COUNTRY CONTEXT – TEST: Only Thailand ===
COUNTRIES = ["Thailand"]

# === [5] RUN OPEN SEARCH (LIMITED) ===
results = []
for country in COUNTRIES:
    for kw in tqdm(list(expanded_keywords)[:5], desc=f"{country} (test batch)"):
        # ⬆ You can increase the number (e.g., [:10]) for larger test runs
        query = f"{kw} {country}"
        try:
            for url in search(query, num_results=3):  # reduce number for faster test
                results.append({
                    "Country": country,
                    "Pillar": pillar_name,
                    "Keyword": kw,
                    "URL": url
                })
        except Exception as e:
            print(f"⚠️ Error during search for '{query}': {e}")

# === [6] CLEAN & SAVE ===
df = pd.DataFrame(results)
df.drop_duplicates(subset="URL", inplace=True)

df = df[~df["URL"].str.contains(
    r"(facebook|linkedin|twitter|youtube|blogspot|reddit|tiktok)",
    case=False
)]

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n💾 Saved {len(df)} unique websites → {OUTPUT_FILE}")
print("✅ Test completed (Thailand only).")
