"""
==========================================================
 RDTII Website Discovery Crawler – Pillar 4 (Open Search)
==========================================================
Reads dataset/pillar_dictionary/pillar_4.json,
expands pillar_dictionary with SentenceTransformer,
runs open Google searches (no domain restriction),
and saves discovered URLs to web_crawler/rdtii_pillar4_open_sites.csv
----------------------------------------------------------
"""

# finally this experiment can return the Google search pages by correct links, but stuck at anti-robot mechanism.

import os
import json
import pandas as pd
from tqdm import tqdm
from googlesearch import search
from sentence_transformers import SentenceTransformer, util

# === [1] PATH CONFIGURATION ===
from pathlib import Path

# Project root (rdtii/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Path to pillar dictionary file
KEYWORD_PATH = BASE_DIR / "BERT_training" / "dataset" / "pillar_dictionary" / "pillar_4.json"

# Output file location
OUTPUT_FILE = BASE_DIR / "web_crawler" / "rdtii_pillar4_open_sites_test.csv"

# Debug: verify paths
print("BASE_DIR:", BASE_DIR)
print("KEYWORD_PATH:", KEYWORD_PATH)
print("Exists:", KEYWORD_PATH.exists())


# === [2] LOAD JSON ===
with open(KEYWORD_PATH, "r", encoding="utf-8") as f:
    pillar_data = json.load(f)

pillar_name = pillar_data.get("pillar_name", "Pillar 4")
print(f"📘 Loaded pillar: {pillar_name}")

# --- FIXED SECTION BELOW ---
# Collect all keywords (base + indicator-level)
all_keywords = set(pillar_data.get("keywords", []))

for _, indicator in pillar_data.get("indicators", {}).items():
    all_keywords.update(indicator.get("keywords", []))
# --- END FIX ---

print(f"✅ {len(all_keywords)} total base keywords loaded")

# === [3] INITIALIZE MODEL & EXPAND KEYWORDS ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# Small helper vocabulary for semantic expansion
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

# === [4] COUNTRY CONTEXT (optional, no domain restriction) ===
COUNTRIES = [
    "Thailand", "Malaysia", "Vietnam", "Philippines",
    "Indonesia", "Singapore", "Cambodia", "Lao PDR",
    "Myanmar", "Brunei"
]

# === [5] RUN OPEN SEARCH ===
results = []
for country in COUNTRIES:
    for kw in tqdm(expanded_keywords, desc=f"{country}"):
        query = f"{kw} {country}"  # keyword + country
        try:
            for url in search(query, num_results=5):
                results.append({
                    "Country": country,
                    "Pillar": pillar_name,
                    "Keyword": kw,
                    "URL": url
                })
        except Exception as e:
            print("⚠️ Error during search:", e)

# === [6] CLEAN & SAVE ===
df = pd.DataFrame(results)
df.drop_duplicates(subset="URL", inplace=True)

# Remove obvious social media / noise
df = df[~df["URL"].str.contains(
    r"(facebook|linkedin|twitter|youtube|blogspot|reddit|tiktok)",
    case=False
)]

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n💾 Saved {len(df)} unique websites → {OUTPUT_FILE}")
print("✅ Done. Each record includes: Country, Pillar, Keyword, URL.")
