"""
==========================================================
 Chunk all cleaned text files into BERT-sized JSONL chunks
==========================================================

Input  : ../dataset/pre_process_new_data/cleaned/*.txt
Output : ../dataset/pre_process_new_data/chunked/{doc_id}.chunks.jsonl
Chunk  : 1000 characters per chunk, 150-character overlap
==========================================================
"""

import json
from pathlib import Path

# === PATH CONFIG ===
CLEAN_DIR = Path("../../dataset/pre_process_new_data/cleaned")
OUT_DIR = Path("../../dataset/pre_process_new_data/chunked")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === CHUNK FUNCTION ===
def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append((start, end, chunk))
        start += chunk_size - overlap
    return chunks

# === PROCESS ONE FILE ===
def process_file(file_path: Path):
    doc_id = file_path.stem  # filename without extension
    out_path = OUT_DIR / f"{doc_id}.chunks.jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, (start, end, chunk) in enumerate(chunks, 1):
            record = {
                "doc_id": doc_id,
                "chunk_id": f"{i:04d}",
                "start_char": start,
                "end_char": end,
                "text": chunk
            }
            f.write(json.dumps(record) + "\n")

    print(f"✅ {doc_id}: {len(chunks)} chunks saved → {out_path.name}")

# === MAIN LOOP ===
if __name__ == "__main__":
    txt_files = list(CLEAN_DIR.glob("*.txt"))
    print(f"🔹 Found {len(txt_files)} cleaned text files to process.\n")

    for file_path in txt_files:
        try:
            process_file(file_path)
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    print("\n🎉 All files processed successfully!")
