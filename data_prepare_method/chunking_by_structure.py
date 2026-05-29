import json
import re
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "cleaned" / "to_process"
CHUNK_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "chunked" / "to_process"

CHUNK_DIR.mkdir(parents=True, exist_ok=True)

# Section header pattern: "10 Title..."
SECTION_RE = re.compile(r"^(\d{1,3})\s+(.+)$")


def remove_running_header(lines):
    """
    Remove running header like 'Biosecurity'
    ONLY if it appears as the first short line on the page.
    """
    if not lines:
        return lines

    first = lines[0].strip()
    if 3 <= len(first) <= 40 and first.isalpha() and first[0].isupper():
        return lines[1:]

    return lines


def main():
    for file in sorted(CLEAN_DIR.glob("*.json")):
        print(f"Processing {file.name}...")

        data = json.loads(file.read_text(encoding="utf-8"))
        doc_id = data["doc_id"]

        # 🔑 RESET PER FILE
        all_chunks = []
        current_chunk = None

        for page in data["pages"]:
            pdf_page = page["page"]
            lines = page["text"].split("\n")

            # Remove running header only
            lines = remove_running_header(lines)

            for line in lines:
                line = line.rstrip()

                if not line:
                    if current_chunk:
                        current_chunk["text"] += "\n"
                    continue

                m = SECTION_RE.match(line)
                if m:
                    # Close previous section
                    if current_chunk:
                        current_chunk["text"] = current_chunk["text"].strip()
                        all_chunks.append(current_chunk)

                    current_chunk = {
                        "doc_id": doc_id,
                        "section": m.group(1),
                        "title": m.group(2).strip(),
                        "pdf_page_start": pdf_page,
                        "pdf_pages": {pdf_page},
                        "text": ""
                    }
                else:
                    if current_chunk:
                        current_chunk["text"] += line + "\n"
                        current_chunk["pdf_pages"].add(pdf_page)

        # Flush last section
        if current_chunk:
            current_chunk["text"] = current_chunk["text"].strip()
            all_chunks.append(current_chunk)

        # Convert page sets → sorted lists
        for c in all_chunks:
            c["pdf_pages"] = sorted(c["pdf_pages"])

        # --------------------------------------------------
        # FILTER: remove sections with < 10 words in TEXT
        # --------------------------------------------------
        filtered_chunks = []

        for chunk in all_chunks:
            if len(chunk["text"].strip().split()) < 10:
                continue  # ❌ drop short/empty sections

            filtered_chunks.append(chunk)

        all_chunks = filtered_chunks

        # 🔑 OUTPUT FILE = SAME NAME AS INPUT
        out_path = CHUNK_DIR / f"{file.stem}.clean.chunks.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        print(f"  → {out_path.name} ({len(all_chunks)} chunks)")

    print("✅ All documents processed.")


if __name__ == "__main__":
    main()
