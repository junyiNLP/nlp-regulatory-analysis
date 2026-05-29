import json
import re
import unicodedata
from pathlib import Path

# ============================================================
# BASE DIR (THIS IS THE FIX)
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[2]

INTERIM_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "interim" / "to_process"
CLEAN_DIR   = BASE_DIR / "dataset" / "pre_process_new_data" / "cleaned" / "txt_files"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLEANING FUNCTION (UNCHANGED)
# ============================================================
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    text = (
        text.replace("“", "\"")
            .replace("”", "\"")
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", "-")
    )

    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"(?m)^(\d+)\s+([A-Z][^\n]{3,})", r"\n\1 \2\n", text)
    text = re.sub(r"(?m)^\((\d+)\)", r"\n(\1) ", text)
    text = re.sub(r"(?m)^\(([a-z])\)", r"\n(\1) ", text)

    return text.strip()

# ============================================================
# MAIN
# ============================================================
def main():
    files = list(INTERIM_DIR.glob("*.json"))
    print(f"Found {len(files)} JSON files")

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []

        for page in data["pages"]:
            page_no = page.get("page")
            cleaned = clean_text(page.get("text", ""))

            # ✅ Keep page number for human checking

            lines.append(f"\n==================================== PAGE {page_no} =============================================\n")
            lines.append(cleaned)

        final_text = "\n\n".join(lines)

        out_path = CLEAN_DIR / f"{data['doc_id']}.txt"
        out_path.write_text(final_text, encoding="utf-8")

        print(f"TXT written → {out_path.name}")

if __name__ == "__main__":
    main()
