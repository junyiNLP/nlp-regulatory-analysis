import json
import re
import unicodedata
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
INTERIM_DIR = Path("../dataset/pre_process_new_data/interim/to_process")
CLEAN_DIR = Path("../dataset/pre_process_new_data/cleaned/to_process")

CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CLEANING FUNCTION
# ============================================================
def clean_text(text: str) -> str:
    """
    Light, structure-preserving cleaning.
    This function prepares text for STRUCTURAL CHUNKING,
    not for model inference.
    """

    # --------------------------------------------------
    # 1. Unicode normalization (safe)
    # --------------------------------------------------
    text = unicodedata.normalize("NFKC", text)

    # --------------------------------------------------
    # 2. Normalize quotes and dashes
    # --------------------------------------------------
    text = (
        text.replace("“", "\"")
            .replace("”", "\"")
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", "-")
    )

    # --------------------------------------------------
    # 3. Normalize line endings (mostly redundant for PyMuPDF)
    # --------------------------------------------------
    text = text.replace("\r\n", "\n")
    # --------------------------------------------------
    # 4. Remove obvious headers / footers (line-based, non-greedy)
    #    Page numbers are NOT deleted globally — chunker can keep metadata
    # --------------------------------------------------
    text = re.sub(r"(?m)^[ \t]*APPENDIX[-–]?\w*[ \t]*$", "", text)
    text = re.sub(r"(?m)^[ \t]*Amended upto.*$", "", text)

    # --------------------------------------------------
    # 5. Whitespace normalization (STRUCTURE-SAFE)
    # --------------------------------------------------
    # Collapse spaces and tabs ONLY
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines, but keep paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # --------------------------------------------------
    # 6. Make legal numbering easier to detect (NO interpretation)
    # --------------------------------------------------
    # Ensure article / regulation numbers start on new lines
    text = re.sub(
        r"(?m)^(\d+)\s+([A-Z][^\n]{3,})",
        r"\n\1 \2\n",
        text
    )

    # Ensure numbered subclauses (1), (2) start on new lines
    text = re.sub(
        r"(?m)^\((\d+)\)",
        r"\n(\1) ",
        text
    )

    # Ensure lettered clauses (a), (b) start on new lines
    text = re.sub(
        r"(?m)^\(([a-z])\)",
        r"\n(\1) ",
        text
    )

    return text.strip()


# ============================================================
# MAIN
# ============================================================
def main():
    for file in INTERIM_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        cleaned_pages = []

        for page in data["pages"]:
            page_text = page.get("text", "")
            cleaned_text = clean_text(page_text)

            cleaned_pages.append({
                "page": page.get("page"),
                "text": cleaned_text
            })

        output = {
            "doc_id": data.get("doc_id"),
            "language": data.get("language", "en"),
            "pages": cleaned_pages
        }

        out_path = CLEAN_DIR / f"{data['doc_id']}.json"
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(f"Cleaned → {out_path.name}")


if __name__ == "__main__":
    main()
