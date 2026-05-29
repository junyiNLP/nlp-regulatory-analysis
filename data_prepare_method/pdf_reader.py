import fitz  # PyMuPDF
import json
import pathlib
import os

# === PATH CONFIG ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # project root
RAW_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "raw" / "to_process" /"digital_pdfs"
OUTPUT_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "interim" / "to_process"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_pdf_text(input_path: pathlib.Path):
    """Extract digital PDF text (no OCR) and save to JSON."""

    doc_id = input_path.stem
    output_file = OUTPUT_DIR / f"{doc_id}.json"

    pdf = fitz.open(input_path)
    pages_data = []

    for i, page in enumerate(pdf, start=1):
        # Direct text extraction — no OCR
        text = page.get_text("text")

        pages_data.append({
            "page": i,
            "text": text
        })

    result = {
        "doc_id": doc_id,
        "language": "en",       # or auto-detect later
        "pages": pages_data
    }

    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Extracted text → {output_file} ({len(pages_data)} pages)")

def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}")
    for pdf_file in pdf_files:
        extract_pdf_text(pdf_file)

if __name__ == "__main__":
    main()
