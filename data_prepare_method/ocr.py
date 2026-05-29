import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io, json, pathlib, os

# Ensure pytesseract points to correct binary (for macOS with Homebrew)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# === PATH CONFIG ===
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # go up to project root
RAW_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "raw" / "to_process" / "scanned_docs"
OUTPUT_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "interim" / "to_process"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def ocr_pdf(input_path: pathlib.Path):
    """OCR a single PDF file and save to JSON"""
    doc_id = input_path.stem  # filename without extension
    output_file = OUTPUT_DIR / f"{doc_id}.json"

    pdf = fitz.open(input_path)
    pages_data = []

    for i, page in enumerate(pdf, start=1):
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng", config="--psm 6")

        pages_data.append({"page": i, "text": text})

    result = {
        "doc_id": doc_id,
        "language": "en",
        "pages": pages_data
    }

    output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OCR complete → {output_file} ({len(pages_data)} pages)")

def main():
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}")
    for pdf_file in pdf_files:
        ocr_pdf(pdf_file)

if __name__ == "__main__":
    main()
