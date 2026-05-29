import requests
import csv
import os
import json
import time

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "web_scrapping_results_Malaysia")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file path
output_file = os.path.join(OUTPUT_DIR, "data_results_malaysia_AmendmentAct.csv")

# --- Write CSV header ---
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Start_Index",
        "Act_No",
        "Title_EN",
        "Title_BM",
        "Royal_Assent_Date",
        "Publication_Date",
        "Commencement_Date",
        "Commencement_Remark",
        "PDF_URL"
    ])

# --- Helper function to form absolute URLs ---
def to_abs(url_part: str) -> str:
    if not url_part:
        return ""
    return f"https://lom.agc.gov.my{url_part}".replace("\\/", "/")

# --- Pagination loop ---
# Each batch = 100 records → total 379 records (~4 pages)
for start in range(0, 400, 100):
    url = f"https://lom.agc.gov.my/json-amendment-2024.php?type=amendment&start={start}&length=100"
    print(f"🔍 Scraping batch starting at {start} ...")

    try:
        response = requests.get(url)
        data = response.json()
        records = data.get("records", [])
        print(f"✅ Found {len(records)} records")

        # --- Write each record ---
        with open(output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for i, record in enumerate(records):
                act_no = record.get("ACTNO_LEGISLATION", "").strip()
                title_bm = record.get("TajukBM", "").strip()
                title_en = record.get("TajukBI", "").strip()

                royal_assent = record.get("ROYALASSENTDATE", "").strip()
                publication = record.get("PUBLICATIONDATE", "").strip()
                commence_date = record.get("COMMENCEMENTDATEBI", "").strip()
                commence_remark = record.get("COMMENCEMENTREMARKBI", "").strip()

                # --- Extract PDF URL ---
                pdf_url = ""
                try:
                    pdf_info = record.get("DOC2DOWNLOADBIgeneratepdf")
                    if pdf_info:
                        pdf_json = json.loads(pdf_info)
                        path = pdf_json.get("path", "")
                        doc_name = pdf_json.get("docName", "")
                        if path and doc_name:
                            pdf_url = f"https://lom.agc.gov.my{path}{doc_name}".replace("\\/", "/")
                except Exception:
                    pass

                if not pdf_url:
                    print(f"⚠️ Missing PDF for {act_no}: {title_en[:60]}")

                writer.writerow([
                    start + i,
                    act_no,
                    title_en,
                    title_bm,
                    royal_assent,
                    publication,
                    commence_date,
                    commence_remark,
                    pdf_url
                ])

        # Respectful pause to avoid overwhelming the server
        time.sleep(1)

    except Exception as e:
        print(f"❌ Error at batch {start}: {e}")

print(f"🎉 All Amendment Act data saved to: {output_file}")
