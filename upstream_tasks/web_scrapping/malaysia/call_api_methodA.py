import requests
import csv
import os
import time

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "web_scrapping_results_Malaysia")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file path
output_file = os.path.join(OUTPUT_DIR, "data_results_malaysia_PUa.csv")

# --- Write CSV header ---
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Start_Index",
        "Publication_Date",
        "PU_No",
        "Title",
        "Status",
        "Date_of_Commencement",
        "PDF_URL"
    ])

# --- Pagination loop ---
# Each batch = 100 records → 6535 total means ~66 pages
for start in range(0, 6600, 100):
    url = f"https://lom.agc.gov.my/json-subsid-2024.php?type=pua&start={start}&length=100"
    print(f"🔍 Scraping batch starting at {start} ...")

    response = requests.get(url)
    try:
        data = response.json()
    except Exception as e:
        print(f"⚠️ Failed to parse JSON at start={start}: {e}")
        continue

    records = data.get("records", [])
    print(f"✅ Found {len(records)} records in this batch")

    if not records:
        break  # stop early if nothing is returned

    # --- Append results ---
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for item in records:
            pub_date = (item.get("publicationDate") or "").strip()
            pu_no = (item.get("noPU") or "").strip()
            title = (item.get("titleBI") or "").replace("\n", " ").strip()
            status = (item.get("statusOfLegislation") or "").strip()
            date_comm = (item.get("commencementDate") or "").strip()

            # Extract PDF URL from HTML tag
            pdf_html = item.get("DOC2DOWNLOAD", "")
            pdf_url = ""
            if "href=" in pdf_html:
                start_idx = pdf_html.find('href="') + 6
                end_idx = pdf_html.find('"', start_idx)
                pdf_url = pdf_html[start_idx:end_idx]
                if not pdf_url.startswith("http"):
                    pdf_url = "https://lom.agc.gov.my/" + pdf_url.lstrip("./")

            writer.writerow([start, pub_date, pu_no, title, status, date_comm, pdf_url])

    time.sleep(1)  # polite pause

print(f"🎉 All data saved to: {output_file}")
