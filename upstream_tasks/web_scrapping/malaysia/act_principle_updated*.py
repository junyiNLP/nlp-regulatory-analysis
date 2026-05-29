import requests
import csv
import os
import time

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "web_scrapping_results_Malaysia")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file path
output_file = os.path.join(OUTPUT_DIR, "data_results_malaysia_ACT_updated.csv")

# --- Write CSV header ---
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Start_Index",
        "Act_No",
        "Title_English_Malay",
        "PDF_URL"
    ])

# --- Pagination loop ---
for start in range(0, 1000, 100):  # 871 entries → ~9 pages
    url = f"https://lom.agc.gov.my/json-updated-2024.php?start={start}&length=100"
    print(f"🔍 Scraping UPDATED Acts batch starting at {start} ...")

    response = requests.get(url)
    try:
        data = response.json()
    except Exception as e:
        print(f"⚠️ Failed to parse JSON at start={start}: {e}")
        continue

    records = data.get("records", [])
    print(f"✅ Found {len(records)} records in this batch")

    if not records:
        break

    # --- Append results ---
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for item in records:
            act_no = (item.get("noAct") or "").strip()
            title = (item.get("titleBI") or "").replace("\n", " ").strip()

            pdf_html = item.get("DOC2DOWNLOAD", "")
            pdf_url = ""
            if "href=" in pdf_html:
                start_idx = pdf_html.find('href="') + 6
                end_idx = pdf_html.find('"', start_idx)
                pdf_url = pdf_html[start_idx:end_idx]
                if not pdf_url.startswith("http"):
                    pdf_url = "https://lom.agc.gov.my/" + pdf_url.lstrip("./")

            writer.writerow([start, act_no, title, pdf_url])

    time.sleep(1)

print(f"🎉 All UPDATED Acts data saved to: {output_file}")
