import requests
import csv
import os
import time
import re

# === 1. Directory setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "web_scrapping_results_Malaysia")
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file = os.path.join(OUTPUT_DIR, "data_results_malaysia_PrincipalAct_original.csv")
BASE = "https://lom.agc.gov.my"

# === 2. Helper functions ===
def to_abs(url_like: str) -> str:
    """Convert relative or escaped paths into valid absolute URLs."""
    if not url_like:
        return ""
    u = url_like.strip().replace("\\/", "/").replace("\\", "/")

    # Remove leading './' or '../'
    while u.startswith("../") or u.startswith("./"):
        u = u[3:] if u.startswith("../") else u[2:]

    # Convert to absolute
    if u.startswith("/"):
        return BASE + u
    if u.startswith("http"):
        return u
    return f"{BASE}/{u}"

def extract_hrefs(html: str):
    """Extract all href links from small HTML fragments."""
    if not html:
        return []
    s = html.replace("\\/", "/").replace("\\", "/")
    return re.findall(r'href=[\'"]([^\'"]+)[\'"]', s)

def get_pdf_url(item):
    """
    Extract the English (BI) PDF link for each Principal Act.
    Covers all JSON patterns found in 2024 dataset.
    """
    # 1️⃣ Structured field (newer format)
    obj = item.get("DOC2DOWNLOADBIgeneratepdf")
    if obj:
        if isinstance(obj, list) and len(obj) > 0:
            path = obj[0].get("path", "").strip()
            name = obj[0].get("docName", "").strip()
            if path and name:
                return to_abs(os.path.join(path, name))
        elif isinstance(obj, dict):
            p, n = obj.get("path", ""), obj.get("docName", "")
            if p and n:
                return to_abs(os.path.join(p, n))

    # 2️⃣ HTML-based field
    html = item.get("DOC2DOWNLOADBI", "") or item.get("DOC2DOWNLOAD", "")
    for h in extract_hrefs(html):
        return to_abs(h)

    # 3️⃣ Fallback to docNameBI
    doc = item.get("docNameBI") or item.get("DOC2DOWNLOADBIgeneratedpdf")
    if doc:
        return to_abs(doc)

    # 4️⃣ Final fallback: Malay version if English completely absent
    doc_bm = item.get("docNameBM") or item.get("DOC2DOWNLOADBM")
    if doc_bm:
        return to_abs(doc_bm)

    return ""

# === 3. Write CSV header ===
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Start_Index",
        "Act_No",
        "Title",
        "Date_of_Royal_Assent",
        "Publication_Date",
        "Date_of_Commencement",
        "PDF_URL"
    ])

# === 4. Main scraping loop ===
for start in range(0, 900, 100):
    url = f"{BASE}/json-principal-2024.php?type=original&start={start}&length=100"
    print(f"🔍 Scraping batch starting at {start} ...")

    try:
        data = requests.get(url, timeout=30).json()
    except Exception as e:
        print(f"⚠️ JSON error at start={start}: {e}")
        continue

    records = data.get("records", [])
    print(f"✅ Found {len(records)} records")
    if not records:
        break

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for it in records:
            act_no = (it.get("ACTNO_LEGISLATION") or "").strip()
            title_html = it.get("LEGISLATIONTITLEBI", "") or ""
            m = re.search(r">([^<]+)<", title_html.replace("\\/", "/"))
            title = (m.group(1).strip() if m else title_html.strip())

            royal = (it.get("ROYALASSENTDATE") or "").strip()
            pub = (it.get("PUBLICATIONDATE") or "").strip()
            comm = (it.get("COMMENCEMENTDATEBM") or "").strip()

            pdf_url = get_pdf_url(it)
            if not pdf_url:
                print(f"⚠️ Missing PDF for {act_no}: {title}")

            writer.writerow([start, act_no, title, royal, pub, comm, pdf_url])

    time.sleep(1)

print(f"🎉 All PDF links saved to: {output_file}")
