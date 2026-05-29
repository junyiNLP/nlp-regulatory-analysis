import re
import pandas as pd
from pathlib import Path
import PyPDF2

# ============================================================
# CONFIG
# ============================================================
PDF_PATH = Path("0. RDTII_guide_pillar_session.pdf")
OUTPUT_CSV = Path("pillar5_guide_dataset.csv")

# Label mapping
indicator_labels = {
    "Lack of passive infrastructure sharing": "5_1",
    "Foreign equity limits in telecom sector": "5_2",
    "Shares owned by the Government": "5_3",
    "Lack of functional/accounting separation": "5_4",
    "Licensing requirements in telecom sector": "5_5",
    "Not in the WTO Telecom Reference Paper": "5_6",
    "Lack of independent telecom authority": "5_7"
}

# ============================================================
# 1. Read Entire PDF into Text
# ============================================================
pdf_text = ""
with open(PDF_PATH, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"

# Normalize spaces
pdf_text = re.sub(r"\s+\n", "\n", pdf_text)
pdf_text = pdf_text.replace("\u00a0", " ")  # remove non-breaking space


# ============================================================
# 2. Extract Pillar 5 description (first 2 paragraphs)
# ============================================================
pillar_intro_pattern = r"Pillar 5.*?services\)\.(.*?)\n\n(.*?)\n\n"
pillar_intro_match = re.search(pillar_intro_pattern, pdf_text, re.DOTALL)

if pillar_intro_match:
    paragraph1 = pillar_intro_match.group(1).strip()
    paragraph2 = pillar_intro_match.group(2).strip()
    pillar_5_text = paragraph1 + "\n\n" + paragraph2
else:
    raise ValueError("Could not locate Pillar 5 description in PDF.")


# ============================================================
# 3. Extract Each Indicator Section
# ============================================================
def extract_indicator(text, title):
    """
    Extract the full text under a given indicator heading up to the next indicator.
    """
    pattern = rf"{re.escape(title)}(.*?)(?=" + "|".join(map(re.escape, indicator_labels.keys())) + r"|$)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None

    section = match.group(1)

    # Remove scoring paragraphs
    section = re.sub(r"The score.*?(\n\n|$)", "", section, flags=re.DOTALL)
    section = re.sub(r"[Ff]unction\s+f\(x\).*?\n\n", "", section, flags=re.DOTALL)

    # Remove "Step 1 / Step 2", weighting, numeric criteria
    section = re.sub(r"Step \d.*?\n\n", "", section, flags=re.DOTALL)
    section = re.sub(r"[Ww]eights?.*?\n\n", "", section, flags=re.DOTALL)

    # Clean extra whitespace
    section = section.strip()

    return section


indicator_rows = []

for name, label in indicator_labels.items():
    extracted = extract_indicator(pdf_text, name)
    if extracted:
        indicator_rows.append((label, extracted, "pillar_guide_desc"))
    else:
        print(f"WARNING: Could not extract indicator: {name}")


# ============================================================
# 4. Build Final DataFrame
# ============================================================
rows = []

# Pillar-level row
rows.append(("5", pillar_5_text, "pillar_guide_desc"))

# Indicator-level rows
for row in indicator_rows:
    rows.append(row)

df = pd.DataFrame(rows, columns=["label", "text", "source"])


# ============================================================
# 5. Save CSV
# ============================================================
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("✅ Done! Saved to:", OUTPUT_CSV)
print(df.head(10))
