"""
==========================================================
 Malaysia Law PDF Downloader
==========================================================

Purpose:
----------------------------------------------------------
This script automatically downloads PDF files listed in
web-scraped CSV datasets (from the Malaysian law website).
Each dataset (Acts, P.U.(A), P.U.(B), etc.) will be saved into
its own subfolder for clear organization.
----------------------------------------------------------
"""

import os
import pandas as pd
import requests
from tqdm import tqdm  # for progress bar

# === [1] PATH CONFIGURATION ===
# Change only BASE_DIR if the project is moved.
BASE_DIR = "/Users/junyi/Downloads/Programming/Python/rdtii"

# Input CSV folder (contains all scraped datasets), change the path to any dataset we want to download from
CSV_DIR = os.path.join(BASE_DIR, "web_scrapping", "malaysia", "web_scrapping_results_Malaysia")

# Output folder where PDFs will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, "Method_downloading_automation", "pdf_files_malaysia")

# List of datasets to_process
CSV_FILES = [
    "data_results_malaysia_AmendmentAct.csv",
    "data_results_malaysia_PrincipalAct_original.csv",
    "data_results_malaysia_PUa.csv",
    "data_results_malaysia_PUb.csv",
]



# === [2] CUSTOM FILTER FUNCTION ===
def filter_dataframe(df):
    """
    Define your conditions here to control which PDFs will be downloaded.
    You can add or combine multiple conditions as needed.

    Examples:
    ----------------------------------------------------------
    # 1. Download only specific manually-entered Titles:
    specific_titles = [
        "PATENTS ACT 1983",
        "PATENTS (AMENDMENT) REGULATIONS 2022",
        "PATENTS (WAIVER OF FEE) REGULATIONS 2020"
    ]
    df = df[df["Title"].str.strip().isin(specific_titles)]
    print(f"✅ Filter applied: {len(df)} matches for manually entered titles.")
    ----------------------------------------------------------
    # 2. Download only if Title contains specific pillar_dictionary:
    pillar_dictionary = ["CONSTITUTION", "FOUNDATION", "EMPLOYEES"]
    df = df[df["Title"].str.contains("|".join(pillar_dictionary), case=False, na=False)]
    ----------------------------------------------------------
    # 3. Download only specific Titles from external list:
    titles_path = "to input the dataset path"
    title_list = pd.read_csv(titles_path)["title"].str.strip().tolist()
    df = df[df["Title"].str.strip().isin(title_list)]
    print(f"✅ Filter applied: {len(df)} titles matched from {titles_path}")
    ----------------------------------------------------------
    # 4. Download only specific Act Numbers:
    allowed_acts = ["Act 123", "Act 456"]
    df = df[df["Act_No"].isin(allowed_acts)]
    ----------------------------------------------------------
    # 5. Download only if Status == "Acts":
    df = df[df["Status"] == "Acts"]
    ----------------------------------------------------------
    # 6. Combine multiple filters:
    df = df[(df["Status"] == "Acts") & (df["Title"].str.contains("AMENDMENT", na=False))]
    ----------------------------------------------------------
    # 7.Default: no filter (downloads all PDFs)
    ----------------------------------------------------------
    """

    # Download only specific manually-entered Titles:
    specific_titles = [
        "LABUAN BUSINESS ACTIVITY TAX ACT 1990",
        "INCOME TAX ACT 1967",
        "PETROLEUM (INCOME TAX) ACT 1967",
        "CUSTOMS ACT 1967",
        "SALES TAX ACT 1972",
        "SERVICE TAX ACT 1975",
        "SUPPLEMENTARY INCOME TAX ACT 1967",
        "REAL PROPERTY GAINS TAX ACT 1976",
        "SALES TAX (AMENDMENT) ACT 2018",
        "GAMING TAX ACT 1972"
    ]

    # --- Auto-detect the correct title column ---
    possible_title_cols = ["Title", "Title_EN", "title"]
    title_col = next((col for col in possible_title_cols if col in df.columns), None)

    if not title_col:
        raise KeyError(f"No title-like column found in dataset. Columns: {list(df.columns)}")

    # --- Apply filter using the detected column ---
    df = df[df[title_col].astype(str).str.strip().isin(specific_titles)]

    print(f"✅ Filter applied using column '{title_col}': {len(df)} matches found.")

    # --- Return filtered DataFrame ---
    return df


# === [3] UTILITY FUNCTION: SAFE FILENAMES ===
def safe_filename(name: str) -> str:
    """Remove or replace invalid characters for filenames."""
    return str(name).strip().replace("/", "_").replace("\\", "_").replace(":", "_")


# === [4] CORE DOWNLOADER ===
def download_pdfs(csv_path: str, output_dir: str, name_field: str = "Title"):
    """
    Read a CSV file, apply filters, and download the PDFs.
    Args:
        csv_path: path to the input CSV file
        output_dir: folder where PDFs will be saved
        name_field: column used for naming files (e.g., Title or Act_No)
    """
    print(f"\n📂 Processing {os.path.basename(csv_path)} ...")
    df = pd.read_csv(csv_path)

    if "PDF_URL" not in df.columns:
        print("⚠️  Skipped: Missing 'PDF_URL' column.")
        return

    # Remove rows without URLs and apply filters
    df = df.dropna(subset=["PDF_URL"])
    df = filter_dataframe(df)
    os.makedirs(output_dir, exist_ok=True)

    failed = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"📥 Downloading {os.path.basename(csv_path)}"):
        pdf_url = str(row["PDF_URL"]).strip()
        if not pdf_url.startswith("http"):
            continue

        base_name = row.get(name_field, f"Document_{i}")
        filename = f"{safe_filename(base_name)[:120]}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            continue

        try:
            response = requests.get(pdf_url, timeout=25)
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
        except Exception as e:
            failed.append((pdf_url, str(e)))

    # Log download results
    if failed:
        log_file = os.path.join(output_dir, "failed_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            for url, err in failed:
                f.write(f"{url} - {err}\n")
        print(f"⚠️  {len(failed)} failed downloads. Log saved to: {log_file}")
    else:
        print(f"✅ Completed: {len(df)} PDFs downloaded successfully.")


# === [5] MAIN LOOP ===
def main():
    """
    Loop through all listed CSV files and download filtered PDFs.
    To add or remove datasets, edit the CSV_FILES list above.
    """
    for csv_file in CSV_FILES:
        csv_path = os.path.join(CSV_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"❌ Missing file: {csv_path}")
            continue

        subfolder = os.path.splitext(csv_file)[0]
        output_subdir = os.path.join(OUTPUT_DIR, subfolder)
        download_pdfs(csv_path, output_subdir)


# === [6] ENTRY POINT ===
if __name__ == "__main__":
    main()
