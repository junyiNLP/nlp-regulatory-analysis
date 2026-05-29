import pandas as pd
import os

# --- Directory setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # this script is inside "malaysia/method_combine_dataset"
DATA_DIR = os.path.join(BASE_DIR, "../web_scrapping_results_Malaysia")  # go up one level to dataset folder
DATA_DIR = os.path.abspath(DATA_DIR)

# --- Input files ---
files = [
    "data_results_malaysia_AmendmentAct.csv",
    "data_results_malaysia_PrincipalAct_original.csv",
    "data_results_malaysia_PUa.csv",
    "data_results_malaysia_PUb.csv"
]

dfs = []

# --- Loop through files ---
for file in files:
    path = os.path.join(DATA_DIR, file)
    if not os.path.exists(path):
        print(f"⚠️ Skipping missing file: {file}")
        continue

    print(f"📂 Reading {file} ...")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]  # clean header spacing

    # --- Match structure by file type ---
    if "AmendmentAct" in file:
        df_small = df[["Act_No", "Title_EN", "PDF_URL"]].copy()
        df_small.rename(columns={"Title_EN": "Title"}, inplace=True)
        df_small["Status"] = "Acts"

    elif "PrincipalAct" in file:
        df_small = df[["Act_No", "Title", "PDF_URL"]].copy()
        df_small["Status"] = "Acts"

    elif "PUa" in file:
        df_small = df[["PU_No", "Title", "PDF_URL"]].copy()
        df_small.rename(columns={"PU_No": "Act_No"}, inplace=True)
        df_small["Status"] = "P.U.(A)"

    elif "PUb" in file:
        df_small = df[["PU_No", "Title", "PDF_URL"]].copy()
        df_small.rename(columns={"PU_No": "Act_No"}, inplace=True)
        df_small["Status"] = "P.U.(B)"

    else:
        print(f"⚠️ Unknown file format: {file}")
        continue

    dfs.append(df_small)

# --- Combine all ---
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.dropna(subset=["PDF_URL"], inplace=True)
    combined_df = combined_df[combined_df["PDF_URL"].astype(str).str.strip() != ""]

    # --- Output path (same folder as this script: "method_combine_dataset") ---
    output_path = os.path.join(BASE_DIR, "data_results_malaysia_AllActs_Combined.csv")
    combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n✅ Combined dataset saved to:\n{output_path}")
    print(f"📊 Total rows: {len(combined_df)}")
else:
    print("❌ No valid CSVs found.")
