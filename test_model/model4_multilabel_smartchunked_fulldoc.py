import json
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "stage1_training_4_multilabel" / "model_training_4_multi"
CHUNK_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "chunked" / "to_process"
OUT_DIR = BASE_DIR / "test_model" / " test_output" / "model_4"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MAX_LEN = 512
STRIDE = 256
TOP_K = 3

# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================
print("🔹 Loading multi-label model (Model-3)...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print("🔹 Using device:", device)
model.to(device)

# ============================================================
# LOAD LABEL CLASSES (AUTHORITATIVE)
# ============================================================
with open(MODEL_DIR / "label_classes.json", "r") as f:
    LABELS = json.load(f)

NUM_LABELS = len(LABELS)
print(f"🔹 Loaded {NUM_LABELS} indicators")

# ============================================================
# SLIDING-WINDOW MULTI-LABEL PREDICTION
# ============================================================
@torch.no_grad()
def predict_topk_multilabel(text: str, k: int = 3):
    """
    Long-text-safe multi-label prediction.
    Covers entire section using sliding windows.
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    ).logits

    # sigmoid for multi-label
    probs = torch.sigmoid(logits).cpu().numpy()

    # aggregate across windows (section-level)
    probs_agg = probs.mean(axis=0)

    # top-k indicators
    top_idx = np.argsort(probs_agg)[::-1][:k]

    return [
        (LABELS[i], float(probs_agg[i]))
        for i in top_idx
    ]

# ============================================================
# APPLY MODEL TO CHUNKED DATA
# ============================================================
for chunk_file in sorted(CHUNK_DIR.glob("*.jsonl")):
    print(f"\n🔮 Processing {chunk_file.name}")

    rows = []

    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)

            top3 = predict_topk_multilabel(c["text"], TOP_K)

            row = {
                "doc_id": c["doc_id"],
                "pdf_pages": ",".join(map(str, c.get("pdf_pages", []))),
                "section": c.get("section"),
                "title": c.get("title"),
                "text": c["text"],

                "prediction_1": top3[0][0],
                "score_1": round(top3[0][1], 4),
                "prediction_2": top3[1][0],
                "score_2": round(top3[1][1], 4),
                "prediction_3": top3[2][0],
                "score_3": round(top3[2][1], 4),
            }

            rows.append(row)

    out_path = OUT_DIR / f"{chunk_file.stem}_model4_top3.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"✅ Saved → {out_path.name}")

print("\n🎉 All chunked documents processed successfully.")