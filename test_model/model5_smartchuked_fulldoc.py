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

MODEL_DIR = BASE_DIR / "stage1_training_5_single_label" / "model_training_5_single_label"
CHUNK_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "chunked" / "to_process"
OUT_DIR = BASE_DIR / "test_model" / " test_output" / "model_5"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
MAX_LEN = 512
STRIDE = 256

# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================
print("🔹 Loading Model-3 (single-label)...")

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
# LOAD LABEL MAP (from config.json)
# ============================================================
with open(MODEL_DIR / "config.json", "r") as f:
    config = json.load(f)

id2label = {int(k): v for k, v in config["id2label"].items()}

# ============================================================
# SLIDING-WINDOW SINGLE-LABEL PREDICTION
# ============================================================
@torch.no_grad()
def predict_singlelabel(text: str):
    """
    Long-text-safe single-label prediction.
    Returns: (label_name, confidence)
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

    # softmax for single-label
    probs = torch.softmax(logits, dim=-1).cpu().numpy()

    # aggregate across windows
    probs_agg = probs.mean(axis=0)

    pred_id = int(np.argmax(probs_agg))
    confidence = float(probs_agg[pred_id])

    return id2label[pred_id], confidence

# ============================================================
# APPLY MODEL TO CHUNKED DATA
# ============================================================
for chunk_file in sorted(CHUNK_DIR.glob("*.jsonl")):
    print(f"\n🔮 Processing {chunk_file.name}")

    rows = []

    with open(chunk_file, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)

            label, conf = predict_singlelabel(c["text"])

            row = {
                "doc_id": c["doc_id"],
                "pdf_pages": ",".join(map(str, c.get("pdf_pages", []))),
                "section": c.get("section"),
                "title": c.get("title"),
                "text": c["text"],
                "predicted_label": label.replace(".", "_"),
                "confidence": round(conf, 4),
            }

            rows.append(row)

    out_path = OUT_DIR / f"{chunk_file.stem}_model5_singlelabel.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"✅ Saved → {out_path.name}")

print("\n ✅ Single-label inference complete.")
