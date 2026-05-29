import json
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

## ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "stage1_training_3_single_label" / "model_training_3"
CHUNK_DIR = BASE_DIR / "dataset" / "pre_process_new_data" / "chunked" / "to_process"
OUT_DIR = BASE_DIR / "test_model" / " test_output" / "model_3"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== LOAD MODEL =====
print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
model.eval()

# Load label mapping (stored in config.json)
config_path = MODEL_DIR / "config.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

id2label = {int(k): v for k, v in config_data["id2label"].items()}

# Detect device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"🔹 Using device: {device}")
model.to(device)

# ===== LOAD CHUNKS =====
chunks = []
with open(CHUNK_FILE, "r", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"🔹 Loaded {len(chunks)} chunks from file")

# ===== PREDICT FUNCTION =====
def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    return pred_id, confidence

# ===== RUN PREDICTIONS =====
summary_rows = []

for c in chunks:
    pred_id, conf = predict(c["text"])
    label_name = id2label[pred_id]

    preview = c["text"][:200].replace("\n", " ") + "..."

    summary_rows.append({
        "doc_id": c["doc_id"].replace("_", " "),
        "chunk_id": c["chunk_id"],
        "predicted_label": label_name.replace(".", "_"),
        "confidence": round(conf, 2),
        "preview": preview
    })

# ===== SAVE =====
df_pretty = pd.DataFrame(summary_rows)
df_pretty.to_csv(OUT_FILE, index=False)

print("✅ CSV saved:", OUT_FILE)
