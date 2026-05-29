import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
import json

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "unified_ds_full.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_training_4_multi")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EPOCHS = 4
BATCH_SIZE = 2
LR = 2e-5
SEED = 42
MAX_LEN = 512
STRIDE = 256


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)

# Normalize label string
df["label"] = df["label"].astype(str)
df["label"] = df["label"].str.replace(",", "_").str.replace(".", "_")

# Drop empty text rows
df = df[df["text"].notna()]
df = df[df["text"].astype(str).str.strip() != ""]
df["combined_text"] = df["text"].astype(str)


# ============================================================
# MULTI-LABEL ENCODING
# ============================================================
df["label_list"] = df["label"].apply(lambda x: [x])

mlb = MultiLabelBinarizer()
label_matrix = mlb.fit_transform(df["label_list"])

NUM_LABELS = label_matrix.shape[1]
LABELS = mlb.classes_

print("🔹 Total labels:", NUM_LABELS)
print("🔹 Example:", LABELS[:10])

df["label_ids"] = label_matrix.tolist()


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))


# ============================================================
# TOKENIZER + SLIDING WINDOWS
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def sliding_tokenize(batch):
    outputs = tokenizer(
        batch["combined_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_special_tokens_mask=False,
    )

    sample_map = outputs.pop("overflow_to_sample_mapping")

    labels = []
    for i in range(len(sample_map)):
        # convert label list → float list (required for BCE loss)
        labels.append([float(x) for x in batch["label_ids"][sample_map[i]]])

    outputs["labels"] = labels
    return outputs


train_ds = train_ds.map(
    sliding_tokenize, batched=True, remove_columns=train_ds.column_names
)
test_ds = test_ds.map(
    sliding_tokenize, batched=True, remove_columns=test_ds.column_names
)

train_ds.set_format("torch")
test_ds.set_format("torch")


# ============================================================
# MODEL — MULTI-LABEL SIGMOID
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"
)


# ============================================================
# TRAINING ARGUMENTS  🔧 FIXED HERE
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    # use the canonical name; same as eval_strategy="epoch"
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    save_total_limit=2,

    # IMPORTANT FIXES FOR MPS + BCE:
    fp16=False,                 # turn off mixed precision (was causing NaNs)
    bf16=False,                 # make sure bf16 is off too
    gradient_accumulation_steps=1,  # simpler & more stable
    max_grad_norm=1.0,          # clip gradients to avoid explosions

    logging_steps=50,
    report_to="none",
)


# ============================================================
# METRICS (F1 with default 0.5 threshold)
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    from sklearn.metrics import f1_score
    f1_micro = f1_score(labels, preds, average="micro")
    f1_macro = f1_score(labels, preds, average="macro")
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}


# ============================================================
# TRAINER
# ============================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)


# ============================================================
# TRAIN
# ============================================================
print(" Starting MULTI-LABEL RDTII training...")
trainer.train()


# ============================================================
# SAVE MODEL + LABELS
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

with open(os.path.join(OUTPUT_DIR, "label_classes.json"), "w") as f:
    json.dump(list(LABELS), f, indent=2)

print(" Training complete!")
print(" Model saved to:", OUTPUT_DIR)
