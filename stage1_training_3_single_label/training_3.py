import os
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "0. unified_semantic_dataset_extended.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_training_3")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
EPOCHS = 10
BATCH_SIZE = 4
LR = 2e-5
SEED = 42
MAX_LEN = 512   # needed for legal act text


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)

# Safety normalization for labels
df["label"] = df["label"].astype(str)
df["label"] = df["label"].str.replace(",", "_")
df["label"] = df["label"].str.replace(".", "_")

# Drop empty or NaN text
df = df[df["text"].notna()]
df = df[df["text"].astype(str).str.strip() != ""]

# Use ONLY text (no source)
df["combined_text"] = df["text"].astype(str)


# ============================================================
# LABEL MAPPING
# ============================================================
labels = sorted(df["label"].unique())

label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

print("🔹 Total unique labels:", len(labels))
print("🔹 Example label mapping:", list(label2id.items())[:10])


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label_id"], random_state=SEED
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))


# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["combined_text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label_id", "labels")
test_ds = test_ds.rename_column("label_id", "labels")

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# ============================================================
# CLASS WEIGHTS (fixed version)
# ============================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array(list(range(len(labels)))),
    y=df["label_id"].values,
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)


# ============================================================
# MODEL
# ============================================================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

# Weighted Loss Function
def custom_loss_fn(outputs, labels):
    loss_fct = torch.nn.CrossEntropyLoss(
        weight=class_weights_tensor.to(outputs.logits.device)
    )
    return loss_fct(outputs.logits, labels)


# Custom Trainer with weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Accept **kwargs to avoid HF Trainer crash
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = custom_loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss


# ============================================================
# METRICS
# ============================================================
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    logging_steps=50,
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    report_to="none",
    save_total_limit=2,
)


# ============================================================
# TRAINER
# ============================================================
trainer = WeightedTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)


# ============================================================
# TRAIN
# ============================================================
print(" Training Final Semantic RDTII Model...")
trainer.train()

# ============================================================
# SAVE
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(" Model saved to:", OUTPUT_DIR)
print(" Final label2id mapping saved:", label2id)
