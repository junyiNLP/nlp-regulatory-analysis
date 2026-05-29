import os
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "unified_ds_full.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_training_5_single_label")

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

EPOCHS = 8
BATCH_SIZE = 4
LR = 2e-5
SEED = 42

MAX_LEN = 512
STRIDE = 256   # overlap size

set_seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)

df["label"] = df["label"].astype(str)
df["label"] = df["label"].str.replace(",", "_").str.replace(".", "_")

df = df[df["text"].notna()]
df = df[df["text"].astype(str).str.strip() != ""]

df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

df["combined_text"] = df["text"].astype(str)


# ============================================================
# LABEL ENCODING
# ============================================================
labels = sorted(df["label"].unique())
label2id = {lab: i for i, lab in enumerate(labels)}
id2label = {i: lab for lab, i in label2id.items()}

df["label_id"] = df["label"].map(label2id)

print("🔹 Total labels:", len(labels))
print("🔹 Total samples:", len(df))


# ============================================================
# STRATIFIED SPLIT
# ============================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_id"],
    random_state=SEED
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))


# ============================================================
# TOKENIZER WITH SLIDING WINDOW
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
    )

    sample_map = outputs.pop("overflow_to_sample_mapping")

    labels = []
    for i in range(len(sample_map)):
        original_idx = sample_map[i]
        labels.append(batch["label_id"][original_idx])

    outputs["labels"] = labels
    return outputs


train_ds = train_ds.map(
    sliding_tokenize,
    batched=True,
    remove_columns=train_ds.column_names
)

test_ds = test_ds.map(
    sliding_tokenize,
    batched=True,
    remove_columns=test_ds.column_names
)

train_ds.set_format("torch")
test_ds.set_format("torch")


# ============================================================
# CLASS WEIGHTS (computed on original train set)
# ============================================================
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(labels)),
    y=train_df["label_id"].values
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


# ============================================================
# CUSTOM TRAINER WITH WEIGHTED LOSS
# ============================================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(
            weight=class_weights_tensor.to(logits.device)
        )
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# METRICS
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

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
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    weight_decay=0.01,
    max_grad_norm=1.0,

    logging_steps=50,
    save_total_limit=2,
    report_to="none",
)


# ============================================================
# TRAINER
# ============================================================
trainer = WeightedTrainer(
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
print(" Starting Generation 5 Single-Label + Sliding Window Training...")
trainer.train()


# ============================================================
# SAVE
# ============================================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(" Model saved to:", OUTPUT_DIR)
print(" Label mapping size:", len(label2id))
