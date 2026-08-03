"""
train_roberta_baseline.py
Task 3: Fine-Tune RoBERTa-base for Binary and Multi-Class AI text detection.

Long-text Strategy (same as data_engineer.ipynb):
  - MAX_LENGTH  = 512 tokens (RoBERTa hard limit)
  - CHUNK_OVERLAP = 50 tokens (matching old BERT pipeline)
  - chunk_size  = MAX_LENGTH - 2  (reserve [CLS] and [SEP])
  - stride      = chunk_size - CHUNK_OVERLAP
  - During TRAINING  → tokenize first chunk only (truncation) for speed.
  - During INFERENCE → average logits over all overlapping chunks (chunk-pool).
  Run via: uv run python train_roberta_baseline.py [--task binary|multi|all]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# ======================== CONFIG ========================
SEED        = 42
MAX_LENGTH  = 512
CHUNK_OVERLAP = 50        # same as data_engineer.ipynb
MODEL_NAME  = "roberta-base"
# =========================================================

set_seed(SEED)


# ------------------------------------------------------------------ #
#  Long-text chunking utility (mirrors data_engineer.ipynb logic)    #
# ------------------------------------------------------------------ #
def chunk_text_by_tokens(text: str, tokenizer, max_length: int = MAX_LENGTH,
                          overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split long text into overlapping token chunks.
    Returns a list of tokenizer output dicts (without padding so
    DataCollatorWithPadding can handle batches efficiently).
    """
    chunk_size = max_length - 2   # reserve [CLS] + [SEP]
    stride = chunk_size - overlap

    # Tokenise WITHOUT special tokens first to get raw token IDs
    raw_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        truncation=False,
    )

    # Fits in a single chunk → standard tokenisation
    if len(raw_ids) <= chunk_size:
        return [tokenizer(
            str(text),
            truncation=True,
            max_length=max_length,
        )]

    # Build overlapping chunks from raw token IDs
    chunks = []
    for start in range(0, len(raw_ids), stride):
        chunk_ids = raw_ids[start: start + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(tokenizer(
            chunk_text,
            truncation=True,
            max_length=max_length,
        ))
        if start + chunk_size >= len(raw_ids):
            break

    return chunks if chunks else [tokenizer(
        str(text),
        truncation=True,
        max_length=max_length,
    )]


# ------------------------------------------------------------------ #
#  PyTorch Dataset  (training only – uses first chunk / truncation)   #
# ------------------------------------------------------------------ #
class TextDataset(Dataset):
    """
    For TRAINING we truncate at max_length (first 512 tokens).
    This is standard practice and avoids inflating training time.
    All long-text handling for evaluation is done with chunk-pooling.
    """
    def __init__(self, texts, labels, tokenizer, max_length: int = MAX_LENGTH):
        self.texts      = list(texts)
        self.labels     = list(labels)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            max_length=self.max_length,
        )
        # Convert BatchEncoding values to tensors (required by DataCollatorWithPadding)
        item = {k: torch.tensor(v) for k, v in encoding.items()}
        item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------------ #
#  Metrics                                                            #
# ------------------------------------------------------------------ #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    _, _, f1,  _ = precision_recall_fscore_support(labels, preds, average="macro",    zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "f1": f1, "weighted_f1": f1w}


# ------------------------------------------------------------------ #
#  Data preparation                                                   #
# ------------------------------------------------------------------ #
def prepare_data(data_path: str, task: str = "binary"):
    print(f"--> Loading data: {data_path}  |  task={task}")
    df = pd.read_csv(data_path, usecols=["Text", "is_AI", "Writer"])
    df = df.dropna(subset=["Text"]).reset_index(drop=True)

    if task == "binary":
        df["label"] = df["is_AI"].astype(int)
        num_labels = 2
        label2id   = {"Human": 0, "AI": 1}
        id2label   = {0: "Human", 1: "AI"}
    else:
        unique_writers = sorted(df["Writer"].unique().tolist())
        label2id = {w: i for i, w in enumerate(unique_writers)}
        id2label = {i: w for i, w in enumerate(unique_writers)}
        df["label"] = df["Writer"].map(label2id)
        num_labels = len(unique_writers)

    stratify_col = df["label"]

    # 80 % train+val, 20 % test  (same ratio as Phase-1)
    train_val_df, test_df = train_test_split(
        df, test_size=0.20, random_state=SEED, stratify=stratify_col
    )
    # 10 % of total → val   (= 12.5 % of train_val)
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=SEED,
        stratify=train_val_df["label"]
    )

    print(f"Splits → Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df, num_labels, label2id, id2label


# ------------------------------------------------------------------ #
#  Main training function                                             #
# ------------------------------------------------------------------ #
def train_model(task: str = "binary", epochs: int = 3,
                batch_size: int = 8, max_length: int = MAX_LENGTH):

    data_path  = "data/processed/processed_articles.csv"
    output_dir = f"results/models/RoBERTa_{task.capitalize()}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    train_df, val_df, test_df, num_labels, label2id, id2label = \
        prepare_data(data_path, task=task)

    # Save test split and label map for reproducible evaluation
    test_df.to_csv(os.path.join(output_dir, "test_split.csv"), index=False)
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    ds_train = TextDataset(train_df["Text"], train_df["label"], tokenizer, max_length)
    ds_val   = TextDataset(val_df["Text"],   val_df["label"],   tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=2,      # effective batch = 16
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,                 # keep BEST checkpoint only
        logging_steps=100,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"\n==> Training {task.upper()} RoBERTa  "
          f"(batch={batch_size}, grad_accum=2, effective_batch=16, "
          f"epochs={epochs}, max_len={max_length})")
    trainer.train()

    # Save best model
    print(f"--> Saving BEST model to {output_dir} …")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SUCCESS: Best {task.capitalize()} RoBERTa saved at → {output_dir}\n")


# ------------------------------------------------------------------ #
#  Entry point                                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoBERTa Baseline Models")
    parser.add_argument("--task",       type=str, choices=["binary", "multi", "all"],
                        default="all",  help="binary | multi | all")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    if args.task in ["binary", "all"]:
        train_model("binary", args.epochs, args.batch_size, args.max_length)
    if args.task in ["multi", "all"]:
        train_model("multi",  args.epochs, args.batch_size, args.max_length)
