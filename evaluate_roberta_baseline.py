"""
evaluate_roberta_baseline.py
Task3: Stress-test fine-tuned RoBERTa models across all Phase-1 attack suites.

Long-text Strategy (same as data_engineer.ipynb):
  - MAX_LENGTH    = 512 tokens
  - CHUNK_OVERLAP = 50 tokens
  - chunk_size    = MAX_LENGTH - 2
  - stride        = chunk_size - CHUNK_OVERLAP
  - Inference: average logits over all chunks, argmax → final prediction.
  Run via: uv run python evaluate_roberta_baseline.py
"""

import os
import time
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================== CONFIG ========================
MAX_LENGTH    = 512
CHUNK_OVERLAP = 50        # same as data_engineer.ipynb
EVAL_BATCH    = 16        # inference batch size (can be larger than train)
# =========================================================


# ------------------------------------------------------------------ #
#  Long-text chunking (mirrors data_engineer.ipynb)                  #
# ------------------------------------------------------------------ #
def chunk_text_by_tokens(text: str, tokenizer,
                          max_length: int = MAX_LENGTH,
                          overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Return a list of tokeniser output dicts for one document.
    Short texts → single dict.  Long texts → overlapping chunks.
    """
    chunk_size = max_length - 2   # reserve [CLS] + [SEP]
    stride     = chunk_size - overlap

    raw_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        truncation=False,
    )

    if len(raw_ids) <= chunk_size:
        return [tokenizer(str(text), truncation=True, max_length=max_length)]

    chunks = []
    for start in range(0, len(raw_ids), stride):
        chunk_ids  = raw_ids[start: start + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(tokenizer(chunk_text, truncation=True, max_length=max_length))
        if start + chunk_size >= len(raw_ids):
            break

    return chunks if chunks else [
        tokenizer(str(text), truncation=True, max_length=max_length)
    ]


# ------------------------------------------------------------------ #
#  Single-text inference with chunk-pool                             #
# ------------------------------------------------------------------ #
@torch.no_grad()
def predict_one(text: str, model, tokenizer, device) -> int:
    """
    Predict label for one (potentially long) text.
    Averages logits across all chunks then returns argmax.
    """
    chunks      = chunk_text_by_tokens(text, tokenizer)
    all_logits  = []

    for chunk in chunks:
        inputs  = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in chunk.items()}
        outputs = model(**inputs)
        all_logits.append(outputs.logits.squeeze(0).cpu().numpy())

    avg_logits = np.mean(all_logits, axis=0)   # average over chunks
    return int(np.argmax(avg_logits))


# ------------------------------------------------------------------ #
#  Batch evaluation over a DataFrame                                 #
# ------------------------------------------------------------------ #
def evaluate_dataset(model, tokenizer, df, device,
                     text_col: str = "Text",
                     label_col: str = "label") -> dict:
    df = df.dropna(subset=[text_col, label_col]).reset_index(drop=True)
    texts  = df[text_col].tolist()
    labels = df[label_col].astype(int).tolist()

    model.eval()
    all_preds  = []
    start_time = time.time()

    for text in texts:
        all_preds.append(predict_one(text, model, tokenizer, device))

    elapsed_s       = time.time() - start_time
    latency_ms      = (elapsed_s / max(len(texts), 1)) * 1000

    acc            = accuracy_score(labels, all_preds)
    _, _, f1,  _   = precision_recall_fscore_support(labels, all_preds, average="macro",    zero_division=0)
    _, _, f1w, _   = precision_recall_fscore_support(labels, all_preds, average="weighted", zero_division=0)

    return {
        "count":                len(texts),
        "accuracy":             round(acc,  4),
        "macro_f1":             round(f1,   4),
        "weighted_f1":          round(f1w,  4),
        "latency_ms_per_sample": round(latency_ms, 2),
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Evaluation device: {device}")

    results = []

    # ── Binary evaluation datasets ──────────────────────────────────
    binary_datasets = {
        "Clean_Test":        "results/models/RoBERTa_Binary/test_split.csv",
        "Misspell_5%":       "data/processed/processed_misspelled_5.csv",
        "Misspell_10%":      "data/processed/processed_misspelled_10.csv",
        "Misspell_15%":      "data/processed/processed_misspelled_15.csv",
        "Misspell_20%":      "data/processed/processed_misspelled_20.csv",
        "CrossDomain_Essay": "data/processed/processed_cross_domain_essay.csv",
        "CrossDomain_WP":    "data/processed/processed_cross_domain_wp.csv",
        "Unseen_Reuters":    "data/processed/processed_unseen_reuters.csv",
        "Paraphrased_Gemma": "data/processed/processed_paraphrased_articles.csv",
        "Translation_NLLB":  "data/processed/processed_translation.csv",
    }

    binary_dir = "results/models/RoBERTa_Binary"
    if os.path.isdir(binary_dir):
        print("\n==========================================")
        print(" Evaluating BINARY RoBERTa")
        print("==========================================")
        tok_bin   = AutoTokenizer.from_pretrained(binary_dir)
        model_bin = AutoModelForSequenceClassification.from_pretrained(binary_dir).to(device)

        for name, path in binary_datasets.items():
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: file not found → {path}")
                continue

            df = pd.read_csv(path)
            # Ensure label column exists (binary: is_AI → 0/1)
            if "label" not in df.columns:
                if "is_AI" in df.columns:
                    df["label"] = df["is_AI"].astype(int)
                else:
                    print(f"  [SKIP] {name}: no label/is_AI column")
                    continue

            print(f"  → {name} ({len(df)} rows) …")
            metrics = evaluate_dataset(model_bin, tok_bin, df, device)
            metrics.update({"Model": "RoBERTa_Binary", "Dataset": name})
            results.append(metrics)
            print(f"     Acc={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}")
    else:
        print(f"[WARN] Binary model dir not found: {binary_dir}")

    # ── Multi-class evaluation datasets ────────────────────────────
    multi_dir = "results/models/RoBERTa_Multi"
    if os.path.isdir(multi_dir):
        print("\n==============================================")
        print(" Evaluating MULTI-CLASS RoBERTa")
        print("==============================================")
        tok_multi   = AutoTokenizer.from_pretrained(multi_dir)
        model_multi = AutoModelForSequenceClassification.from_pretrained(multi_dir).to(device)

        with open(os.path.join(multi_dir, "label_mapping.json")) as f:
            label2id = json.load(f)["label2id"]

        multi_datasets = {
            "Clean_Test_Multi":       os.path.join(multi_dir, "test_split.csv"),
            "Paraphrased_Gemma_Multi":"data/processed/processed_paraphrased_articles.csv",
            "Translation_NLLB_Multi": "data/processed/processed_translation.csv",
        }

        for name, path in multi_datasets.items():
            if not os.path.exists(path):
                print(f"  [SKIP] {name}: file not found → {path}")
                continue

            df = pd.read_csv(path)
            if "Writer" not in df.columns and "label" not in df.columns:
                print(f"  [SKIP] {name}: no Writer/label column")
                continue

            if "label" not in df.columns:
                df = df[df["Writer"].isin(label2id.keys())].copy()
                df["label"] = df["Writer"].map(label2id)

            df = df.dropna(subset=["label"]).reset_index(drop=True)
            df["label"] = df["label"].astype(int)

            print(f"  → {name} ({len(df)} rows) …")
            metrics = evaluate_dataset(model_multi, tok_multi, df, device)
            metrics.update({"Model": "RoBERTa_Multi", "Dataset": name})
            results.append(metrics)
            print(f"     Acc={metrics['accuracy']:.4f}  MacroF1={metrics['macro_f1']:.4f}")
    else:
        print(f"[WARN] Multi model dir not found: {multi_dir}")

    # ── Save results ────────────────────────────────────────────────
    if results:
        cols    = ["Model", "Dataset", "count", "accuracy", "macro_f1", "weighted_f1", "latency_ms_per_sample"]
        res_df  = pd.DataFrame(results)[cols]
        out_csv = "results/tables/T15_Neural_vs_Hybrid_XGBoost.csv"
        os.makedirs("results/tables", exist_ok=True)
        res_df.to_csv(out_csv, index=False)
        print(f"\nSUCCESS: Results saved → {out_csv}")
        print("\n" + res_df.to_string(index=False))
    else:
        print("[WARN] No results collected – have the models been trained?")


if __name__ == "__main__":
    main()
