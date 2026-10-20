"""
Task 3 Remediation: Fix Data Leakage and Evaluate RoBERTa on Genuine NLLB Translated Test Split.

Steps:
1. Locate genuine NLLB-200 round-trip translation outputs in archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv.
2. Verify text differs from clean articles (assert TF-IDF cosine similarity < 1.0 and pairwise string mismatch).
3. Filter strictly for held-out test split (0% train/val overlap guaranteed).
4. Save clean evaluation dataset to results/phase3/T3_true_test_translated.csv.
5. Re-evaluate RoBERTa binary models (Seed 42 and all 5 seeds) on genuine translated test split.
6. Save results to results/phase3/T3_roberta_true_translation_metrics.csv.
"""

import os
import re
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLEAN_MAPPING = {
    'Human_story': 'Human',
    'Human_story_paraphrased': 'Human',
    'Human_story_translated': 'Human',
    'human': 'Human',
    'GPT_4-o': 'GPT-4o',
    'GPT_4-o_paraphrased': 'GPT-4o',
    'GPT_4-o_translated': 'GPT-4o',
    'gemma-2-9b': 'Gemma-2-9B',
    'gemma-2-9b_paraphrased': 'Gemma-2-9B',
    'gemma-2-9b_translated': 'Gemma-2-9B',
    'mistral-7B': 'Mistral-7B',
    'mistral-7B_paraphrased': 'Mistral-7B',
    'mistral-7B_translated': 'Mistral-7B',
    'llama-8B': 'Llama-8B',
    'llama-8B_paraphrased': 'Llama-8B',
    'llama-8B_translated': 'Llama-8B',
    'qwen-2-72B': 'Qwen-2-72B',
    'qwen-2-72B_paraphrased': 'Qwen-2-72B',
    'qwen-2-72B_translated': 'Qwen-2-72B',
    'accounts/yi-01-ai/models/yi-large': 'Yi-Large',
    'accounts/yi-01-ai/models/yi-large_paraphrased': 'Yi-Large',
    'yi-large_translated': 'Yi-Large',
}


def get_splits_for_seed(df_articles: pd.DataFrame, seed: int):
    """Reconstruct exact train/val/test splits used by RoBERTa training."""
    stratify_col = df_articles["is_AI"].astype(int)
    train_val_df, test_df = train_test_split(
        df_articles, test_size=0.20, random_state=seed, stratify=stratify_col
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=seed,
        stratify=train_val_df["is_AI"].astype(int)
    )
    return set(train_df.index), set(val_df.index), set(test_df.index)


def chunk_text_by_tokens(text: str, tokenizer, max_length: int = 512, overlap: int = 64):
    chunk_size = max_length - 2
    stride = chunk_size - overlap
    raw_ids = tokenizer.encode(str(text), add_special_tokens=False, truncation=False)
    if len(raw_ids) <= chunk_size:
        return [tokenizer(str(text), truncation=True, max_length=max_length, return_tensors="pt")]

    chunks = []
    for start in range(0, len(raw_ids), stride):
        chunk_ids = raw_ids[start: start + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(tokenizer(chunk_text, truncation=True, max_length=max_length, return_tensors="pt"))
        if start + chunk_size >= len(raw_ids):
            break
    return chunks


def evaluate_roberta_binary_on_df(model, tokenizer, df: pd.DataFrame, device):
    """Evaluates RoBERTa binary classification with token chunk pooling."""
    model.eval()
    all_preds = []
    all_probs = []
    labels = df["label"].values

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating RoBERTa"):
            chunks = chunk_text_by_tokens(row["Text"], tokenizer)
            chunk_logits = []
            for c in chunks:
                inputs = {k: v.to(device) for k, v in c.items()}
                out = model(**inputs)
                chunk_logits.append(out.logits.squeeze(0))
            # Average logits across chunks
            avg_logits = torch.stack(chunk_logits).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=-1)
            pred = torch.argmax(probs).item()
            all_preds.append(pred)
            all_probs.append(probs[1].item())

    all_preds = np.array(all_preds)
    acc = accuracy_score(labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, all_preds)
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "preds": all_preds,
        "probs": all_probs
    }


def main():
    print("=" * 80)
    print("TASK 3 REMEDIATION: FIX DATA LEAKAGE & EVALUATE ON TRUE TRANSLATED TEST SET")
    print("=" * 80)

    # 1. Locate genuine translated dataset
    raw_nllb_path = "archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv"
    if not os.path.exists(raw_nllb_path):
        raw_nllb_path = "archive/legacy/old_things/paraphrasing/nllb/DB_test_paraphrased_1.3B_ULTRA.csv"
    
    print(f"Loading genuine NLLB dataset from: {raw_nllb_path}")
    df_nllb = pd.read_csv(raw_nllb_path)
    print(f"  Rows loaded: {len(df_nllb)}")
    print(f"  Columns: {list(df_nllb.columns)}")

    # Standardize writer
    df_nllb["Writer"] = df_nllb["Writer"].map(CLEAN_MAPPING).fillna(df_nllb["Writer"])
    df_nllb["is_AI"] = df_nllb["is_AI"].astype(int)

    # 2. Verify text divergence (genuine translation check)
    print("\n--> Verifying genuine text divergence between original and translated...")
    original_texts = df_nllb["Text"].astype(str)
    translated_texts = df_nllb["paraphrased_text"].astype(str)

    exact_matches = (original_texts == translated_texts).sum()
    print(f"  Exact string matches: {exact_matches} / {len(df_nllb)} ({(exact_matches/len(df_nllb))*100:.2f}%)")
    assert exact_matches == 0, "Error: Some texts in paraphrased_text are identical to original!"

    # TF-IDF cosine similarity between original and translated pairs
    tfidf = TfidfVectorizer(max_features=10000, stop_words="english")
    tfidf.fit(pd.concat([original_texts, translated_texts]))
    orig_vecs = tfidf.transform(original_texts)
    trans_vecs = tfidf.transform(translated_texts)
    
    pairwise_sims = np.array([
        cosine_similarity(orig_vecs[i], trans_vecs[i])[0, 0] for i in range(min(500, len(df_nllb)))
    ])
    mean_sim = np.mean(pairwise_sims)
    print(f"  Mean TF-IDF Cosine Similarity (sample N=500): {mean_sim:.4f}")
    assert mean_sim < 0.99, f"Error: Translation similarity too high ({mean_sim:.4f}), text may not be translated!"
    print("  [SUCCESS] Verified genuine NLLB-200 round-trip translation!")

    # 3. Match against processed_articles.csv to identify original dataset indices
    print("\n--> Matching translated rows back to processed_articles.csv for exact split mapping...")
    articles_path = "data/processed/processed_articles.csv"
    df_articles = pd.read_csv(articles_path, usecols=["Text", "Writer", "is_AI"])
    
    # Frozen duplicate-text rule: last occurrence wins, matching the historical
    # NLLB evaluation archive and task3_leakage_audit.py.  Do not replace with a
    # first-occurrence setdefault; that changes the 667/580/610/615/588 Ns.
    text_to_idx = {}
    for idx, text in enumerate(df_articles["Text"]):
        text_to_idx[text] = idx
    
    matched_indices = []
    unmatched_count = 0
    for t in df_nllb["Text"]:
        if t in text_to_idx:
            matched_indices.append(text_to_idx[t])
        else:
            matched_indices.append(-1)
            unmatched_count += 1

    df_nllb["orig_idx"] = matched_indices
    matched_df = df_nllb[df_nllb["orig_idx"] >= 0].copy()
    print(f"  Successfully matched {len(matched_df)} / {len(df_nllb)} to processed_articles.csv indices")

    # 4. Filter strictly for Seed 42 Test Split (Zero Leakage)
    print("\n--> Partitioning strictly for held-out Test Splits (Zero Data Leakage)...")
    seeds = [42, 123, 999, 2024, 8080]
    seed_splits = {s: get_splits_for_seed(df_articles, s) for s in seeds}

    # For Seed 42
    s42_train, s42_val, s42_test = seed_splits[42]
    
    df_s42_test = matched_df[matched_df["orig_idx"].isin(s42_test)].copy()
    # Explicitly check for zero leakage
    leakage_train = df_s42_test["orig_idx"].isin(s42_train).sum()
    leakage_val = df_s42_test["orig_idx"].isin(s42_val).sum()
    assert leakage_train == 0 and leakage_val == 0, "FATAL: Train/Val leakage detected!"
    print(f"  [Zero Leakage Verified] Seed 42 genuine test translated samples: {len(df_s42_test)}")
    print(f"  Class balance: Human={sum(df_s42_test['is_AI'] == 0)}, AI={sum(df_s42_test['is_AI'] == 1)}")

    # Construct clean evaluation dataset
    clean_eval_df = pd.DataFrame({
        "Text": df_s42_test["paraphrased_text"],  # Use GENUINE translated text!
        "Writer": df_s42_test["Writer"],
        "is_AI": df_s42_test["is_AI"],
        "label": df_s42_test["is_AI"],
        "original_text": df_s42_test["Text"],
        "orig_idx": df_s42_test["orig_idx"]
    }).reset_index(drop=True)

    os.makedirs("results/phase3", exist_ok=True)
    clean_eval_path = "results/phase3/T3_true_test_translated.csv"
    clean_eval_df.to_csv(clean_eval_path, index=False)
    print(f"  Saved clean evaluation dataset to: {clean_eval_path}")

    # 5. Evaluate RoBERTa Binary Baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--> Evaluating RoBERTa Binary on Device: {device}")

    eval_results = []

    for s in seeds:
        model_dir = f"results/models/RoBERTa_Binary_s{s}"
        if not os.path.exists(model_dir):
            if s == 42 and os.path.exists("results/models/RoBERTa_Binary"):
                model_dir = "results/models/RoBERTa_Binary"
            else:
                print(f"  [SKIP] Model dir not found: {model_dir}")
                continue

        # Get the seed-specific strict test set
        s_train, s_val, s_test = seed_splits[s]
        seed_test_df = matched_df[matched_df["orig_idx"].isin(s_test)].copy()
        assert seed_test_df["orig_idx"].isin(s_train).sum() == 0, f"Leakage detected in seed {s}!"
        
        eval_seed_df = pd.DataFrame({
            "Text": seed_test_df["paraphrased_text"],
            "label": seed_test_df["is_AI"]
        }).reset_index(drop=True)

        print(f"\nEvaluating Seed {s} on {len(eval_seed_df)} strictly held-out translated test samples...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

        metrics = evaluate_roberta_binary_on_df(model, tokenizer, eval_seed_df, device)
        print(f"  Seed {s} Results -> Acc: {metrics['accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

        eval_results.append({
            "Seed": s,
            "Model": "RoBERTa_Binary",
            "Dataset": "NLLB_True_Translated_Test",
            "N_samples": len(eval_seed_df),
            "Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "Macro_F1": round(metrics["macro_f1"], 4),
            "Confusion_Matrix": json.dumps(metrics["confusion_matrix"])
        })

    df_results = pd.DataFrame(eval_results)
    print("\n" + "=" * 80)
    print("VERIFIED ROBERTA TRANSLATION ROBUSTNESS METRICS:")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # Mean and standard deviation across seeds
    accs = df_results["Accuracy"].values
    f1s = df_results["Macro_F1"].values
    print(f"\nOverall Summary Across Seeds:")
    print(f"  Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    metrics_csv = "results/phase3/T3_roberta_true_translation_metrics.csv"
    df_results.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv}")


if __name__ == "__main__":
    main()
