"""Task 3: Dual Evaluation Settings for Translation Robustness & Adversarial Attack.

Implements the two distinct evaluation regimes required by the supervisor:
Setting A: Both Human & AI NLLB-translated (illustrates RoBERTa detecting NLLB neural signature on human texts).
Setting B (Standard Attack Setting):
  - B1: Unperturbed Held-Out Test Human vs. NLLB-Translated Held-Out Test AI (Full Attack Split).
  - B2: Balanced 1:1 Subset (Unperturbed Test Human vs. NLLB-Translated Test AI, 50% Human : 50% AI).
Evaluates all 5 seeds (42, 123, 999, 2024, 8080) and writes comprehensive comparison to
results/phase3/T3_roberta_true_translation_metrics.csv.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
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
    stratify_col = df_articles["is_AI"].astype(int)
    train_val_df, test_df = train_test_split(
        df_articles, test_size=0.20, random_state=seed, stratify=stratify_col
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.125, random_state=seed,
        stratify=train_val_df["is_AI"].astype(int)
    )
    return set(train_df.index), set(val_df.index), set(test_df.index)


def drop_cross_split_source_duplicates(test_df: pd.DataFrame, articles: pd.DataFrame,
                                       train_idx: set, val_idx: set) -> pd.DataFrame:
    """Drop evaluation rows whose source text also occurs in train/val.

    The corpus contains duplicate source articles.  A row-level split can place
    one copy in train and another in test even when ``orig_idx`` itself is in
    the test split.  Filtering on the source text as well as the mapped index
    closes that leak.  This removes 1 row for seed 42, 1 for seed 123 and 2 for
    seed 8080, giving N = 666, 579, 610, 615, 586.
    """
    train_texts = set(articles.loc[list(train_idx), "Text"].astype(str))
    val_texts = set(articles.loc[list(val_idx), "Text"].astype(str))
    seen = train_texts | val_texts
    overlap = test_df["Text"].astype(str).isin(seen)
    return test_df.loc[~overlap].copy().reset_index(drop=True)

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

def evaluate_roberta(model, tokenizer, df: pd.DataFrame, device):
    model.eval()
    all_preds = []
    labels = df["label"].values

    with torch.no_grad():
        for _, row in df.iterrows():
            chunks = chunk_text_by_tokens(row["Text"], tokenizer)
            chunk_logits = []
            for c in chunks:
                inputs = {k: v.to(device) for k, v in c.items()}
                out = model(**inputs)
                chunk_logits.append(out.logits.squeeze(0))
            avg_logits = torch.stack(chunk_logits).mean(dim=0)
            probs = torch.softmax(avg_logits, dim=-1)
            pred = torch.argmax(probs).item()
            all_preds.append(pred)

    all_preds = np.array(all_preds)
    acc = accuracy_score(labels, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(labels, all_preds) # [[TN, FP], [FN, TP]]
    
    # Class-specific recall
    # Label 0: Human, Label 1: AI
    tn, fp = cm[0, 0], cm[0, 1] if cm.shape[0] > 1 and cm.shape[1] > 1 else (cm[0, 0], 0)
    fn, tp = (cm[1, 0], cm[1, 1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else (0, 0)
    
    ai_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    human_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "macro_f1": f1,
        "ai_recall": ai_recall,
        "human_recall": human_recall,
        "confusion_matrix": cm.tolist()
    }

def run_dual_evaluation():
    raw_nllb_path = "archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv"
    if not os.path.exists(raw_nllb_path):
        raw_nllb_path = "archive/legacy/old_things/paraphrasing/nllb/DB_test_paraphrased_1.3B_ULTRA.csv"
        
    df_nllb = pd.read_csv(raw_nllb_path)
    df_nllb["Writer"] = df_nllb["Writer"].map(CLEAN_MAPPING).fillna(df_nllb["Writer"])
    df_nllb["is_AI"] = df_nllb["is_AI"].astype(int)

    articles_path = "data/processed/processed_articles.csv"
    df_articles = pd.read_csv(articles_path, usecols=["Text", "Writer", "is_AI"])
    # Frozen duplicate-text rule: last occurrence wins, matching the historical
    # NLLB evaluation archive and task3_leakage_audit.py.
    text_to_idx = {}
    for idx, text in enumerate(df_articles["Text"]):
        text_to_idx[text] = idx
    df_nllb["orig_idx"] = [text_to_idx.get(t, -1) for t in df_nllb["Text"]]
    matched_df = df_nllb[df_nllb["orig_idx"] >= 0].copy()

    seeds = [42, 123, 999, 2024, 8080]
    seed_splits = {s: get_splits_for_seed(df_articles, s) for s in seeds}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RoBERTa Dual Settings on device: {device}")

    all_metrics = []

    for s in seeds:
        model_dir = f"results/models/RoBERTa_Binary_s{s}"
        if not os.path.exists(model_dir):
            if s == 42 and os.path.exists("results/models/RoBERTa_Binary"):
                model_dir = "results/models/RoBERTa_Binary"
            else:
                continue

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

        s_train, s_val, s_test = seed_splits[s]
        seed_test_df = matched_df[matched_df["orig_idx"].isin(s_test)].copy().reset_index(drop=True)
        seed_test_df = drop_cross_split_source_duplicates(seed_test_df, df_articles, s_train, s_val)
        assert seed_test_df["orig_idx"].isin(s_train).sum() == 0, f"Leakage detected in seed {s}!"
        assert seed_test_df["orig_idx"].isin(s_val).sum() == 0, f"Val leakage detected in seed {s}!"
        train_text_set = set(df_articles.loc[list(s_train), "Text"].astype(str))
        val_text_set = set(df_articles.loc[list(s_val), "Text"].astype(str))
        assert not seed_test_df["Text"].astype(str).isin(train_text_set | val_text_set).any(), \
            f"Source-text leakage detected in seed {s}!"
        print(f"Seed {s}: N={len(seed_test_df)} "
              f"(Human={(seed_test_df['is_AI'] == 0).sum()}, AI={(seed_test_df['is_AI'] == 1).sum()})")

        # Save the deduplicated Seed-42 evaluation set used by the leakage,
        # surface-artifact, and shallow-baseline scripts.
        if s == 42:
            pd.DataFrame({
                "Text": seed_test_df["paraphrased_text"],
                "Writer": seed_test_df["Writer"],
                "is_AI": seed_test_df["is_AI"],
                "label": seed_test_df["is_AI"],
                "original_text": seed_test_df["Text"],
                "orig_idx": seed_test_df["orig_idx"],
            }).to_csv("results/phase3/T3_true_test_translated.csv", index=False)

        # ---------------------------------------------------------------------
        # Setting A: Both Human & AI NLLB-translated
        # ---------------------------------------------------------------------
        df_A = pd.DataFrame({
            "Text": seed_test_df["paraphrased_text"],
            "label": seed_test_df["is_AI"]
        })
        mA = evaluate_roberta(model, tokenizer, df_A, device)
        all_metrics.append({
            "Setting": "Setting A: Both Human & AI NLLB-Translated",
            "Seed": s,
            "N_Samples": len(df_A),
            "N_Human": int((df_A["label"] == 0).sum()),
            "N_AI": int((df_A["label"] == 1).sum()),
            "Accuracy": round(mA["accuracy"], 4),
            "Precision": round(mA["precision"], 4),
            "Recall": round(mA["recall"], 4),
            "Macro_F1": round(mA["macro_f1"], 4),
            "AI_Recall": round(mA["ai_recall"], 4),
            "Human_Recall": round(mA["human_recall"], 4),
            "Confusion_Matrix": json.dumps(mA["confusion_matrix"]),
            "Evaluation_Notes": "Errors driven by False Positives on machine-translated Human text (RoBERTa detects NLLB neural signature)"
        })

        # ---------------------------------------------------------------------
        # Setting B1: Standard Attack Setting (Unperturbed Test Human vs NLLB-Translated Test AI)
        # ---------------------------------------------------------------------
        texts_B1 = []
        for _, r in seed_test_df.iterrows():
            if r["is_AI"] == 0:
                texts_B1.append(r["Text"])  # Clean unperturbed Human text
            else:
                texts_B1.append(r["paraphrased_text"])  # NLLB-translated AI text
                
        df_B1 = pd.DataFrame({
            "Text": texts_B1,
            "label": seed_test_df["is_AI"]
        })
        mB1 = evaluate_roberta(model, tokenizer, df_B1, device)
        all_metrics.append({
            "Setting": "Setting B1: Standard Attack (Unperturbed Human vs NLLB AI)",
            "Seed": s,
            "N_Samples": len(df_B1),
            "N_Human": int((df_B1["label"] == 0).sum()),
            "N_AI": int((df_B1["label"] == 1).sum()),
            "Accuracy": round(mB1["accuracy"], 4),
            "Precision": round(mB1["precision"], 4),
            "Recall": round(mB1["recall"], 4),
            "Macro_F1": round(mB1["macro_f1"], 4),
            "AI_Recall": round(mB1["ai_recall"], 4),
            "Human_Recall": round(mB1["human_recall"], 4),
            "Confusion_Matrix": json.dumps(mB1["confusion_matrix"]),
            "Evaluation_Notes": "Standard attack: Clean test Human articles vs NLLB-translated test AI articles"
        })

        # ---------------------------------------------------------------------
        # Setting B2: Balanced 1:1 Standard Attack Subset
        # ---------------------------------------------------------------------
        human_sub = df_B1[df_B1["label"] == 0]
        ai_sub = df_B1[df_B1["label"] == 1].sample(n=len(human_sub), random_state=s)
        df_B2 = pd.concat([human_sub, ai_sub]).sample(frac=1.0, random_state=s).reset_index(drop=True)
        mB2 = evaluate_roberta(model, tokenizer, df_B2, device)
        all_metrics.append({
            "Setting": "Setting B2: Balanced 1:1 Standard Attack (50% Human : 50% NLLB AI)",
            "Seed": s,
            "N_Samples": len(df_B2),
            "N_Human": int((df_B2["label"] == 0).sum()),
            "N_AI": int((df_B2["label"] == 1).sum()),
            "Accuracy": round(mB2["accuracy"], 4),
            "Precision": round(mB2["precision"], 4),
            "Recall": round(mB2["recall"], 4),
            "Macro_F1": round(mB2["macro_f1"], 4),
            "AI_Recall": round(mB2["ai_recall"], 4),
            "Human_Recall": round(mB2["human_recall"], 4),
            "Confusion_Matrix": json.dumps(mB2["confusion_matrix"]),
            "Evaluation_Notes": "Balanced 1:1 evaluation (N_human == N_ai) under standard attack"
        })
        print(f"Seed {s} completed across Settings A, B1, and B2.")

    out_df = pd.DataFrame(all_metrics)
    out_csv = "results/phase3/T3_roberta_true_translation_metrics.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved comprehensive translation metrics to {out_csv}")
    
    # Print summary
    for sett in out_df["Setting"].unique():
        sub = out_df[out_df["Setting"] == sett]
        print(f"\n--- {sett} ---")
        print(f"  Accuracy:  {sub['Accuracy'].mean():.4f} +/- {sub['Accuracy'].std():.4f}")
        print(f"  Macro-F1:  {sub['Macro_F1'].mean():.4f} +/- {sub['Macro_F1'].std():.4f}")
        print(f"  AI Recall: {sub['AI_Recall'].mean():.4f} +/- {sub['AI_Recall'].std():.4f}")
        print(f"  Human Rec: {sub['Human_Recall'].mean():.4f} +/- {sub['Human_Recall'].std():.4f}")

if __name__ == "__main__":
    run_dual_evaluation()
