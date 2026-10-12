"""
Task 2.3 → 2.4 → 2.5: Zero-Shot Cross-Corpus Testing Pipeline
====================================================================
Step 2.3: Feature extraction (11-step) on results/extra/task2/generated_full1000.csv
Step 2.4: Zero-shot Inference using Phase-1 Bin_BERT & Multi_BERT XGBoost models
Step 2.5: Save T14 table + F15 confusion matrix figure

GPU target: NVIDIA RTX 3060 12 GB VRAM
Run with:   uv run python helper/cross_corpus_feature_eval.py
"""

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import spacy
from empath import Empath
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import helper.paraphrase_feature_eval as pfe

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

INPUT_CSV = "results/extra/task2/generated_full1000.csv"
OUTPUT_CSV = "data/processed/processed_second_corpus_cnn.csv"
TABLE_DIR  = "results/tables"
FIGURE_DIR = "results/figures"

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

WRITER_MAPPING = {
    "Human_story": "Human",
    "gpt-4o-mini": "GPT-4o",
    "gemma_2_9b": "Gemma-2-9B",
    "mistral_7b": "Mistral-7B",
    "llama_3_8b": "Llama-8B",
    "qwen_2_72b": "Qwen-2-72B",
    "deepseek_pro": "Yi-Large",  # Mapped to Yi-Large for zero-shot compat
    "claude_haiku": "Claude",    # Exists, but will drop in multi-class inference
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_corpus_pipeline(input_csv: str, output_csv: str):
    ckpt = output_csv.replace(".csv", "_checkpoint.csv")

    print(f"\n{'='*70}")
    print(f"  CROSS-CORPUS FEATURE PIPELINE")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"{'='*70}\n")

    if os.path.exists(ckpt):
        df = pd.read_csv(ckpt)
        if "bert_0" in df.columns:
            print("[RESUME] All steps done → saving final file.")
            df.to_csv(output_csv, index=False)
            os.remove(ckpt)
            return df
        print(f"[RESUME] Checkpoint found ({len(df)} rows). Resuming…")
    else:
        print("[1/11] Loading and melting dataset…")
        df_raw = pd.read_csv(input_csv)
        
        # Melt wide to long
        id_vars = ["Prompt", "URL"] if "URL" in df_raw.columns else ["Prompt"]
        melted = df_raw.melt(
            id_vars=id_vars, 
            value_vars=list(WRITER_MAPPING.keys()),
            var_name="Writer_Raw", 
            value_name="Text"
        )
        
        melted["Writer"] = melted["Writer_Raw"].map(WRITER_MAPPING)
        df = melted[["Writer", "Text"]].dropna(subset=["Text"]).reset_index(drop=True)
        
        print(f"  Rows: {len(df)}")
        print(f"  Writers:\n{df['Writer'].value_counts()}")
        
        # Step 2/11
        print("\n[2/11] Loading GPT-2 for Perplexity & UID…")
        gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(pfe.DEVICE)
        if pfe.DEVICE == "cuda":
            gpt_mod = gpt_mod.half()
        gpt_mod.eval()

        df["ppl"] = pfe.process_perplexity(df["Text"].tolist(), gpt_mod, gpt_tok, pfe.DEVICE)
        df.to_csv(ckpt, index=False)

        # Step 3/11
        df["burstiness"] = pfe.process_burstiness(df["Text"].tolist())
        df.to_csv(ckpt, index=False)

        # Step 4/11
        df["ttr"] = pfe.process_ttr(df["Text"].tolist())
        df.to_csv(ckpt, index=False)

        # Step 5/11
        cx, awl, asl = pfe.process_stylometric(df["Text"].tolist())
        df["complex"] = cx
        df["avg_word_len"] = awl
        df["avg_sent_len"] = asl
        df.to_csv(ckpt, index=False)

        # Step 6/11
        df["uid"] = pfe.process_uid(df["Text"].tolist(), gpt_mod, gpt_tok, pfe.DEVICE)
        df.to_csv(ckpt, index=False)

        del gpt_mod, gpt_tok
        gc.collect()
        if pfe.DEVICE == "cuda": torch.cuda.empty_cache()

        # Step 7/11
        lexicon = Empath()
        empath_df = pfe.process_empath(df["Text"].tolist(), lexicon)
        df = pd.concat([df, empath_df], axis=1)
        df.to_csv(ckpt, index=False)

    # Step 8/11
    if "syntax_depth" not in df.columns:
        print("\n[8/11] Syntax Depth (spaCy)…")
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        df["syntax_depth"] = pfe.process_syntax_depth(df["Text"].tolist(), nlp)
        del nlp
        gc.collect()
        df.to_csv(ckpt, index=False)

    # Step 9/11
    if "semantic_mean" not in df.columns:
        print("\n[9/11] Semantic Consistency (MiniLM)…")
        sem_model = SentenceTransformer("all-MiniLM-L6-v2").to(pfe.DEVICE)
        means, stds = pfe.process_semantic(df["Text"].tolist(), sem_model, pfe.DEVICE)
        df["semantic_mean"] = means
        df["semantic_std"] = stds
        del sem_model
        gc.collect()
        if pfe.DEVICE == "cuda": torch.cuda.empty_cache()
        df.to_csv(ckpt, index=False)

    # Step 10/11
    if "is_AI" not in df.columns:
        print("\n[10/11] Adding is_AI label…")
        df["is_AI"] = (df["Writer"].str.lower() != "human").astype(int)
        df = df.dropna().reset_index(drop=True)
        df.to_csv(ckpt, index=False)

    # Step 11/11
    if "bert_0" not in df.columns:
        print("\n[11/11] BERT Embeddings (bert-base-uncased)…")
        bert_emb = pfe.get_bert_embeddings_batched(df["Text"].tolist())
        bert_cols = [f"bert_{i}" for i in range(bert_emb.shape[1])]
        df_bert = pd.DataFrame(bert_emb, columns=bert_cols, index=df.index)
        df = pd.concat([df, df_bert], axis=1)
        del bert_emb, df_bert
        gc.collect()
        if pfe.DEVICE == "cuda": torch.cuda.empty_cache()

    print(f"\n[DONE] Saving → {output_csv}  (shape={df.shape})")
    df.to_csv(output_csv, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  TASK 2.3 → 2.5 : CROSS-CORPUS VALIDATION PIPELINE")
    print("="*70)

    # ── Step 2.3: Feature Extraction ─────────────────────────────────────────
    if not os.path.exists(OUTPUT_CSV):
        df = run_cross_corpus_pipeline(INPUT_CSV, OUTPUT_CSV)
    else:
        print(f"[SKIP] {OUTPUT_CSV} already exists. Loading...")
        df = pd.read_csv(OUTPUT_CSV)

    # ── Load Phase-1 models ──────────────────────────────────────────────────
    print("\n>>> Ensuring Phase-1 XGBoost models …")
    models = pfe.ensure_phase1_models()

    # ── Step 2.4: Zero-Shot Inference + Evaluation ───────────────────────────
    print("\n>>> STEP 2.4 — Zero-Shot Inference & Evaluation")
    
    features   = models["Bin_BERT"]["features"]
    bin_model  = models["Bin_BERT"]["model"]
    bin_scaler = models["Bin_BERT"]["scaler"]
    mul_model  = models["Multi_BERT"]["model"]
    mul_scaler = models["Multi_BERT"]["scaler"]
    le         = models["Multi_BERT"]["le"]

    # Binary evaluation
    y_bin_true = df["is_AI"]
    X_bin = df.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc = pd.DataFrame(bin_scaler.transform(X_bin).astype("float32"), columns=features)
    y_bin_pred = bin_model.predict(X_bin_sc)

    bin_acc = pfe.accuracy_score(y_bin_true, y_bin_pred)
    prec, rec, f1, _ = pfe.precision_recall_fscore_support(
        y_bin_true, y_bin_pred, average="macro", zero_division=0
    )
    print(f"\n  [Binary]  Acc={bin_acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # Multi-class evaluation
    valid_classes = list(le.classes_)
    df_multi = df[df["Writer"].isin(valid_classes)].copy()
    y_mul_true = le.transform(df_multi["Writer"])

    X_mul = df_multi.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc = pd.DataFrame(mul_scaler.transform(X_mul).astype("float32"), columns=features)
    y_mul_pred = mul_model.predict(X_mul_sc)

    mul_acc = pfe.accuracy_score(y_mul_true, y_mul_pred)
    mul_prec, mul_rec, mul_f1, _ = pfe.precision_recall_fscore_support(
        y_mul_true, y_mul_pred, average="macro", zero_division=0
    )
    print(f"  [Multi ]  Acc={mul_acc:.4f}  P={mul_prec:.4f}  R={mul_rec:.4f}  F1={mul_f1:.4f}")

    print("\n  Per-class report:")
    print(pfe.classification_report(
        y_mul_true, y_mul_pred,
        target_names=valid_classes,
        zero_division=0,
    ))

    # ── Step 2.5: Save Table & Figures ───────────────────────────────────────
    print("\n>>> STEP 2.5 — Saving Table & Figures")
    fig_path = f"{FIGURE_DIR}/F15_Second_Corpus_Confusion_Matrix.png"
    pfe.plot_confusion_matrix(
        y_mul_true, y_mul_pred,
        class_names=valid_classes,
        save_path=fig_path,
        title="Zero-Shot Cross-Corpus Multi-Class Attribution",
    )

    t14_path = f"{TABLE_DIR}/T14_Second_Corpus_Validation.csv"
    res = {
        "Dataset": "Second_Corpus_CNN",
        "Binary_Accuracy": round(bin_acc, 4),
        "Binary_Macro_Precision": round(prec, 4),
        "Binary_Macro_Recall":    round(rec,  4),
        "Binary_Macro_F1":        round(f1,   4),
        "Multi_Accuracy":         round(mul_acc,  4),
        "Multi_Macro_Precision":  round(mul_prec, 4),
        "Multi_Macro_Recall":     round(mul_rec,  4),
        "Multi_Macro_F1":         round(mul_f1,   4),
    }
    t14_df = pd.DataFrame([res])
    t14_df.to_csv(t14_path, index=False)
    print(f"  Saved → {t14_path}")
    print("\n" + t14_df.to_string(index=False))

    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
    os.system('notify-send "Task 2 Evaluation Complete" && /home/basel/Work/notify.sh')
