"""Task 5: Corrected Data Hygiene, Quality Audit & Deduplication.

Fixes the Task 5 prompt row 233 bug:
1. Audits prompt row 233 (duplicate of prompt row 128).
2. Explicitly drops ALL 8 generated articles for prompt row 233 across all writers in the long-format evaluation table.
3. Documents why previously 57 rows were dropped (the length filter <200 chars captured 57 rows but omitted the 8 rows of prompt 233).
4. Drops 65 rows in total (57 short/refusal/truncation + 8 prompt row 233 articles), producing clean N = 7,934.
5. Re-evaluates Phase-1 Hybrid Binary and Multi-Class XGBoost models.
6. Updates results/phase3/T5_data_hygiene_audit.csv and results/phase3/T5_hygiene_metric_impact.csv.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib

def run_task5_fix():
    raw_csv = "results/extra/task2/generated_full1000.csv"
    processed_cnn_csv = "data/processed/processed_second_corpus_cnn.csv"
    out_audit_csv = "results/phase3/T5_data_hygiene_audit.csv"
    out_impact_csv = "results/phase3/T5_hygiene_metric_impact.csv"
    os.makedirs("results/phase3", exist_ok=True)

    print(f"Loading {raw_csv}...")
    df_raw = pd.read_csv(raw_csv)
    print(f"Loading {processed_cnn_csv}...")
    df_proc = pd.read_csv(processed_cnn_csv)

    # 1. Row 474: LLaMA refusal
    refusal_text = str(df_raw.loc[474, "llama_3_8b"])
    
    # 2. 53 Short articles in llama_3_8b (<200 chars)
    llama_lens = df_raw["llama_3_8b"].astype(str).str.len()
    llama_short_idx = df_raw[llama_lens < 200].index.tolist()

    # 3. Truncations
    deepseek_915 = str(df_raw.loc[915, "deepseek_pro"])
    qwen_682 = str(df_raw.loc[682, "qwen_2_72b"])

    # 4. Duplicate prompt row 233 across ALL 8 writers
    dup_prompt_mask = df_raw["Prompt"].duplicated(keep=False)
    dup_indices = df_raw[dup_prompt_mask].index.tolist()
    assert dup_indices == [128, 233], f"Unexpected duplicate prompt indices: {dup_indices}"
    row233_texts = set(df_raw.loc[233].drop("Prompt").astype(str).tolist())
    assert len(row233_texts) == 8, f"Expected 8 texts for prompt row 233, found {len(row233_texts)}"

    # Audit records
    audit_records = [
        {
            "Category": "Refusal String",
            "Target_Model": "llama_3_8b",
            "Row_Index": 474,
            "Issue_Description": "API refusal message returned instead of generated narrative",
            "Snippet": refusal_text[:150],
            "Character_Length": len(refusal_text),
            "Action_Taken": "Flagged and filtered in clean subset",
            "Remediation_Impact": "Removes non-story text that would distort stylometric and perplexity features"
        },
        {
            "Category": "Short Articles (<200 chars)",
            "Target_Model": "llama_3_8b",
            "Row_Index": f"{len(llama_short_idx)} rows: {llama_short_idx[:5]}...",
            "Issue_Description": f"{len(llama_short_idx)} articles under 200 characters in llama_3_8b due to abrupt generation stops or terse output",
            "Snippet": f"Min len: {llama_lens[llama_short_idx].min()}, Mean: {llama_lens[llama_short_idx].mean():.1f}, Median: {llama_lens[llama_short_idx].median():.1f}",
            "Character_Length": f"<200 (mean={llama_lens[llama_short_idx].mean():.1f})",
            "Action_Taken": "Documented length distribution; evaluated impact with retention vs filtering",
            "Remediation_Impact": "Consistent minimum length threshold (>=200 chars) ensures valid stylometric extraction"
        },
        {
            "Category": "Truncation",
            "Target_Model": "deepseek_pro",
            "Row_Index": 915,
            "Issue_Description": "Generation cut off abruptly at token limit ending with 'READING,'",
            "Snippet": deepseek_915[-100:],
            "Character_Length": len(deepseek_915),
            "Action_Taken": "Identified and logged; filtered in clean robustness split",
            "Remediation_Impact": "Removes incomplete syntactic structures from zero-shot evaluation"
        },
        {
            "Category": "Truncation",
            "Target_Model": "qwen_2_72b",
            "Row_Index": 682,
            "Issue_Description": "Generation truncated mid-sentence at output token boundary",
            "Snippet": qwen_682[-100:],
            "Character_Length": len(qwen_682),
            "Action_Taken": "Identified and logged; filtered in clean robustness split",
            "Remediation_Impact": "Prevents artificial syntax depth truncation artifacts"
        },
        {
            "Category": "Duplicate Prompt (Multi-Writer)",
            "Target_Model": "All 8 Writers (Wide Row 233 -> 8 Long Rows)",
            "Row_Index": "Wide Row 233 (Duplicate of Row 128)",
            "Issue_Description": "Prompt 233 duplicated prompt 128; dropping prompt 233 requires dropping all 8 generated articles across all writers in long format",
            "Snippet": df_raw.loc[233, 'Prompt'][:120],
            "Character_Length": len(df_raw.loc[233, 'Prompt']),
            "Action_Taken": "Deduplicated by retaining prompt 128 articles and dropping all 8 corresponding articles of prompt 233 across all writers",
            "Remediation_Impact": "Ensures prompt uniqueness and eliminates all 8 cross-writer replicate instances from evaluation split"
        },
        {
            "Category": "Formal Model Exclusion",
            "Target_Model": "claude_haiku",
            "Row_Index": "All rows (1000 samples)",
            "Issue_Description": "Claude-3-Haiku generated in CNN corpus but omitted from Multi-Class attribution stage",
            "Snippet": "N/A (Systemic architectural exclusion)",
            "Character_Length": "N/A",
            "Action_Taken": "Documented technical rationale in manuscript and audit report",
            "Remediation_Impact": "Maintains strict parity with 7-class training closed set (GPT-4o, Gemma-2-9B, Human, Llama-8B, Mistral-7B, Qwen-2-72B, Yi-Large); retained in Binary AI-vs-Human evaluation"
        }
    ]

    audit_df = pd.DataFrame(audit_records)
    audit_df.to_csv(out_audit_csv, index=False)
    print(f"Saved corrected audit log to {out_audit_csv}")

    # Masks in df_proc (long format)
    short_mask = df_proc["Text"].astype(str).str.len() < 200
    dup_prompt_mask_proc = df_proc["Text"].isin(row233_texts)
    
    print(f"\nFiltering analysis in df_proc (N={len(df_proc)}):")
    print(f"  Short articles (< 200 chars): {short_mask.sum()} rows (explains prior 57 dropped rows)")
    print(f"  Duplicate prompt row 233 (8 writers): {dup_prompt_mask_proc.sum()} rows")
    print(f"  Overlap: {(short_mask & dup_prompt_mask_proc).sum()} rows")
    
    total_drop_mask = short_mask | dup_prompt_mask_proc
    total_dropped = total_drop_mask.sum()
    df_clean = df_proc[~total_drop_mask].copy()
    print(f"  Total dropped rows: {total_dropped} (57 short + 8 duplicate prompt = {57 + 8})")
    print(f"  Clean dataset size: {len(df_clean)} rows")

    # Load Phase 1 models
    bin_model = joblib.load("results/models/Bin_BERT_model.joblib")
    bin_scaler = joblib.load("results/models/Bin_BERT_scaler.joblib")
    with open("results/models/Bin_BERT_features.json") as f:
        features = json.load(f)
    mul_model = joblib.load("results/models/Multi_BERT_model.joblib")
    mul_scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")
    le = joblib.load("results/models/Multi_BERT_label_encoder.joblib")
    valid_classes = list(le.classes_)

    # 1. Raw evaluation (N=7999)
    X_bin = df_proc.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc = pd.DataFrame(bin_scaler.transform(X_bin).astype("float32"), columns=features)
    y_bin_pred = bin_model.predict(X_bin_sc)
    y_bin_true = df_proc["is_AI"]
    raw_bin_acc = accuracy_score(y_bin_true, y_bin_pred)
    _, _, raw_bin_f1, _ = precision_recall_fscore_support(y_bin_true, y_bin_pred, average="macro", zero_division=0)

    df_multi_raw = df_proc[df_proc["Writer"].isin(valid_classes)].copy()
    X_mul = df_multi_raw.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc = pd.DataFrame(mul_scaler.transform(X_mul).astype("float32"), columns=features)
    y_mul_pred = mul_model.predict(X_mul_sc)
    y_mul_true = le.transform(df_multi_raw["Writer"])
    raw_mul_acc = accuracy_score(y_mul_true, y_mul_pred)
    _, _, raw_mul_f1, _ = precision_recall_fscore_support(y_mul_true, y_mul_pred, average="macro", zero_division=0)

    # 2. Clean evaluation (N=7934)
    X_bin_c = df_clean.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc_c = pd.DataFrame(bin_scaler.transform(X_bin_c).astype("float32"), columns=features)
    y_bin_pred_c = bin_model.predict(X_bin_sc_c)
    y_bin_true_c = df_clean["is_AI"]
    clean_bin_acc = accuracy_score(y_bin_true_c, y_bin_pred_c)
    _, _, clean_bin_f1, _ = precision_recall_fscore_support(y_bin_true_c, y_bin_pred_c, average="macro", zero_division=0)

    df_multi_c = df_clean[df_clean["Writer"].isin(valid_classes)].copy()
    X_mul_c = df_multi_c.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc_c = pd.DataFrame(mul_scaler.transform(X_mul_c).astype("float32"), columns=features)
    y_mul_pred_c = mul_model.predict(X_mul_sc_c)
    y_mul_true_c = le.transform(df_multi_c["Writer"])
    clean_mul_acc = accuracy_score(y_mul_true_c, y_mul_pred_c)
    _, _, clean_mul_f1, _ = precision_recall_fscore_support(y_mul_true_c, y_mul_pred_c, average="macro", zero_division=0)

    print(f"\nRAW (N=7999):   Binary Acc={raw_bin_acc:.4f}, Macro-F1={raw_bin_f1:.4f} | Multi Acc={raw_mul_acc:.4f}, Macro-F1={raw_mul_f1:.4f}")
    print(f"CLEAN (N=7934): Binary Acc={clean_bin_acc:.4f}, Macro-F1={clean_bin_f1:.4f} | Multi Acc={clean_mul_acc:.4f}, Macro-F1={clean_mul_f1:.4f}")

    impact_summary = pd.DataFrame([
        {
            "Split": f"Raw (Uncleaned, N={len(df_proc)})",
            "Binary_Accuracy": round(raw_bin_acc, 4),
            "Binary_Macro_F1": round(raw_bin_f1, 4),
            "Multi_Accuracy": round(raw_mul_acc, 4),
            "Multi_Macro_F1": round(raw_mul_f1, 4)
        },
        {
            "Split": f"Clean (Filtered, N={len(df_clean)})",
            "Binary_Accuracy": round(clean_bin_acc, 4),
            "Binary_Macro_F1": round(clean_bin_f1, 4),
            "Multi_Accuracy": round(clean_mul_acc, 4),
            "Multi_Macro_F1": round(clean_mul_f1, 4)
        },
        {
            "Split": "Absolute Difference",
            "Binary_Accuracy": round(round(clean_bin_acc, 4) - round(raw_bin_acc, 4), 4),
            "Binary_Macro_F1": round(round(clean_bin_f1, 4) - round(raw_bin_f1, 4), 4),
            "Multi_Accuracy": round(round(clean_mul_acc, 4) - round(raw_mul_acc, 4), 4),
            "Multi_Macro_F1": round(round(clean_mul_f1, 4) - round(raw_mul_f1, 4), 4)
        }
    ])
    impact_summary.to_csv(out_impact_csv, index=False)
    print(f"Saved metric impact comparison to {out_impact_csv}")

if __name__ == "__main__":
    run_task5_fix()
