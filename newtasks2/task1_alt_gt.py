"""Task 1: Alternative Ground Truth for Paraphrased Sets.
Relabel each paraphrased doc with paraphraser -> nearest training class,
recompute Accuracy + Macro F1, compare vs original GT.
Uses saved Multi_BERT XGBoost (no retraining). CPU-only.
"""
import json, joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

MODEL_DIR = "results/models"
CONFIGS = [
    ("Gemma-27B",   "data/processed/processed_paraphrased_articles.csv",  "Gemma-2-9B"),
    ("Gemma-2-9B",  "data/processed/processed_paraphrased_gemma2.csv",    "Gemma-2-9B"),
    ("Qwen-2",      "data/processed/processed_paraphrased_qwen2.csv",     "Qwen-2-72B"),
    ("Mistral-7B",  "data/processed/processed_paraphrased_mistral7b.csv", "Mistral-7B"),
    ("LLaMA-3",     "data/processed/processed_paraphrased_llama3.csv",    "Llama-8B"),
    ("GPT-4o-mini", "data/processed/processed_paraphrased_gpt4o_mini.csv","GPT-4o"),
]

with open(f"{MODEL_DIR}/Bin_BERT_features.json") as f:
    features = json.load(f)
mul_model = joblib.load(f"{MODEL_DIR}/Multi_BERT_model.joblib")
mul_scaler = joblib.load(f"{MODEL_DIR}/Multi_BERT_scaler.joblib")
le = joblib.load(f"{MODEL_DIR}/Multi_BERT_label_encoder.joblib")
classes = list(le.classes_)
print("Classes:", classes)

rows = []
for pname, csv, alt_class in CONFIGS:
    df = pd.read_csv(csv, low_memory=True)
    # original GT: Writer column (drop rows with unknown writers — should be none)
    dfm = df[df["Writer"].isin(classes)].copy()
    dropped = len(df) - len(dfm)
    y_orig = le.transform(dfm["Writer"])
    X = dfm.reindex(columns=features, fill_value=0).astype("float32")
    Xs = pd.DataFrame(mul_scaler.transform(X).astype("float32"), columns=features)
    y_pred = mul_model.predict(Xs)

    # original-GT metrics (standard 7-class)
    acc_o = accuracy_score(y_orig, y_pred)
    _, _, f1_o, _ = precision_recall_fscore_support(y_orig, y_pred, average="macro", zero_division=0)

    # alternative GT: every sample -> paraphraser class
    alt_id = int(le.transform([alt_class])[0])
    y_alt = [alt_id] * len(dfm)
    acc_a = accuracy_score(y_alt, y_pred)
    _, _, f1_a, _ = precision_recall_fscore_support(y_alt, y_pred, average="macro", zero_division=0)
    # per-class F1 for the paraphraser class itself (= attribution recall harmonic)
    _, _, f1_per, _ = precision_recall_fscore_support(y_alt, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
    # predicted-class distribution
    cm = confusion_matrix(y_orig, y_pred, labels=list(range(len(classes))))
    pred_counts = cm.sum(axis=0)
    pred_dist = {c: int(pred_counts[i]) for i, c in enumerate(classes)}
    top3 = sorted(pred_dist.items(), key=lambda kv: -kv[1])[:3]

    rows.append({
        "Paraphraser": pname,
        "Alt_Class": alt_class,
        "N": len(dfm),
        "Dropped_unknown_writer": dropped,
        "Orig_Acc": round(float(acc_o), 4),
        "Orig_MacroF1": round(float(f1_o), 4),
        "Alt_Acc_attr_rate": round(float(acc_a), 4),
        "Alt_MacroF1_7class": round(float(f1_a), 4),
        f"F1_{alt_class}": round(float(f1_per[alt_id]), 4),
        "Top3_predicted": str(top3),
        "Pred_dist": str(pred_dist),
    })
    print(f"{pname:12s} -> {alt_class:10s} N={len(dfm)} | Orig Acc={acc_o:.4f} F1={f1_o:.4f} | Alt Acc={acc_a:.4f} F1={f1_a:.4f} | top3={top3}")

out = pd.DataFrame(rows)
out.to_csv("results/tables/T16_Alternative_GT.csv", index=False)
print("\nSaved -> results/tables/T16_Alternative_GT.csv")
print(out.to_string(index=False))

# Variant excluding Human-origin rows (AI-only alternative GT) for appendix
print("\n--- AI-only variant (exclude Human-origin rows from alt GT) ---")
rows2 = []
for pname, csv, alt_class in CONFIGS:
    df = pd.read_csv(csv, low_memory=True)
    dfm = df[(df["Writer"].isin(classes)) & (df["Writer"] != "Human")].copy()
    y_orig = le.transform(dfm["Writer"])
    X = dfm.reindex(columns=features, fill_value=0).astype("float32")
    Xs = pd.DataFrame(mul_scaler.transform(X).astype("float32"), columns=features)
    y_pred = mul_model.predict(Xs)
    acc_o = accuracy_score(y_orig, y_pred)
    _, _, f1_o, _ = precision_recall_fscore_support(y_orig, y_pred, average="macro", zero_division=0)
    alt_id = int(le.transform([alt_class])[0])
    y_alt = [alt_id] * len(dfm)
    acc_a = accuracy_score(y_alt, y_pred)
    _, _, f1_a, _ = precision_recall_fscore_support(y_alt, y_pred, average="macro", zero_division=0)
    print(f"{pname:12s} N_AI={len(dfm)} | Orig Acc={acc_o:.4f} F1={f1_o:.4f} | Alt Acc={acc_a:.4f} F1={f1_a:.4f}")
    rows2.append({"Paraphraser": pname, "Alt_Class": alt_class, "N_AI": len(dfm),
                  "Orig_Acc": round(float(acc_o),4), "Orig_MacroF1": round(float(f1_o),4),
                  "Alt_Acc": round(float(acc_a),4), "Alt_MacroF1": round(float(f1_a),4)})
pd.DataFrame(rows2).to_csv("results/tables/T16_Alternative_GT_AIonly.csv", index=False)
print("Saved -> results/tables/T16_Alternative_GT_AIonly.csv")
