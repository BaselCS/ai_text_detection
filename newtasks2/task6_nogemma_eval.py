"""Task 6 eval: No-Gemma hybrid model on Gemma-27B, Mistral-7B, CNN_DailyMail sets.
Loads saved newtasks2/T21_NoGemma_*.joblib. CPU/GPU inference only.
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

OUT = "newtasks2"
with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
model = joblib.load(f"{OUT}/T21_NoGemma_model.joblib")
sc = joblib.load(f"{OUT}/T21_NoGemma_scaler.joblib")
le = joblib.load(f"{OUT}/T21_NoGemma_label_encoder.joblib")
print("classes:", list(le.classes_))


def predict(df):
    X = df.reindex(columns=features, fill_value=0).astype("float32")
    return model.predict(pd.DataFrame(sc.transform(X).astype("float32"), columns=features))


def evaluate(name, csv):
    d = pd.read_csv(csv, low_memory=True)
    d_known = d[d["Writer"].isin(le.classes_)].copy()
    d_gemma = d[d["Writer"] == "Gemma-2-9B"].copy()
    d_other = d[~d["Writer"].isin(list(le.classes_) + ["Gemma-2-9B"])].copy()
    out = {}
    yp_known = predict(d_known)
    yt = le.transform(d_known["Writer"])
    a = accuracy_score(yt, yp_known)
    _, _, f1, _ = precision_recall_fscore_support(yt, yp_known, average="macro", zero_division=0)
    cm = confusion_matrix(yt, yp_known, labels=list(range(len(le.classes_))))
    dist = dict(sorted(zip(le.classes_, cm.sum(axis=0).tolist()), key=lambda kv: -kv[1]))
    print(f"{name}/scorable: N={len(d_known)} Acc={a:.4f} F1={f1:.4f} dist={dist}")
    out["scorable"] = (a, f1, dist, len(d_known))
    pd.DataFrame(cm, index=list(le.classes_), columns=list(le.classes_)).to_csv(f"{OUT}/T21_CM_{name}.csv")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm / cm.sum(axis=1, keepdims=True), annot=True, fmt=".2f", cmap="Blues",
                xticklabels=list(le.classes_), yticklabels=list(le.classes_), ax=ax)
    ax.set_title(f"No-Gemma model - {name} (row-norm)", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{OUT}/F21_CM_{name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    if len(d_gemma):
        yp_g = predict(d_gemma)
        counts = np.bincount(yp_g, minlength=len(le.classes_))
        dist_g = dict(sorted(zip(le.classes_, counts.tolist()), key=lambda kv: -kv[1]))
        print(f"{name}/gemma-origin unscorable: N={len(d_gemma)} pred_dist={dist_g}")
        out["gemma_origin"] = (dist_g, len(d_gemma))
    if len(d_other):
        print(f"{name}/other-dropped writers: {d_other['Writer'].value_counts().to_dict()}")
        out["dropped"] = d_other["Writer"].value_counts().to_dict()
    return out


res = {}
res["Gemma-27B"] = evaluate("Gemma-27B", "data/processed/processed_paraphrased_articles.csv")
res["Mistral-7B"] = evaluate("Mistral-7B", "data/processed/processed_paraphrased_mistral7b.csv")
res["CNN_DailyMail"] = evaluate("CNN_DailyMail", "data/processed/processed_second_corpus_cnn.csv")

rows = []
for name, r in res.items():
    a, f1, dist, n = r["scorable"]
    rows.append({"Dataset": name, "N_scorable": n, "Acc": round(float(a), 4),
                 "MacroF1": round(float(f1), 4), "Pred_dist": str(dist)})
    if "gemma_origin" in r:
        dist_g, ng = r["gemma_origin"]
        rows.append({"Dataset": name + " (Gemma-origin, unscorable)", "N_scorable": ng,
                     "Acc": "n/a", "MacroF1": "n/a", "Pred_dist": str(dist_g)})
pd.DataFrame(rows).to_csv(f"{OUT}/T21_NoGemma_summary.csv", index=False)
print("\nSaved summary:")
print(pd.DataFrame(rows).to_string(index=False))
