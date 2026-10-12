"""Task 3: RoBERTa multi-class CM under Gemma-27B paraphrase.
Runs chunk-pooled inference (same as evaluate_roberta_baseline.py),
exports numeric CM (counts + row-normalised), per-doc predictions, figure.
"""
import os, json, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

MAX_LENGTH, OVERLAP, MULTI_DIR = 512, 50, "results/models/RoBERTa_Multi"
CSV = "data/processed/processed_paraphrased_articles.csv"

def chunks(text, tok):
    cs, st = MAX_LENGTH - 2, (MAX_LENGTH - 2) - OVERLAP
    ids = tok.encode(str(text), add_special_tokens=False, truncation=False)
    if len(ids) <= cs:
        return [tok(str(text), truncation=True, max_length=MAX_LENGTH)]
    out = []
    for s in range(0, len(ids), st):
        w = ids[s:s+cs]
        if not w: break
        out.append(tok(tok.decode(w, skip_special_tokens=True), truncation=True, max_length=MAX_LENGTH))
        if s + cs >= len(ids): break
    return out

@torch.no_grad()
def predict_one(text, model, tok, device):
    ls = []
    for c in chunks(text, tok):
        inp = {k: torch.tensor(v).unsqueeze(0).to(device) for k, v in c.items()}
        ls.append(model(**inp).logits.squeeze(0).cpu().numpy())
    return int(np.argmax(np.mean(ls, axis=0)))

device = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(MULTI_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MULTI_DIR).to(device).eval()
with open(os.path.join(MULTI_DIR, "label_mapping.json")) as f:
    m = json.load(f)
label2id, id2label = m["label2id"], {int(k): v for k, v in m["id2label"].items()}
names = [id2label[i] for i in range(len(id2label))]

df = pd.read_csv(CSV, low_memory=True)
df = df[df["Writer"].isin(label2id)].copy()
df["label"] = df["Writer"].map(label2id)
texts, y_true = df["Text"].tolist(), df["label"].astype(int).tolist()
print(f"N={len(texts)} classes={names}")

t0 = time.time()
y_pred = [predict_one(t, model, tok, device) for t in texts]
dt = time.time() - t0
print(f"done {dt:.1f}s latency={dt/len(texts)*1000:.1f}ms/sample")

acc = accuracy_score(y_true, y_pred)
_, _, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
print(f"Acc={acc:.4f} MacroF1={f1:.4f}")
print(pd.crosstab(pd.Series(y_true).map(id2label).rename("true"),
                  pd.Series(y_pred).map(id2label).rename("pred")))

cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
pd.DataFrame(cm, index=names, columns=names).to_csv("newtasks2/T17_RoBERTa_Gemma_CM_counts.csv")
pd.DataFrame(np.round(cmn, 4), index=names, columns=names).to_csv("newtasks2/T17_RoBERTa_Gemma_CM_rownorm.csv")
pd.DataFrame({"Writer_true": [id2label[t] for t in y_true],
              "Pred": [id2label[p] for p in y_pred]}).to_csv("newtasks2/T17_RoBERTa_Gemma_preds.csv", index=False)
pd.DataFrame([{"Dataset": "Paraphrased_Gemma-27B", "N": len(texts), "Accuracy": round(acc,4),
               "MacroF1": round(float(f1),4), "Latency_ms": round(dt/len(texts)*1000,2)}]
             ).to_csv("newtasks2/T17_RoBERTa_Gemma_summary.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for ax, data, fmt, ttl in zip(axes, [cm, cmn], ["d", ".2f"], ["Count", "Row-normalised"]):
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=names, yticklabels=names, ax=ax)
    ax.set_title(f"RoBERTa Multi — Gemma-27B paraphrase\n{ttl}", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (original author)")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("newtasks2/F17_RoBERTa_Gemma_CM.png", dpi=150, bbox_inches="tight")
print("Saved -> newtasks2/T17_*, F17_RoBERTa_Gemma_CM.png")
print("Pred dist:", pd.Series([id2label[p] for p in y_pred]).value_counts().to_dict())
