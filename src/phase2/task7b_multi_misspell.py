"""Task 7b: fill the missing RoBERTa_Multi Misspell_5%/20% evaluations.

Background: results/phase2/run_seed.py only evaluated Multi on Clean/Paraphrased/Translation,
so T25 tables have no Multi-Misspell rows. Seed-42 Multi model is deleted from disk,
so only the 4 surviving seeds (123/999/2024/8080) can be scored -> reported as 4-seed.

Math identical to evaluate_roberta_baseline.predict_one (chunk -> mean logits -> argmax),
but chunks are batched for speed. Deterministic (eval mode), same predictions.
"""
import json
import os
import sys
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evaluate_roberta_baseline as ev

SEEDS = [123, 999, 2024, 8080]
SETS = {
    "Misspell_5%": "data/processed/processed_misspelled_5.csv",
    "Misspell_20%": "data/processed/processed_misspelled_20.csv",
}
CHUNK_BATCH = 128
device = torch.device("cuda")

rows = []
for seed in SEEDS:
    mdir = f"results/models/RoBERTa_Multi_s{seed}"
    with open(f"{mdir}/label_mapping.json") as f:
        label2id = json.load(f)["label2id"]
    tok = AutoTokenizer.from_pretrained(mdir)
    mod = AutoModelForSequenceClassification.from_pretrained(mdir).to(device).eval()
    for name, path in SETS.items():
        df = pd.read_csv(path, low_memory=True)
        df = df[df["Writer"].isin(label2id)].copy()
        df["label"] = df["Writer"].map(label2id)
        df = df.dropna(subset=["label", "Text"]).reset_index(drop=True)
        labels = df["label"].astype(int).tolist()
        texts = df["Text"].astype(str).tolist()
        # chunk all texts (same function as predict_one)
        per_text_chunks, flat, owner = [], [], []
        for i, t in enumerate(texts):
            ch = ev.chunk_text_by_tokens(t, tok)
            per_text_chunks.append(len(ch))
            for c in ch:
                flat.append(c)
                owner.append(i)
        # batched forward over chunks
        chunk_logits = [None] * len(flat)
        with torch.no_grad():
            for s in range(0, len(flat), CHUNK_BATCH):
                batch = flat[s:s + CHUNK_BATCH]
                enc = tok.pad(batch, padding=True, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                out = mod(**enc).logits.cpu().numpy()
                for k, lg in enumerate(out):
                    chunk_logits[s + k] = lg
        # mean logits per text -> argmax
        preds = []
        pos = 0
        for n in per_text_chunks:
            avg = np.mean(chunk_logits[pos:pos + n], axis=0)
            preds.append(int(np.argmax(avg)))
            pos += n
        acc = accuracy_score(labels, preds)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        _, _, f1w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
        rows.append({"Seed": seed, "Model": "RoBERTa_Multi", "Dataset": name,
                     "count": len(texts), "accuracy": round(float(acc), 4),
                     "macro_f1": round(float(f1), 4), "weighted_f1": round(float(f1w), 4)})
        print(f"seed {seed} multi {name}: N={len(texts)} Acc={acc:.4f} F1={f1:.4f}", flush=True)
    del mod, tok
    torch.cuda.empty_cache()

out = pd.DataFrame(rows)
out.to_csv("results/phase2/T25b_multi_misspell_4seed.csv", index=False)
print("\nSaved -> results/phase2/T25b_multi_misspell_4seed.csv")
print(out.to_string(index=False))
print("\n4-seed mean±std:")
print(out.groupby("Dataset")[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(4).to_string())
