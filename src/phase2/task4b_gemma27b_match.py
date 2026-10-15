"""Task 4b: Reconstruct Gemma-27B (original) paraphrase pairs + perturbation magnitude.

Gemma-27B paraphrases live in data/need_processing/paraphrased_articles.csv
(Writer + Article only, N=2094). Originals are recovered by TF-IDF top-1 match
within the same Writer against the base corpus data/need_processing/articles.csv
(N=51247, 7321 per Writer), then BERTScore-F1 (roberta-large) + sacrebleu
sentence-BLEU are computed exactly like task4_perturb.py.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

need = pd.read_csv("data/need_processing/paraphrased_articles.csv")
base = pd.read_csv("data/need_processing/articles.csv")
print(f"need: {need.shape}, base: {base.shape}")

pairs = []
margins = []
for w in sorted(need["Writer"].unique()):
    ng = need[need["Writer"] == w].reset_index(drop=True)
    cand = base[base["Writer"] == w].reset_index(drop=True)
    ctexts = cand["Article"].astype(str).tolist()
    qtexts = ng["Article"].astype(str).tolist()
    print(f"{w}: {len(qtexts)} queries vs {len(ctexts)} candidates -> TF-IDF...", flush=True)
    vec = TfidfVectorizer(max_features=30000, stop_words="english")
    C = vec.fit_transform(ctexts)
    Q = vec.transform(qtexts)
    sims = cosine_similarity(Q, C)  # (nq, nc)
    top1 = sims.argmax(axis=1)
    part = np.partition(sims, -2, axis=1)[:, -2:]
    margin = part[:, 1] - part[:, 0]  # top1 - top2
    margins.extend(margin.tolist())
    for i, j in enumerate(top1):
        pairs.append({
            "Writer": f"{w}_paraphrased",
            "Original_Text": ctexts[j],
            "Paraphrased_Text": qtexts[i],
            "match_cosine": round(float(sims[i, j]), 4),
            "margin_top1_top2": round(float(margin[i]), 4),
        })
    print(f"  {w}: mean top1={sims[np.arange(len(qtexts)), top1].mean():.3f}, mean margin={margin.mean():.3f}", flush=True)

out = pd.DataFrame(pairs)
out.to_csv("data/processed/raw_paraphrased_gemma27b.csv", index=False)
print(f"Saved {len(out)} pairs -> data/processed/raw_paraphrased_gemma27b.csv")
m = np.array(margins)
print(f"Global: top1-margin mean={m.mean():.3f} median={np.median(m):.3f} frac<0.05={np.mean(m < 0.05):.3f}")
print(out.groupby("Writer").size())
print(out[["match_cosine", "margin_top1_top2"]].describe().to_string())
