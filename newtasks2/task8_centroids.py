"""Task 8: centroid distances in hybrid feature space.
Standardised (scaler) space, same as XGBoost sees.
Centroid of each paraphrased set vs centroids of all training classes
from processed_articles.csv. Euclidean + cosine.
"""
import json
import numpy as np
import pandas as pd
import joblib

with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")

print("Loading training corpus...", flush=True)
train = pd.read_csv("data/processed/processed_articles.csv", usecols=["Writer"] + features, low_memory=True)
Xtr = scaler.transform(train[features].astype("float32")).astype(np.float64)
train_c = {w: Xtr[(train["Writer"] == w).to_numpy()].mean(axis=0) for w in sorted(train["Writer"].unique())}
print("train classes:", list(train_c.keys()))

SETS = {
    "Gemma-27B": "data/processed/processed_paraphrased_articles.csv",
    "Gemma-2-9B": "data/processed/processed_paraphrased_gemma2.csv",
    "Qwen-2": "data/processed/processed_paraphrased_qwen2.csv",
    "Mistral-7B": "data/processed/processed_paraphrased_mistral7b.csv",
    "LLaMA-3": "data/processed/processed_paraphrased_llama3.csv",
    "GPT-4o-mini": "data/processed/processed_paraphrased_gpt4o_mini.csv",
}
ALT = {"Gemma-27B": "Gemma-2-9B", "Gemma-2-9B": "Gemma-2-9B", "Qwen-2": "Qwen-2-72B",
       "Mistral-7B": "Mistral-7B", "LLaMA-3": "Llama-8B", "GPT-4o-mini": "GPT-4o"}

rows_e, rows_c = [], []
for sname, csv in SETS.items():
    df = pd.read_csv(csv, low_memory=True)
    df = df.reindex(columns=features, fill_value=0)
    c = scaler.transform(df.astype("float32")).astype(np.float64).mean(axis=0)
    n = np.linalg.norm(c)
    erow, crow = {"Paraphrased_set": sname}, {"Paraphrased_set": sname}
    for w, tc in train_c.items():
        erow[w] = round(float(np.linalg.norm(c - tc)), 3)
        crow[w] = round(float(np.dot(c, tc) / (n * np.linalg.norm(tc))), 4)
    erow["nearest_Euclidean"] = min(train_c, key=lambda w: erow[w])
    crow["nearest_cosine"] = max(train_c, key=lambda w: crow[w])
    crow["paraphraser_cosine_rank"] = sorted(train_c, key=lambda w: -crow[w]).index(ALT[sname]) + 1
    rows_e.append(erow)
    rows_c.append(crow)

pd.DataFrame(rows_e).to_csv("newtasks2/T22_Centroid_Euclidean.csv", index=False)
pd.DataFrame(rows_c).to_csv("newtasks2/T22_Centroid_Cosine.csv", index=False)
print(pd.DataFrame(rows_e)[["Paraphrased_set"] + sorted(train_c) + ["nearest_Euclidean"]].to_string(index=False))
print()
print(pd.DataFrame(rows_c)[["Paraphrased_set"] + sorted(train_c) + ["nearest_cosine", "paraphraser_cosine_rank"]].to_string(index=False))
print("\nSaved T22_Centroid_{Euclidean,Cosine}.csv")
