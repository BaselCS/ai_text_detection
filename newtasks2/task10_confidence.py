"""Task 10: mean predicted confidence (max softmax/proba) for correct
vs incorrect attributions, hybrid Multi_BERT, across 6 paraphrased sets."""
import json
import joblib
import numpy as np
import pandas as pd

with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
model = joblib.load("results/models/Multi_BERT_model.joblib")
sc = joblib.load("results/models/Multi_BERT_scaler.joblib")
le = joblib.load("results/models/Multi_BERT_label_encoder.joblib")

SETS = {
    "Gemma-27B": "data/processed/processed_paraphrased_articles.csv",
    "Gemma-2-9B": "data/processed/processed_paraphrased_gemma2.csv",
    "Qwen-2": "data/processed/processed_paraphrased_qwen2.csv",
    "Mistral-7B": "data/processed/processed_paraphrased_mistral7b.csv",
    "LLaMA-3": "data/processed/processed_paraphrased_llama3.csv",
    "GPT-4o-mini": "data/processed/processed_paraphrased_gpt4o_mini.csv",
}
rows = []
for sname, csv in SETS.items():
    df = pd.read_csv(csv, low_memory=True)
    df = df[df["Writer"].isin(le.classes_)].copy()
    yt = le.transform(df["Writer"])
    X = pd.DataFrame(sc.transform(df.reindex(columns=features, fill_value=0).astype("float32")).astype("float32"), columns=features)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    ok = conf[pred == yt]
    bad = conf[pred != yt]
    print(f"{sname}: N={len(df)} acc={(pred == yt).mean():.4f} "
          f"conf_correct={ok.mean():.4f}±{ok.std():.4f} (n={len(ok)}) "
          f"conf_wrong={bad.mean():.4f}±{bad.std():.4f} (n={len(bad)})")
    rows.append({"Paraphrased_set": sname, "N": len(df), "Acc": round(float((pred == yt).mean()), 4),
                 "Conf_correct_mean": round(float(ok.mean()), 4), "Conf_correct_std": round(float(ok.std()), 4),
                 "Conf_wrong_mean": round(float(bad.mean()), 4), "Conf_wrong_std": round(float(bad.std()), 4)})
pd.DataFrame(rows).to_csv("newtasks2/T23_Confidence.csv", index=False)
print("Saved -> newtasks2/T23_Confidence.csv")
