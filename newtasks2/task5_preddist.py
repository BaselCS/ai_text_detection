"""Task 5: predicted-class distribution (confusion-matrix column totals)
for hybrid Multi_BERT under misspelling 5%/20% and NLLB translation."""
import json, joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
model = joblib.load("results/models/Multi_BERT_model.joblib")
scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")
le = joblib.load("results/models/Multi_BERT_label_encoder.joblib")
classes = list(le.classes_)

for name, csv in [("Misspell_5%", "data/processed/processed_misspelled_5.csv"),
                  ("Misspell_20%", "data/processed/processed_misspelled_20.csv"),
                  ("Translation_NLLB", "data/processed/processed_translation.csv")]:
    df = pd.read_csv(csv, low_memory=True)
    dfm = df[df["Writer"].isin(classes)].copy()
    y_true = le.transform(dfm["Writer"])
    X = dfm.reindex(columns=features, fill_value=0).astype("float32")
    Xs = pd.DataFrame(scaler.transform(X).astype("float32"), columns=features)
    y_pred = model.predict(Xs)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    col = cm.sum(axis=0)
    tot = col.sum()
    print(f"\n{name} N={len(dfm)}")
    for c, n in sorted(zip(classes, col), key=lambda kv: -kv[1]):
        print(f"  {c:10s} {n:5d} ({n/tot*100:5.1f}%)")
    pd.DataFrame(cm, index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]).to_csv(f"newtasks2/T20_PredDist_{name}.csv")
print("\nSaved -> newtasks2/T20_PredDist_*.csv")
