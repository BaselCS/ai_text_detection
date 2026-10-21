"""Task 1: CNN/DailyMail multi-class OOD evaluation, 7-class and 6-class.

The CNN corpus was generated with *different* checkpoints / providers than the
training corpus.  The evaluation therefore confounds corpus, prompt and
generator shifts.  This script reports both label regimes transparently:

* 7-class: the original training label set, with the slot populated by
  ``deepseek-v4-pro``; the output tables label it ``deepseek-v4-pro``.
* 6-class: the ``deepseek-v4-pro`` slot is formally dropped, and a new
  6-class XGBoost model is trained on the six remaining classes.

Outputs
-------
results/phase3/T1_cnn_multiclass_ood_metrics.csv
results/phase3/T1_cnn_7class_confusion_raw.csv
results/phase3/T1_cnn_7class_confusion_clean.csv
results/phase3/T1_cnn_6class_confusion_raw.csv
results/phase3/T1_cnn_6class_confusion_clean.csv
results/models/Multi_BERT_6class_model.joblib
results/models/Multi_BERT_6class_scaler.joblib
results/models/Multi_BERT_6class_label_encoder.joblib
results/models/Multi_BERT_6class_metadata.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from task5_hygiene_audit_rebuild import map_processed_to_raw

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "results/extra/task2/generated_full1000.csv"
PROC = ROOT / "data/processed/processed_second_corpus_cnn.csv"
TRAIN = ROOT / "data/processed/processed_articles.csv"
MODEL_DIR = ROOT / "results/models"
OUT_METRICS = ROOT / "results/phase3/T1_cnn_multiclass_ood_metrics.csv"

SEED = 999
FEATURES_PATH = MODEL_DIR / "Bin_BERT_features.json"
DISPLAY = {"Yi-Large": "deepseek-v4-pro"}


def clean_mask(proc: pd.DataFrame, raw: pd.DataFrame) -> pd.Series:
    proc = proc.copy()
    proc["Raw_Row"] = map_processed_to_raw(raw, proc).astype(int)
    short = proc["Text"].astype(str).str.len() < 200
    duplicate_prompt = set(raw.loc[233].drop("Prompt").astype(str).tolist())
    duplicate = proc["Text"].astype(str).isin(duplicate_prompt)
    return short | duplicate


def metric_row(name: str, split: str, y_true, y_pred, classes) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Evaluation": name,
        "Split": split,
        "N": int(len(y_true)),
        "Classes": int(len(classes)),
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Macro_Precision": round(float(precision), 4),
        "Macro_Recall": round(float(recall), 4),
        "Macro_F1": round(float(f1), 4),
        "Class_List": "; ".join(DISPLAY.get(c, c) for c in classes),
    }


def predict_existing(df: pd.DataFrame, features, model, scaler, le) -> tuple:
    valid = df[df["Writer"].isin(le.classes_)].copy()
    X = pd.DataFrame(
        scaler.transform(valid.reindex(columns=features, fill_value=0).astype("float32")),
        columns=features,
    )
    pred = model.predict(X)
    y = le.transform(valid["Writer"])
    return valid, y, pred, list(le.classes_)


def train_six_class(features) -> tuple:
    df = pd.read_csv(TRAIN, low_memory=False)
    df = df[df["Writer"] != "Yi-Large"].copy()
    le = LabelEncoder()
    y = le.fit_transform(df["Writer"])
    X = df.reindex(columns=features, fill_value=0).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr).astype("float32"), columns=features)
    X_te_s = pd.DataFrame(scaler.transform(X_te).astype("float32"), columns=features)
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        tree_method="hist",
        device="cuda",
        n_estimators=2000,
        early_stopping_rounds=100,
        learning_rate=0.01,
        max_depth=10,
        random_state=SEED,
        n_jobs=8,
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
    train_acc = float(accuracy_score(y_te, model.predict(X_te_s)))
    joblib.dump(model, MODEL_DIR / "Multi_BERT_6class_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "Multi_BERT_6class_scaler.joblib")
    joblib.dump(le, MODEL_DIR / "Multi_BERT_6class_label_encoder.joblib")
    metadata = {
        "seed": SEED,
        "data_split": "80/20",
        "task_type": "multiclass_6class_drop_yilarge_deepseek",
        "training_rows": int(len(df)),
        "holdout_accuracy": round(train_acc, 4),
        "classes": list(le.classes_),
        "feature_count": len(features),
        "hyperparameters": {
            "objective": "multi:softprob",
            "n_estimators": 2000,
            "early_stopping_rounds": 100,
            "learning_rate": 0.01,
            "max_depth": 10,
            "tree_method": "hist",
            "device": "cuda",
        },
    }
    (MODEL_DIR / "Multi_BERT_6class_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"6-class holdout accuracy: {train_acc:.4f}; training rows={len(df)}", flush=True)
    return model, scaler, le


def main() -> None:
    features = json.load(open(FEATURES_PATH))
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    drop = clean_mask(proc, raw)
    splits = {"raw": proc, "clean": proc[~drop].copy()}

    # ---- 7-class existing model ----
    model7 = joblib.load(MODEL_DIR / "Multi_BERT_model.joblib")
    scaler7 = joblib.load(MODEL_DIR / "Multi_BERT_scaler.joblib")
    le7 = joblib.load(MODEL_DIR / "Multi_BERT_label_encoder.joblib")
    rows = []
    for split, df in splits.items():
        valid, y, pred, classes = predict_existing(df, features, model7, scaler7, le7)
        rows.append(metric_row("7-class (deepseek-v4-pro class)", split, y, pred, classes))
        y_names = [DISPLAY.get(c, c) for c in le7.inverse_transform(y)]
        p_names = [DISPLAY.get(c, c) for c in le7.inverse_transform(pred)]
        order = [DISPLAY.get(c, c) for c in le7.classes_]
        cm = pd.crosstab(pd.Series(y_names, name="True"), pd.Series(p_names, name="Predicted"))
        cm = cm.reindex(index=order, columns=order, fill_value=0)
        cm.to_csv(ROOT / f"results/phase3/T1_cnn_7class_confusion_{split}.csv")

    # ---- 6-class model, Yi-Large / deepseek-v4-pro slot dropped ----
    model6_path = MODEL_DIR / "Multi_BERT_6class_model.joblib"
    scaler6_path = MODEL_DIR / "Multi_BERT_6class_scaler.joblib"
    le6_path = MODEL_DIR / "Multi_BERT_6class_label_encoder.joblib"
    if model6_path.exists() and scaler6_path.exists() and le6_path.exists():
        model6 = joblib.load(model6_path)
        scaler6 = joblib.load(scaler6_path)
        le6 = joblib.load(le6_path)
        print("Loaded existing 6-class model.", flush=True)
    else:
        model6, scaler6, le6 = train_six_class(features)

    for split, df in splits.items():
        eval_df = df[df["Writer"] != "Yi-Large"].copy()
        valid, y, pred, classes = predict_existing(eval_df, features, model6, scaler6, le6)
        rows.append(metric_row("6-class (deepseek-v4-pro dropped)", split, y, pred, classes))
        y_names = [DISPLAY.get(c, c) for c in le6.inverse_transform(y)]
        p_names = [DISPLAY.get(c, c) for c in le6.inverse_transform(pred)]
        order = [DISPLAY.get(c, c) for c in le6.classes_]
        cm = pd.crosstab(pd.Series(y_names, name="True"), pd.Series(p_names, name="Predicted"))
        cm = cm.reindex(index=order, columns=order, fill_value=0)
        cm.to_csv(ROOT / f"results/phase3/T1_cnn_6class_confusion_{split}.csv")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_METRICS, index=False)
    print(out.to_string(index=False), flush=True)
    print(f"Saved {OUT_METRICS}", flush=True)


if __name__ == "__main__":
    main()
