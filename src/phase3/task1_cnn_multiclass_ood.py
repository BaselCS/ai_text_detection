"""Task 1: CNN/DailyMail OOD evaluation, 7-class and 6-class, plus binary.

The CNN corpus was generated with *different* checkpoints / providers than the
training corpus.  The evaluation therefore confounds corpus, prompt and
generator shifts.  This script reports both label regimes transparently:

* 7-class: the original training label set, with the slot populated by
  ``deepseek-v4-pro``; every column is labelled ``actual-generator (slot)``.
* 6-class: the ``deepseek-v4-pro`` slot is formally dropped, and a new
  6-class XGBoost model is trained on the six remaining classes.
* binary: the Phase-1 binary detector, scored on all eight writers, with
  Wilson intervals for accuracy and per-class recall and a bootstrap interval
  for balanced accuracy.

Outputs
-------
results/phase3/T1_cnn_multiclass_ood_metrics.csv
results/phase3/T1_cnn_binary_and_multiclass_ood_metrics.csv
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
OUT_METRICS_EXT = ROOT / "results/phase3/T1_cnn_binary_and_multiclass_ood_metrics.csv"

SEED = 999
FEATURES_PATH = MODEL_DIR / "Bin_BERT_features.json"
Z95 = 1.959963984540054

# Every CNN column is named by the checkpoint that actually generated it, with
# the Phase-1 training slot in parentheses.  See docs/06_cnn_provenance.md.
DISPLAY = {
    "GPT-4o": "gpt-4o-mini (GPT-4o slot)",
    "Gemma-2-9B": "gemma-3-12b-it (Gemma-2-9B slot)",
    "Llama-8B": "llama-3.1-8b (Llama-8B slot)",
    "Mistral-7B": "mistral-nemo (Mistral-7B slot)",
    "Qwen-2-72B": "qwen-2.5-72b (Qwen-2-72B slot)",
    "Yi-Large": "deepseek-v4-pro (Yi-Large slot)",
}


def load_features():
    """Feature list for the Multi_BERT model, asserted identical to the binary."""
    with open(FEATURES_PATH) as f:
        bin_features = json.load(f)
    with open(MODEL_DIR / "Multi_BERT_metadata.json") as f:
        multi_features = json.load(f)["feature_list"]
    if bin_features != multi_features:
        raise AssertionError("Binary and multi-class feature lists have diverged.")
    return multi_features


def assert_schema(df, features, name):
    """Fail loudly on missing feature columns instead of zero-filling them."""
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"{name}: missing {len(missing)} feature(s) {missing}. "
            "Regenerate the processed file; do not zero-fill."
        )
    return df[features]


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


def wilson(k: int, n: int) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + Z95 ** 2 / n
    centre = p + Z95 ** 2 / (2 * n)
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n))
    return (float((centre - half) / denom), float((centre + half) / denom))


def bootstrap_balanced(y_true, y_pred, n_boot: int = 1000, seed: int = SEED) -> tuple:
    """Percentile bootstrap interval for balanced accuracy (macro recall)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    n = len(y_true)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y_true[idx], y_pred[idx]
        recs = [(pb[yb == lab] == lab).mean() for lab in labels if (yb == lab).any()]
        vals[b] = float(np.mean(recs)) if recs else np.nan
    return (float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5)))


def augment_row(row: dict, y_true, y_pred) -> dict:
    """Add accuracy CI, balanced accuracy, per-class recall and majority baseline."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    lo, hi = wilson(correct, n)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=np.unique(y_true), zero_division=0
    )[1]
    bal = float(np.mean(per_class))
    b_lo, b_hi = bootstrap_balanced(y_true, y_pred)
    counts = pd.Series(y_true).value_counts()
    row.update({
        "N_Correct": correct,
        "Accuracy_CI_Lower": round(lo, 4),
        "Accuracy_CI_Higher": round(hi, 4),
        "Balanced_Accuracy": round(bal, 4),
        "Balanced_Accuracy_CI_Lower": round(b_lo, 4),
        "Balanced_Accuracy_CI_Higher": round(b_hi, 4),
        "Majority_Baseline": round(float(counts.max() / n), 4),
        "Majority_Class": str(counts.idxmax()),
    })
    if row["Classes"] == 2:
        for label in (0, 1):
            m = y_true == label
            k = int((y_pred[m] == label).sum())
            r_lo, r_hi = wilson(k, int(m.sum()))
            tag = "Human" if label == 0 else "AI"
            row[f"{tag}_Recall"] = round(k / max(int(m.sum()), 1), 4)
            row[f"{tag}_Recall_CI_Lower"] = round(r_lo, 4)
            row[f"{tag}_Recall_CI_Higher"] = round(r_hi, 4)
    return row


def predict_existing(df: pd.DataFrame, features, model, scaler, le) -> tuple:
    valid = df[df["Writer"].isin(le.classes_)].copy()
    X = pd.DataFrame(
        scaler.transform(assert_schema(valid, features, "CNN corpus").astype("float32")),
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
    X = assert_schema(df, features, "training corpus").astype("float32")
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
    features = load_features()
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    drop = clean_mask(proc, raw)
    splits = {"raw": proc, "clean": proc[~drop].copy()}

    rows, ext_rows = [], []

    # ---- binary detector on all eight writers ----
    bin_model = joblib.load(MODEL_DIR / "Bin_BERT_model.joblib")
    bin_scaler = joblib.load(MODEL_DIR / "Bin_BERT_scaler.joblib")
    for split, df in splits.items():
        X = pd.DataFrame(
            bin_scaler.transform(assert_schema(df, features, "CNN corpus").astype("float32")),
            columns=features,
        )
        pred = bin_model.predict(X).astype(int)
        y = df["is_AI"].astype(int).to_numpy()
        row = metric_row("Binary (AI vs Human)", split, y, pred, ["Human", "AI"])
        ext_rows.append(augment_row(dict(row), y, pred))
        cm = pd.crosstab(pd.Series(y, name="True"), pd.Series(pred, name="Predicted"))
        cm = cm.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        cm.index = ["Human", "AI"]
        cm.columns = ["Human", "AI"]
        cm.to_csv(ROOT / f"results/phase3/T1_cnn_binary_confusion_{split}.csv")

    # ---- 7-class existing model ----
    model7 = joblib.load(MODEL_DIR / "Multi_BERT_model.joblib")
    scaler7 = joblib.load(MODEL_DIR / "Multi_BERT_scaler.joblib")
    le7 = joblib.load(MODEL_DIR / "Multi_BERT_label_encoder.joblib")
    for split, df in splits.items():
        valid, y, pred, classes = predict_existing(df, features, model7, scaler7, le7)
        row = metric_row("7-class (deepseek-v4-pro class)", split, y, pred, classes)
        rows.append(row)
        ext_rows.append(augment_row(dict(row), y, pred))
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
        row = metric_row("6-class (deepseek-v4-pro dropped)", split, y, pred, classes)
        rows.append(row)
        ext_rows.append(augment_row(dict(row), y, pred))
        y_names = [DISPLAY.get(c, c) for c in le6.inverse_transform(y)]
        p_names = [DISPLAY.get(c, c) for c in le6.inverse_transform(pred)]
        order = [DISPLAY.get(c, c) for c in le6.classes_]
        cm = pd.crosstab(pd.Series(y_names, name="True"), pd.Series(p_names, name="Predicted"))
        cm = cm.reindex(index=order, columns=order, fill_value=0)
        cm.to_csv(ROOT / f"results/phase3/T1_cnn_6class_confusion_{split}.csv")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_METRICS, index=False)

    ext = pd.DataFrame(ext_rows)
    front = [
        "Evaluation", "Split", "N", "Classes", "Accuracy",
        "Accuracy_CI_Lower", "Accuracy_CI_Higher",
        "Balanced_Accuracy", "Balanced_Accuracy_CI_Lower", "Balanced_Accuracy_CI_Higher",
        "Macro_Precision", "Macro_Recall", "Macro_F1",
        "Human_Recall", "Human_Recall_CI_Lower", "Human_Recall_CI_Higher",
        "AI_Recall", "AI_Recall_CI_Lower", "AI_Recall_CI_Higher",
        "Majority_Baseline", "Majority_Class", "N_Correct", "Class_List",
    ]
    ext = ext[[c for c in front if c in ext.columns]]
    ext.to_csv(OUT_METRICS_EXT, index=False)

    print(out.to_string(index=False), flush=True)
    print("\n" + ext.to_string(index=False), flush=True)
    print(f"Saved {OUT_METRICS}", flush=True)
    print(f"Saved {OUT_METRICS_EXT}", flush=True)


if __name__ == "__main__":
    main()
