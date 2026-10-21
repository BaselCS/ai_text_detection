"""Reproducible shallow surface baseline on the 666-row deduplicated test set.

Inputs
------
results/phase3/T3_true_test_translated.csv
    ``original_text`` is the pre-NLLB text; ``Text`` is the post-NLLB text.

Outputs
-------
results/phase3/T3_surface_artifacts_comparison.csv
    Pre/post surface-artifact means, standard deviations, paired shifts and
    paired t-test p-values for All / Human / AI subsets.
results/phase3/T3_shallow_classifier_667_metrics.csv
    Random-forest shallow baselines (formatting-only and extended surface
    features) under (a) the natural 611 AI / 55 Human imbalance and (b) a
    balanced 55 AI / 55 Human diagnostic sample.

The classifier is deliberately non-neural and is not a replacement for the
RoBERTa evaluation.  It quantifies how much of the held-out signal is
available from surface formatting alone.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "results/phase3/T3_true_test_translated.csv"
ARTIFACTS = ROOT / "results/phase3/T3_surface_artifacts_comparison.csv"
METRICS = ROOT / "results/phase3/T3_shallow_classifier_667_metrics.csv"

RANDOM_STATE = 42
N_ESTIMATORS = 500
MAX_DEPTH = 4

SURFACE_PATTERNS = {
    "has_markdown_bold": r"\*\*",
    "has_title_tag": r"\bTitle:",
    "has_by_tag": r"\bBy:",
    "has_date_tag": r"\bDate:",
    "has_bullet": r"(?m)^[-*\u2022]\s",
}


def surface_features(text: pd.Series) -> pd.DataFrame:
    text = text.astype(str)
    words = text.str.split()
    quote_chars = ['"', "'", "\u201c", "\u201d", "\u2018", "\u2019"]
    dash_chars = ["-", "\u2013", "\u2014"]
    out = pd.DataFrame(
        {
            "char_len": text.str.len(),
            "word_count": words.str.len(),
            "sent_count": text.str.count(r"[.!?]+") + 1,
            "avg_word_len": text.str.len() / words.str.len().clip(lower=1),
            "quote_counts": sum(text.str.count(c) for c in quote_chars),
            "dash_counts": sum(text.str.count(c) for c in dash_chars),
            "colon_counts": text.str.count(":"),
            "paragraph_count": text.str.count(r"\n\s*\n"),
            "exclamation_count": text.str.count("!"),
            "question_count": text.str.count(r"\?"),
            "digit_ratio": text.apply(lambda x: sum(c.isdigit() for c in x) / max(len(x), 1)),
            "upper_ratio": text.apply(lambda x: sum(c.isupper() for c in x) / max(len(x), 1)),
        }
    )
    for name, pattern in SURFACE_PATTERNS.items():
        out[name] = text.str.contains(pattern, case=False, regex=True).astype(int)
    out["has_markdown_heading"] = text.str.contains(r"(?m)^#{1,6}\s", regex=True).astype(int)
    return out


def artifact_table(pre: pd.DataFrame, post: pd.DataFrame, is_ai: pd.Series) -> pd.DataFrame:
    subsets = [
        ("All_Test_Articles", pd.Series(True, index=is_ai.index)),
        ("Human_Subset", is_ai == 0),
        ("AI_Subset", is_ai == 1),
    ]
    rows = []
    for subset_name, mask in subsets:
        for feature in pre.columns:
            a = pre.loc[mask, feature].astype(float)
            b = post.loc[mask, feature].astype(float)
            delta = b.mean() - a.mean()
            relative = (100 * delta / a.mean()) if a.mean() != 0 else np.nan
            if len(a) and len(b) and not (a.nunique() == 1 and b.nunique() == 1 and a.iloc[0] == b.iloc[0]):
                p_value = float(ttest_rel(b, a, nan_policy="omit").pvalue)
            else:
                p_value = 1.0
            rows.append(
                {
                    "Subset": subset_name,
                    "N_Articles": int(mask.sum()),
                    "Feature": feature,
                    "Pre_Translation_Mean": round(float(a.mean()), 4),
                    "Pre_Translation_Std": round(float(a.std()), 4),
                    "Post_NLLB_Mean": round(float(b.mean()), 4),
                    "Post_NLLB_Std": round(float(b.std()), 4),
                    "Mean_Shift_Delta": round(float(delta), 4),
                    "Relative_Shift_Pct": round(float(relative), 2) if pd.notna(relative) else np.nan,
                    "p_value_paired_ttest": round(p_value, 6),
                }
            )
    return pd.DataFrame(rows)


def evaluate(X: pd.DataFrame, y: pd.Series, protocol: str) -> dict:
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    return {
        "Protocol": protocol,
        "N": int(len(y)),
        "N_Human": int((y == 0).sum()),
        "N_AI": int((y == 1).sum()),
        "Accuracy": round(float(accuracy_score(y, pred)), 4),
        "Balanced_Accuracy": round(
            float(np.mean([accuracy_score(y[y == c], pred[y == c]) for c in [0, 1]])), 4
        ),
        "Macro_F1": round(float(f1_score(y, pred, average="macro")), 4),
        "Features": ", ".join(X.columns),
        "Model": f"RandomForestClassifier(max_depth={MAX_DEPTH}, n_estimators={N_ESTIMATORS}, seed={RANDOM_STATE})",
        "Validation": f"5-fold stratified CV (seed={RANDOM_STATE})",
    }


def main() -> None:
    df = pd.read_csv(INPUT, usecols=["Text", "original_text", "is_AI"])
    if df[["Text", "original_text"]].isna().any().any():
        raise ValueError("T3_true_test_translated.csv contains missing text")

    pre = surface_features(df["original_text"])
    post = surface_features(df["Text"])
    artifact_features = [
        "char_len", "word_count", "sent_count", "avg_word_len",
        "quote_counts", "dash_counts", "colon_counts",
        "has_markdown_bold", "has_title_tag", "has_by_tag",
        "has_date_tag", "has_bullet",
    ]
    artifacts = artifact_table(pre[artifact_features], post[artifact_features], df["is_AI"].astype(int))
    artifacts.to_csv(ARTIFACTS, index=False)
    print(artifacts.to_string(index=False))

    formatting = ["has_markdown_bold", "has_title_tag", "has_date_tag", "has_bullet"]
    extended = formatting + [
        "has_by_tag",
        "has_markdown_heading",
        "char_len",
        "word_count",
        "sent_count",
        "paragraph_count",
        "exclamation_count",
        "question_count",
        "avg_word_len",
        "quote_counts",
        "dash_counts",
        "colon_counts",
        "digit_ratio",
        "upper_ratio",
    ]

    y = df["is_AI"].astype(int)
    rows = [
        evaluate(post[formatting], y, "Formatting-only | natural 666 (611 AI / 55 Human)"),
        evaluate(post[extended], y, "Extended surface | natural 666 (611 AI / 55 Human)"),
    ]
    human = df[df["is_AI"] == 0]
    ai_sample = df[df["is_AI"] == 1].sample(n=len(human), random_state=RANDOM_STATE)
    balanced = pd.concat([human, ai_sample])
    yb = balanced["is_AI"].astype(int)
    post_balanced = post.loc[balanced.index]
    rows.extend(
        [
            evaluate(post_balanced[formatting], yb, "Formatting-only | balanced 110 (55 AI / 55 Human)"),
            evaluate(post_balanced[extended], yb, "Extended surface | balanced 110 (55 AI / 55 Human)"),
        ]
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS, index=False)
    print("\n" + metrics.to_string(index=False))
    print(f"\nSaved {ARTIFACTS} and {METRICS}")


if __name__ == "__main__":
    main()
