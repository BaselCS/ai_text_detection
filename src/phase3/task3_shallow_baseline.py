"""Reproducible Task 3 surface-artifact audit and shallow baseline.

This script replaces the unverifiable historical 94.97% / 89.71% claim with a
self-contained, seeded calculation.  It reports:

1. Surface-artifact prevalence on the historical, unperturbed 1,729-row
   translation file (``processed_translation.csv``), split by Human/AI.
2. A formatting-only classifier (bold, Title:, Date:, bullets).
3. An extended surface-feature classifier (formatting + length/punctuation).

The classifiers are deliberately non-neural.  Metrics use 5-fold stratified
cross-validation; no result depends on a single random train/test split.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT = ROOT / "data/processed/processed_translation.csv"
ARTIFACTS = ROOT / "results/phase3/T3_surface_artifacts_historical.csv"
METRICS = ROOT / "results/phase3/T3_shallow_classifier_metrics.csv"

# Exact, reproducible regexes.  Date: is intentionally the literal string
# "Date:"; it occurs in 130/1,482 AI rows = 8.77%, i.e. 8.8%.
PATTERNS = {
    "Markdown_Bold": r"\*\*",
    "Title_Tag": r"Title:",
    "Date_Tag": r"Date:",
    "Byline_Tag": r"By:",
    "Markdown_Heading": r"(?m)^#{1,6}\s",
    "Bullet_Point": r"\n[-*\u2022]\s",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["Text"].astype(str)
    out = df.copy()
    for name, pattern in PATTERNS.items():
        out[name] = text.str.contains(pattern, regex=True, na=False).astype(int)

    words = text.str.findall(r"\w+")
    out["Character_Length"] = text.str.len()
    out["Word_Count"] = words.str.len()
    out["Sentence_Count"] = text.str.count(r"[.!?]+")
    out["Paragraph_Count"] = text.str.count(r"\n\s*\n")
    out["Exclamation_Count"] = text.str.count("!")
    out["Question_Count"] = text.str.count(r"\?")
    out["Quote_Count"] = text.str.count('"')
    out["Colon_Count"] = text.str.count(":")
    out["Dash_Count"] = text.str.count("-")
    out["Digit_Ratio"] = text.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    out["Upper_Ratio"] = text.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    return out


def prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in PATTERNS:
        ai = df[df["is_AI"] == 1][name]
        human = df[df["is_AI"] == 0][name]
        rows.append(
            {
                "Artifact": name,
                "AI_Count": int(ai.sum()),
                "AI_N": int(len(ai)),
                "AI_Percent": round(100 * ai.mean(), 2),
                "Human_Count": int(human.sum()),
                "Human_N": int(len(human)),
                "Human_Percent": round(100 * human.mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def evaluate(X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        class_weight=None,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1)
    return {
        "Protocol": label,
        "N": int(len(y)),
        "N_Human": int((y == 0).sum()),
        "N_AI": int((y == 1).sum()),
        "Accuracy": round(float(accuracy_score(y, pred)), 4),
        "Macro_F1": round(float(f1_score(y, pred, average="macro")), 4),
        "Features": ", ".join(X.columns),
        "Model": "RandomForestClassifier(max_depth=4, n_estimators=500, seed=42)",
        "Validation": "5-fold stratified CV (seed=42)",
    }


def main():
    df = pd.read_csv(INPUT, usecols=["Text", "Writer", "is_AI"])
    df = add_features(df)

    art = prevalence(df)
    art.to_csv(ARTIFACTS, index=False)
    print(art.to_string(index=False))

    formatting = ["Markdown_Bold", "Title_Tag", "Date_Tag", "Bullet_Point"]
    extended = formatting + [
        "Byline_Tag",
        "Markdown_Heading",
        "Character_Length",
        "Word_Count",
        "Sentence_Count",
        "Paragraph_Count",
        "Exclamation_Count",
        "Question_Count",
        "Quote_Count",
        "Colon_Count",
        "Dash_Count",
        "Digit_Ratio",
        "Upper_Ratio",
    ]

    y = df["is_AI"].astype(int)
    rows = [
        evaluate(df[formatting], y, "Formatting-only | full 1,729"),
        evaluate(df[extended], y, "Extended surface | full 1,729"),
    ]

    # Balanced diagnostic: all 247 Human + 247 AI sampled with seed 42.
    ai_sample = df[df["is_AI"] == 1].sample(n=int((df["is_AI"] == 0).sum()), random_state=42)
    balanced = pd.concat([df[df["is_AI"] == 0], ai_sample])
    yb = balanced["is_AI"].astype(int)
    rows.extend(
        [
            evaluate(balanced[formatting], yb, "Formatting-only | balanced 494"),
            evaluate(balanced[extended], yb, "Extended surface | balanced 494"),
        ]
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(METRICS, index=False)
    print("\n" + metrics.to_string(index=False))
    print(f"\nSaved {ARTIFACTS} and {METRICS}")


if __name__ == "__main__":
    main()
