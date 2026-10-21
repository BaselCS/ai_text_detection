"""Reproducible leakage audit for the remediated 667-row NLLB test set.

The audit reports two independent checks against the 51,181-row training
corpus:

1. Exact string matches between the post-NLLB ``Text`` and training ``Text``.
2. Nearest-neighbour TF-IDF cosine similarity, using the same vectoriser
   configuration that produced the reported audit values:
   ``stop_words='english'``, unigrams + bigrams, ``min_df=2``,
   ``max_features=50_000``, fit on the combined train + test corpus.

Outputs
-------
results/phase3/T3_667_leakage_audit.csv
    Per-row exact-match flag and nearest-neighbour cosine.
results/phase3/T3_667_leakage_audit.json
    Summary counts and cosine distribution.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN = ROOT / "data/processed/processed_articles.csv"
TEST = ROOT / "results/phase3/T3_true_test_translated.csv"
OUT_CSV = ROOT / "results/phase3/T3_667_leakage_audit.csv"
OUT_JSON = ROOT / "results/phase3/T3_667_leakage_audit.json"


def main() -> None:
    train = pd.read_csv(TRAIN, usecols=["Text"])
    test = pd.read_csv(TEST, usecols=["Text", "original_text", "is_AI"])

    train_texts = train["Text"].astype(str)
    test_texts = test["Text"].astype(str)
    train_set = set(train_texts)
    exact_match = test_texts.isin(train_set)

    corpus = pd.concat([train_texts, test_texts], ignore_index=True)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
    )
    matrix = vectorizer.fit_transform(corpus)
    train_matrix = matrix[: len(train_texts)]
    test_matrix = matrix[len(train_texts) :]
    similarities = linear_kernel(test_matrix, train_matrix)
    nn_cosine = similarities.max(axis=1)
    nn_index = similarities.argmax(axis=1)

    out = pd.DataFrame(
        {
            "is_AI": test["is_AI"].astype(int),
            "exact_string_match": exact_match.astype(int),
            "nearest_train_row": nn_index.astype(int),
            "nearest_train_cosine": np.round(nn_cosine, 6),
        }
    )
    out.to_csv(OUT_CSV, index=False)

    summary = {
        "n_test_rows": int(len(test)),
        "n_train_rows": int(len(train)),
        "n_exact_string_matches": int(exact_match.sum()),
        "tfidf_config": {
            "stop_words": "english",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_features": 50000,
            "fit": "combined train + test corpus",
        },
        "nearest_neighbour_cosine": {
            "mean": round(float(nn_cosine.mean()), 4),
            "median": round(float(np.median(nn_cosine)), 4),
            "max": round(float(nn_cosine.max()), 4),
            "min": round(float(nn_cosine.min()), 4),
            "n_at_or_above_0_99": int((nn_cosine >= 0.99).sum()),
            "n_at_or_above_0_95": int((nn_cosine >= 0.95).sum()),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {OUT_CSV} and {OUT_JSON}")


if __name__ == "__main__":
    main()
