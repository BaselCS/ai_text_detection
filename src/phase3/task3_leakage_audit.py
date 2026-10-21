"""Task 3 leakage audit with a frozen, deterministic duplicate-text mapping.

The historical audit used ``dict`` insertion order that silently kept the *last*
occurrence of duplicated article texts.  This script freezes the audit rule to
the *first* occurrence in ``processed_articles.csv`` and regenerates the leakage
summary and detailed match table from that single rule.  The separate 3,064-row
NLLB evaluation archive uses an explicit last-occurrence rule; its pre-dedup
sizes are 667/580/610/615/588 and its source-text-deduplicated evaluation sizes
are 666/579/610/615/586.

The 1,729-row file audited here is the historical pre-subsampling evaluation
file ``data/processed/processed_translation.csv``.  It is *not* the remediated
666-row NLLB test set; the report must keep those provenance statements
separate.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
ARTICLES = ROOT / "data/processed/processed_articles.csv"
TRANSLATED = ROOT / "data/processed/processed_translation.csv"
SUMMARY = ROOT / "results/phase3/T3_leakage_audit_summary.csv"
DETAILED = ROOT / "results/phase3/T3_leakage_detailed_matches.csv"
SEEDS = [42, 123, 999, 2024, 8080]


def build_first_occurrence_map(texts: pd.Series) -> dict:
    """Map every exact text to its FIRST row index.

    Frozen rule for the 1,729-row historical audit file.  This reproduces the
    documented Seed-42 counts (1,205 train + 152 validation + 372 test).  The
    separate 3,064-row NLLB evaluation archive uses the last-occurrence rule in
    ``task3_remediation.py``/``task3_dual_settings.py`` to reproduce the
    pre-dedup 667/580/610/615/588 sizes (666/579/610/615/586 after source-text
    deduplication).  Both rules are explicit and deterministic; neither relies
    on an accidental dict comprehension.
    """
    mapping = {}
    for idx, text in enumerate(texts):
        mapping.setdefault(text, idx)
    return mapping


def get_splits_for_seed(df_articles: pd.DataFrame, seed: int):
    stratify_col = df_articles["is_AI"].astype(int)
    train_val_df, test_df = train_test_split(
        df_articles, test_size=0.20, random_state=seed, stratify=stratify_col
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.125,
        random_state=seed,
        stratify=train_val_df["is_AI"].astype(int),
    )
    return set(train_df.index), set(val_df.index), set(test_df.index)


def main():
    articles = pd.read_csv(ARTICLES, usecols=["Text", "Writer", "is_AI"])
    translated = pd.read_csv(TRANSLATED, usecols=["Text", "Writer", "is_AI"])

    text_to_idx = build_first_occurrence_map(articles["Text"])
    matched_indices = [text_to_idx.get(t, -1) for t in translated["Text"]]
    translated = translated.copy()
    translated["matched_orig_idx"] = matched_indices
    translated["processed_translation_row"] = np.arange(len(translated))

    unmatched = int((translated["matched_orig_idx"] < 0).sum())
    if unmatched:
        raise ValueError(f"{unmatched} translated texts have no exact article match")

    matched = translated.merge(
        articles.reset_index().rename(
            columns={
                "index": "matched_orig_idx",
                "Text": "matched_orig_text",
                "Writer": "matched_orig_writer",
                "is_AI": "matched_orig_is_AI",
            }
        ),
        on="matched_orig_idx",
        how="left",
        validate="many_to_one",
    )
    matched["match_similarity"] = 1.0

    detailed_cols = [
        "processed_translation_row",
        "Writer",
        "is_AI",
        "matched_orig_writer",
        "matched_orig_is_AI",
        "match_similarity",
        "matched_orig_idx",
    ]
    matched[detailed_cols].to_csv(DETAILED, index=False)

    summary_rows = []
    for seed in SEEDS:
        train_idx, val_idx, test_idx = get_splits_for_seed(articles, seed)
        in_train = matched["matched_orig_idx"].isin(train_idx)
        in_val = matched["matched_orig_idx"].isin(val_idx)
        in_test = matched["matched_orig_idx"].isin(test_idx)
        total = len(matched)
        leak = int((in_train | in_val).sum())
        summary_rows.append(
            {
                "Seed": seed,
                "Total_Translated": total,
                "In_Train_Count": int(in_train.sum()),
                "In_Train_Pct": round(100 * in_train.mean(), 2),
                "In_Val_Count": int(in_val.sum()),
                "In_Val_Pct": round(100 * in_val.mean(), 2),
                "In_Test_Count": int(in_test.sum()),
                "In_Test_Pct": round(100 * in_test.mean(), 2),
                "Leakage_Into_Training_Pct": round(100 * leak / total, 2),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY, index=False)

    metadata = {
        "mapping_rule": "first occurrence in processed_articles.csv (frozen audit rule)",
        "audited_file": str(TRANSLATED.relative_to(ROOT)),
        "audited_rows": int(len(matched)),
        "unmatched_rows": unmatched,
        "writer_mismatches": int((matched["Writer"] != matched["matched_orig_writer"]).sum()),
        "label_mismatches": int((matched["is_AI"] != matched["matched_orig_is_AI"]).sum()),
        "seed42": summary[summary.Seed == 42].iloc[0].to_dict(),
    }
    with open(ROOT / "results/phase3/T3_leakage_audit_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(summary.to_string(index=False))
    print(f"\nSaved {SUMMARY} and {DETAILED}")


if __name__ == "__main__":
    main()
