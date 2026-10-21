"""Task 5 (rebuilt): 65-row hygiene audit and clean-vs-raw metric impact.

The original audit CSV contained six summary records and a truncated list of
the 53 LLaMA rows.  This script reconstructs the full 65-row drop catalog by
mapping every processed CNN row back to its raw prompt row via exact text, then
applies the length and duplicate-prompt filters.

It also reports the metric impact with the binary and multi-class denominators
separated, because the multi-class evaluation excludes Claude.
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "results/extra/task2/generated_full1000.csv"
PROC = ROOT / "data/processed/processed_second_corpus_cnn.csv"
AUDIT = ROOT / "results/phase3/T5_data_hygiene_audit.csv"
SUMMARY = ROOT / "results/phase3/T5_data_hygiene_summary.csv"
IMPACT = ROOT / "results/phase3/T5_hygiene_metric_impact.csv"

WRITER_COLUMNS = {
    "Human": "Human_story",
    "GPT-4o": "gpt-4o-mini",
    "Gemma-2-9B": "gemma_2_9b",
    "Mistral-7B": "mistral_7b",
    "Llama-8B": "llama_3_8b",
    "Qwen-2-72B": "qwen_2_72b",
    "Claude": "claude_haiku",
    "Yi-Large": "deepseek_pro",
}


def map_processed_to_raw(raw: pd.DataFrame, proc: pd.DataFrame) -> pd.Series:
    """Return the raw prompt-row index for every processed row."""
    raw_indices = pd.Series(index=proc.index, dtype="Int64")
    for writer, group in proc.groupby("Writer", sort=False):
        col = WRITER_COLUMNS[writer]
        text_to_rows = {}
        for i, text in enumerate(raw[col].astype(str)):
            text_to_rows.setdefault(text, []).append(i)
        used = {}
        for j, text in group["Text"].astype(str).items():
            candidates = text_to_rows.get(text, [])
            k = used.get(text, 0)
            if k >= len(candidates):
                raise ValueError(f"No raw match for processed row {j} ({writer})")
            raw_indices.loc[j] = candidates[k]
            used[text] = k + 1
    if raw_indices.isna().any():
        raise ValueError("Unmapped processed rows")
    return raw_indices


def classify(row: pd.Series, duplicate_prompt_rows: set) -> tuple[str, str]:
    text = str(row["Text"])
    length = len(text)
    raw_row = int(row["Raw_Row"])
    writer = row["Writer"]
    if text in duplicate_prompt_rows:
        return "Duplicate prompt", "Prompt row 233 duplicates prompt row 128; dropped across all 8 writers"
    if raw_row == 474 and writer == "Llama-8B":
        return "Refusal string", "LLaMA returned a refusal instead of a narrative (also <200 chars)"
    if length < 200:
        if raw_row in (175, 228):
            return "Short article", f"DeepSeek/Yi-Large generation truncated at {length} chars"
        if writer == "Qwen-2-72B" and raw_row in (526, 682):
            return "Short/truncated article", f"Qwen-2.5 generation truncated at {length} chars"
        return "Short article", f"LLaMA generation under 200 chars ({length} chars)"
    return "Other", "Unexpected flagged row"


def main():
    os.makedirs(ROOT / "results/phase3", exist_ok=True)
    raw = pd.read_csv(RAW)
    proc = pd.read_csv(PROC)
    proc = proc.copy()
    proc["Raw_Row"] = map_processed_to_raw(raw, proc).astype(int)

    duplicate_prompt_rows = set(raw.loc[233].drop("Prompt").astype(str).tolist())
    short_mask = proc["Text"].astype(str).str.len() < 200
    duplicate_mask = proc["Text"].astype(str).isin(duplicate_prompt_rows)
    drop_mask = short_mask | duplicate_mask

    dropped = proc[drop_mask].copy()
    records = []
    for _, row in dropped.iterrows():
        category, issue = classify(row, duplicate_prompt_rows)
        records.append(
            {
                "Writer": row["Writer"],
                "Raw_Row": int(row["Raw_Row"]),
                "Category": category,
                "Issue_Description": issue,
                "Character_Length": len(str(row["Text"])),
                "Snippet": str(row["Text"])[:200].replace("\n", " "),
                "Action_Taken": "Dropped from clean evaluation split",
                "Included_In_Clean_Filter": "Yes",
            }
        )
    audit = pd.DataFrame(records).sort_values(["Writer", "Raw_Row"]).reset_index(drop=True)
    if len(audit) != 65:
        raise AssertionError(f"Expected 65 dropped rows, found {len(audit)}")
    audit.to_csv(AUDIT, index=False)

    summary_rows = [
        {"Category": "Short article (<200 chars)", "Count": int(short_mask.sum())},
        {"Category": "Duplicate prompt (row 233)", "Count": int(duplicate_mask.sum())},
        {"Category": "Total dropped", "Count": int(drop_mask.sum())},
        {"Category": "Raw CNN rows", "Count": int(len(proc))},
        {"Category": "Clean CNN rows", "Count": int((~drop_mask).sum())},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY, index=False)

    # ---- Metric impact, with explicit denominators ----
    features = json.load(open(ROOT / "results/models/Bin_BERT_features.json"))
    bin_model = joblib.load(ROOT / "results/models/Bin_BERT_model.joblib")
    bin_scaler = joblib.load(ROOT / "results/models/Bin_BERT_scaler.joblib")
    mul_model = joblib.load(ROOT / "results/models/Multi_BERT_model.joblib")
    mul_scaler = joblib.load(ROOT / "results/models/Multi_BERT_scaler.joblib")
    mul_le = joblib.load(ROOT / "results/models/Multi_BERT_label_encoder.joblib")

    clean = proc[~drop_mask].copy()

    def _feat(df, label):
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise ValueError(f"{label}: {len(missing)} missing feature column(s) {missing[:8]}. "
                             "Regenerate the processed file; do not zero-fill.")
        return df[features].astype("float32")

    def binary_metrics(df):
        X = pd.DataFrame(
            bin_scaler.transform(_feat(df, "hygiene/binary")),
            columns=features,
        )
        pred = bin_model.predict(X)
        y = df["is_AI"].astype(int)
        return (
            float(accuracy_score(y, pred)),
            float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[2]),
            len(df),
        )

    def multi_metrics(df):
        valid = df[df["Writer"].isin(mul_le.classes_)].copy()
        X = pd.DataFrame(
            mul_scaler.transform(_feat(valid, "hygiene/multi")),
            columns=features,
        )
        pred = mul_model.predict(X)
        y = mul_le.transform(valid["Writer"])
        return (
            float(accuracy_score(y, pred)),
            float(precision_recall_fscore_support(y, pred, average="macro", zero_division=0)[2]),
            len(valid),
        )

    rows = []
    for task, func in [("Binary", binary_metrics), ("Multi-Class", multi_metrics)]:
        raw_acc, raw_f1, raw_n = func(proc)
        clean_acc, clean_f1, clean_n = func(clean)
        rows.extend(
            [
                {
                    "Task": task,
                    "Metric": "Accuracy",
                    "Raw_N": raw_n,
                    "Clean_N": clean_n,
                    "Raw_Value": round(raw_acc, 4),
                    "Clean_Value": round(clean_acc, 4),
                    "Delta": round(clean_acc - raw_acc, 4),
                },
                {
                    "Task": task,
                    "Metric": "Macro_F1",
                    "Raw_N": raw_n,
                    "Clean_N": clean_n,
                    "Raw_Value": round(raw_f1, 4),
                    "Clean_Value": round(clean_f1, 4),
                    "Delta": round(round(clean_f1, 4) - round(raw_f1, 4), 4),
                },
            ]
        )
    impact = pd.DataFrame(rows)
    impact.to_csv(IMPACT, index=False)

    # Row 915 is a source-level truncation that is not present in the processed
    # corpus (Yi-Large has 999 rows); keep it in a notes file so the audit is
    # explicit about why it is not one of the 65 dropped rows.
    notes = pd.DataFrame(
        [
            {
                "Writer": "Yi-Large (source: deepseek_pro)",
                "Raw_Row": 915,
                "Category": "Source-level truncation, not present in processed corpus",
                "Issue_Description": "Raw deepseek_pro row 915 is 'READING,' (8 chars), but processed Yi-Large has 999 rows and does not contain it.",
                "Included_In_Clean_Filter": "No",
            }
        ]
    )
    notes.to_csv(ROOT / "results/phase3/T5_data_hygiene_notes.csv", index=False)

    print(audit.groupby(["Category", "Writer"]).size().to_string())
    print("\nSummary:\n", summary.to_string(index=False))
    print("\nImpact:\n", impact.to_string(index=False))
    print(f"\nSaved {AUDIT}, {SUMMARY}, {IMPACT}")


if __name__ == "__main__":
    main()
