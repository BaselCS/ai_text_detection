"""Task 6 (rebuilt): Document-Level Logistic Regression on BERTScore.

This version fixes the row-alignment bug in the original pipeline.  Raw and
processed paraphrase files are joined *only* on exact normalized
(Paraphrased_Text, Writer) pairs; no positional ``iloc`` zipping is used.
BERTScore is recomputed for every processed row against the matched raw
``Original_Text`` so that every BERTScore, prediction, and cluster ID refers to
the same document.

Outputs
-------
results/phase3/T6_document_level_bertscore_cache.csv
results/phase3/T6_logistic_regression_bertscore.csv

The regression CSV contains both the baseline specification (paraphraser
controls) and the writer-fixed-effects specification, for full and restricted
regimes.  Binary AI-only and Human-only subsets are included as diagnostics.
"""

import hashlib
import json
import os
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from bert_score import score as bert_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_FILE = ROOT / "results/phase3/T6_document_level_bertscore_cache.csv"
OUT_RESULTS_CSV = ROOT / "results/phase3/T6_logistic_regression_bertscore.csv"
SUMMARY_JSON = ROOT / "results/phase3/T6_rebuild_summary.json"

DATA_CONFIGS = [
    {
        "name": "Gemma-2-9B",
        "raw": "data/processed/raw_paraphrased_gemma2.csv",
        "processed": "data/processed/processed_paraphrased_gemma2.csv",
        "is_reconstructed": False,
    },
    {
        "name": "Qwen-2-72B",
        "raw": "data/processed/raw_paraphrased_qwen2.csv",
        "processed": "data/processed/processed_paraphrased_qwen2.csv",
        "is_reconstructed": False,
    },
    {
        "name": "LLaMA-3",
        "raw": "data/processed/raw_paraphrased_llama3.csv",
        "processed": "data/processed/processed_paraphrased_llama3.csv",
        "is_reconstructed": False,
    },
    {
        "name": "Mistral-7B",
        "raw": "data/processed/raw_paraphrased_mistral7b.csv",
        "processed": "data/processed/processed_paraphrased_mistral7b.csv",
        "is_reconstructed": False,
    },
    {
        "name": "GPT-4o-mini",
        "raw": "data/processed/raw_paraphrased_gpt4o_mini.csv",
        "processed": "data/processed/processed_paraphrased_gpt4o_mini.csv",
        "is_reconstructed": False,
    },
    {
        "name": "Gemma-27B",
        "raw": "data/processed/raw_paraphrased_gemma27b.csv",
        "processed": "data/processed/processed_paraphrased_articles.csv",
        "is_reconstructed": True,
    },
]


def normalize_text(value) -> str:
    """Whitespace-normalized text used only for exact joining."""
    return re.sub(r"\s+", " ", str(value)).strip()


def article_hash(text: str) -> str:
    """Canonical cluster ID: MD5 of the first 300 normalized characters."""
    clean = " ".join(str(text).strip().split())[:300]
    return hashlib.md5(clean.encode("utf-8")).hexdigest()[:12]


def build_exact_join(raw: pd.DataFrame, proc: pd.DataFrame, name: str) -> pd.DataFrame:
    """Attach raw Original_Text to each processed row by exact text+writer join.

    Duplicate (text, writer) keys are matched in order of occurrence, which is
    deterministic and preserves the one-to-one pairing for the two duplicate
    Gemma-27B rows.
    """
    raw = raw.copy()
    proc = proc.copy()
    raw["writer_norm"] = raw["Writer"].astype(str).str.replace("_paraphrased", "", regex=False)
    proc["writer_norm"] = proc["Writer"].astype(str)
    raw["text_norm"] = raw["Paraphrased_Text"].map(normalize_text)
    proc["text_norm"] = proc["Text"].map(normalize_text)

    # The raw files contain three rows per writer where Original_Text is NaN.
    # They are meta-refusals to an empty prompt. Recover the missing reference
    # from the same prompt index in a parallel writer block; the prompt text is
    # identical across writers, so this is an exact reference recovery rather
    # than a fuzzy guess.
    raw["prompt_idx"] = raw.groupby("Writer", sort=False).cumcount()
    prompt_original = (
        raw.dropna(subset=["Original_Text"])
        .groupby("prompt_idx")["Original_Text"]
        .first()
        .to_dict()
    )

    key_to_raw = {}
    for i, row in raw.iterrows():
        key_to_raw.setdefault((row["text_norm"], row["writer_norm"]), []).append(i)

    used = {}
    attached = []
    for j, row in proc.iterrows():
        key = (row["text_norm"], row["writer_norm"])
        candidates = key_to_raw.get(key, [])
        if not candidates:
            raise ValueError(f"{name}: no exact raw match for processed row {j} ({row['writer_norm']})")
        k = used.get(key, 0)
        if k >= len(candidates):
            raise ValueError(f"{name}: duplicate key exhausted at row {j}: {key[1]}")
        raw_idx = candidates[k]
        used[key] = k + 1
        original = raw.loc[raw_idx, "Original_Text"]
        match_type = "exact_text_writer"
        if pd.isna(original):
            original = prompt_original.get(raw.loc[raw_idx, "prompt_idx"])
            match_type = "recovered_prompt_reference"
        attached.append(
            {
                "raw_index": int(raw_idx),
                "original_text": original,
                "match_type": match_type if pd.notna(original) else "missing_original_text",
            }
        )

    out = pd.concat([proc.reset_index(drop=True), pd.DataFrame(attached)], axis=1)
    if len(out) != len(proc):
        raise AssertionError(f"{name}: join changed row count")
    return out


def load_models():
    with open(ROOT / "results/models/Bin_BERT_features.json") as f:
        features = json.load(f)

    bin_model = joblib.load(ROOT / "results/models/Bin_BERT_model.joblib")
    bin_scaler = joblib.load(ROOT / "results/models/Bin_BERT_scaler.joblib")

    mul_model = joblib.load(ROOT / "results/models/Multi_BERT_model.joblib")
    mul_scaler = joblib.load(ROOT / "results/models/Multi_BERT_scaler.joblib")
    mul_le = joblib.load(ROOT / "results/models/Multi_BERT_label_encoder.joblib")
    return features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le


def predict_rows(df: pd.DataFrame, features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le):
    X_bin = df.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc = pd.DataFrame(bin_scaler.transform(X_bin).astype("float32"), columns=features)
    bin_pred = bin_model.predict(X_bin_sc)

    X_mul = df.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc = pd.DataFrame(mul_scaler.transform(X_mul).astype("float32"), columns=features)
    mul_pred_idx = mul_model.predict(X_mul_sc)
    mul_pred = mul_le.inverse_transform(mul_pred_idx)

    valid_classes = set(mul_le.classes_)
    writer = df["Writer"].astype(str).str.replace("_paraphrased", "", regex=False)
    mul_scorable = writer.isin(valid_classes).to_numpy()
    mul_correct = np.zeros(len(df), dtype=int)
    for i in range(len(df)):
        if mul_scorable[i]:
            mul_correct[i] = int(mul_pred[i] == writer.iloc[i])

    return bin_pred, mul_pred, mul_scorable, mul_correct


def build_cache() -> pd.DataFrame:
    features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le = load_models()
    records = []
    join_stats = {}

    for cfg in DATA_CONFIGS:
        name = cfg["name"]
        print(f"\n[{name}] joining raw and processed by exact text...", flush=True)
        raw = pd.read_csv(ROOT / cfg["raw"])
        proc = pd.read_csv(ROOT / cfg["processed"])
        joined = build_exact_join(raw, proc, name)
        print(f"[{name}] rows={len(joined)} exact joins=100%", flush=True)

        bin_pred, mul_pred, mul_scorable, mul_correct = predict_rows(
            joined, features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le
        )

        valid = joined["original_text"].notna()
        f1 = np.full(len(joined), np.nan, dtype=float)
        if valid.any():
            refs = joined.loc[valid, "original_text"].astype(str).tolist()
            hyps = joined.loc[valid, "Text"].astype(str).tolist()
            print(f"[{name}] computing BERTScore for {len(refs)} valid pairs...", flush=True)
            _, _, f1_valid = bert_score(
                hyps,
                refs,
                lang="en",
                model_type="roberta-large",
                batch_size=64,
                device="cuda",
                verbose=False,
            )
            f1[valid.to_numpy()] = f1_valid.cpu().numpy()
        n_missing = int((~valid).sum())
        if n_missing:
            print(f"[{name}] {n_missing} rows have no Original_Text and are excluded from regression", flush=True)

        writer = joined["Writer"].astype(str).str.replace("_paraphrased", "", regex=False)
        binary_true = joined["is_AI"].astype(int).to_numpy()
        for i in range(len(joined)):
            records.append(
                {
                    "paraphraser": name,
                    "is_gemma27b": int(name == "Gemma-27B"),
                    "writer": writer.iloc[i],
                    "article_id": article_hash(joined.loc[i, "original_text"]) if valid.iloc[i] else None,
                    "bertscore": float(f1[i]) if np.isfinite(f1[i]) else np.nan,
                    "binary_correct": int(bin_pred[i] == binary_true[i]),
                    "binary_true": int(binary_true[i]),
                    "binary_pred": int(bin_pred[i]),
                    "multi_scorable": int(mul_scorable[i]),
                    "multi_correct": int(mul_correct[i]),
                    "multi_true": writer.iloc[i],
                    "multi_pred": mul_pred[i],
                    "raw_index": int(joined.loc[i, "raw_index"]),
                    "match_type": joined.loc[i, "match_type"],
                }
            )

        join_stats[name] = {
            "raw_rows": int(len(raw)),
            "processed_rows": int(len(proc)),
            "joined_rows": int(len(joined)),
            "unmatched": 0,
        }

    cache = pd.DataFrame(records)
    n_missing = int(cache["bertscore"].isna().sum())
    if n_missing:
        print(f"[cache] {n_missing} rows excluded from regression (missing Original_Text)", flush=True)

    cache.to_csv(CACHE_FILE, index=False)
    expected_rows = 6 * 2100
    missing_by_paraphraser = {
        name: int(2100 - stats["processed_rows"]) for name, stats in join_stats.items()
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(
            {
                "join_stats": join_stats,
                "n_rows": int(len(cache)),
                "n_valid_rows": int(cache["bertscore"].notna().sum()),
                "n_missing_original_text": n_missing,
                "expected_rows": int(expected_rows),
                "missing_rows": int(expected_rows - len(cache)),
                "missing_by_paraphraser": missing_by_paraphraser,
                "n_clusters": int(cache["article_id"].nunique()),
                "cluster_count_note": (
                    "2,100 prompt slots minus 3 recovered prompt references that duplicate "
                    "existing articles minus 1 exact duplicate original text."
                ),
                "bertscore_mean": float(cache["bertscore"].mean()),
                "bertscore_std": float(cache["bertscore"].std()),
                "gemma27b_bertscore_mean": float(cache.loc[cache.is_gemma27b == 1, "bertscore"].mean()),
                "gemma27b_bertscore_std": float(cache.loc[cache.is_gemma27b == 1, "bertscore"].std()),
            },
            f,
            indent=2,
        )
    print(f"\n[cache] {len(cache)} rows, {cache['article_id'].nunique()} clusters -> {CACHE_FILE}", flush=True)
    return cache


def fit_model(df: pd.DataFrame, target: str, extra: str = "", label: str = "") -> dict:
    data = df.dropna(subset=[target, "bertscore", "article_id", "paraphraser"]).copy()
    formula = f"{target} ~ bertscore + C(paraphraser, Treatment(reference='GPT-4o-mini'))"
    if extra:
        formula += " + " + extra
    model = smf.logit(formula, data=data).fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": data["article_id"]},
    )
    beta = float(model.params["bertscore"])
    se = float(model.bse["bertscore"])
    z = float(model.tvalues["bertscore"])
    p = float(model.pvalues["bertscore"])
    ci_low, ci_high = model.conf_int().loc["bertscore"]
    return {
        "Model": label,
        "Dependent_Variable": target,
        "N_Observations": int(len(data)),
        "N_Clusters": int(data["article_id"].nunique()),
        "Pseudo_R2": round(float(model.prsquared), 4),
        "BERTScore_Beta": round(beta, 4),
        "Clustered_SE": round(se, 4),
        "z_statistic": round(z, 2),
        "p_value": p,
        "Odds_Ratio_per_0.1": round(float(np.exp(0.1 * beta)), 2),
        "Beta_CI_Lower": round(float(ci_low), 4),
        "Beta_CI_Upper": round(float(ci_high), 4),
        "OR_CI_Lower": round(float(np.exp(0.1 * ci_low)), 2),
        "OR_CI_Upper": round(float(np.exp(0.1 * ci_high)), 2),
    }


def fit_all(cache: pd.DataFrame) -> pd.DataFrame:
    full = cache[cache["multi_scorable"] == 1].copy()
    restricted = full[full["is_gemma27b"] == 0].copy()

    specs = [
        ("1A", "multi_correct", full, "", "Multi-Class Attribution | Full | Paraphraser controls"),
        ("1A-WFE", "multi_correct", full, "C(writer)", "Multi-Class Attribution | Full | + Writer FE"),
        ("1B", "multi_correct", restricted, "", "Multi-Class Attribution | Restricted | Paraphraser controls"),
        ("1B-WFE", "multi_correct", restricted, "C(writer)", "Multi-Class Attribution | Restricted | + Writer FE"),
        ("2A", "binary_correct", cache, "", "Binary Detection | Full | Paraphraser controls"),
        ("2A-WFE", "binary_correct", cache, "C(writer)", "Binary Detection | Full | + Writer FE"),
        ("2A-AI", "binary_correct", cache[cache.binary_true == 1], "", "Binary Detection | Full | AI-only"),
        ("2A-Human", "binary_correct", cache[cache.binary_true == 0], "", "Binary Detection | Full | Human-only"),
        ("2B", "binary_correct", cache[cache.is_gemma27b == 0], "", "Binary Detection | Restricted | Paraphraser controls"),
        ("2B-WFE", "binary_correct", cache[cache.is_gemma27b == 0], "C(writer)", "Binary Detection | Restricted | + Writer FE"),
        ("2B-AI", "binary_correct", cache[(cache.is_gemma27b == 0) & (cache.binary_true == 1)], "", "Binary Detection | Restricted | AI-only"),
        ("2B-Human", "binary_correct", cache[(cache.is_gemma27b == 0) & (cache.binary_true == 0)], "", "Binary Detection | Restricted | Human-only"),
    ]

    rows = []
    for key, target, data, extra, label in specs:
        print(f"Fitting {key}: N={len(data)}...", flush=True)
        row = fit_model(data, target, extra=extra, label=label)
        row["Specification"] = "Writer FE" if "Writer FE" in label else ("AI-only" if "AI-only" in label else ("Human-only" if "Human-only" in label else "Paraphraser controls"))
        row["Regime"] = "Full" if "Full" in label else "Restricted"
        row["Model_Key"] = key
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = [
        "Model_Key", "Model", "Regime", "Specification", "Dependent_Variable",
        "N_Observations", "N_Clusters", "Pseudo_R2", "BERTScore_Beta",
        "Clustered_SE", "z_statistic", "p_value", "Odds_Ratio_per_0.1",
        "Beta_CI_Lower", "Beta_CI_Upper", "OR_CI_Lower", "OR_CI_Upper",
    ]
    out = out[cols]
    out.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\n[results] saved {len(out)} specifications -> {OUT_RESULTS_CSV}", flush=True)
    print(out.to_string(index=False), flush=True)
    return out


def main():
    os.makedirs(ROOT / "results/phase3", exist_ok=True)
    cache = build_cache()
    fit_all(cache)


if __name__ == "__main__":
    main()
