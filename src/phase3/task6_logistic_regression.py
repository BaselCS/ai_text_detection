"""Task 6 (rebuilt): document-level logistic regressions on BERTScore.

Design
------
* Raw and processed paraphrase files are joined only on exact normalized
  ``(Paraphrased_Text, Writer)`` pairs.  No positional ``iloc`` zipping.
* The 18 rows whose retained raw ``Original_Text`` was missing are explicitly
  excluded by ``results/phase3/T6_excluded_rows.csv``.  They are not recovered
  from a parallel writer block: the raw files are stacked writer blocks, so the
  same prompt index is a different article in another block.
* The five-paraphraser restricted regime is the primary clean comparison.
  Gemma-27B is a reconstructed, largely different article sample; the
  six-paraphraser regime is labelled an Extended-Pool Sensitivity Analysis.
* Inference uses paraphraser-specific BERTScore slopes in addition to the
  common-slope models.  The invalid independent-SE formula
  ``sqrt(SE_full^2 + SE_restricted^2)`` is not used.

Outputs
-------
results/phase3/T6_document_level_bertscore_cache.csv
results/phase3/T6_logistic_regression_bertscore.csv
results/phase3/T6_within_class_diagnostics.csv
results/phase3/T6_per_paraphraser_slopes.csv
results/phase3/T6_interaction_test.csv
results/phase3/T6_rebuild_summary.json
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
WITHIN_CSV = ROOT / "results/phase3/T6_within_class_diagnostics.csv"
SLOPES_CSV = ROOT / "results/phase3/T6_per_paraphraser_slopes.csv"
INTERACTION_CSV = ROOT / "results/phase3/T6_interaction_test.csv"
SUMMARY_JSON = ROOT / "results/phase3/T6_rebuild_summary.json"
EXCLUSIONS = ROOT / "results/phase3/T6_excluded_rows.csv"

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
    return re.sub(r"\s+", " ", str(value)).strip()


def article_hash(text: str) -> str:
    clean = " ".join(str(text).strip().split())[:300]
    return hashlib.md5(clean.encode("utf-8")).hexdigest()[:12]


def build_exact_join(raw: pd.DataFrame, proc: pd.DataFrame, name: str) -> pd.DataFrame:
    raw = raw.copy()
    proc = proc.copy()
    raw["writer_norm"] = raw["Writer"].astype(str).str.replace("_paraphrased", "", regex=False)
    proc["writer_norm"] = proc["Writer"].astype(str)
    raw["text_norm"] = raw["Paraphrased_Text"].map(normalize_text)
    proc["text_norm"] = proc["Text"].map(normalize_text)

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
        attached.append(
            {
                "raw_index": int(raw_idx),
                "original_text": original,
                "match_type": "exact_text_writer" if pd.notna(original) else "missing_original_text",
                "pair_match_cosine": raw.loc[raw_idx, "match_cosine"] if "match_cosine" in raw.columns else np.nan,
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


def predict_rows(df, features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le):
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


def pairing_validation(raw_gemma: pd.DataFrame) -> dict:
    base_files = [
        "data/processed/raw_paraphrased_gemma2.csv",
        "data/processed/raw_paraphrased_qwen2.csv",
        "data/processed/raw_paraphrased_llama3.csv",
        "data/processed/raw_paraphrased_mistral7b.csv",
        "data/processed/raw_paraphrased_gpt4o_mini.csv",
    ]
    pool = set()
    for path in base_files:
        d = pd.read_csv(ROOT / path)
        pool.update(d["Original_Text"].map(normalize_text))
    hits = raw_gemma["Original_Text"].map(normalize_text).isin(pool)
    by_writer = (
        raw_gemma.assign(_hit=hits)
        .groupby("Writer")["_hit"]
        .agg(["mean", "sum", "count"])
        .reset_index()
    )
    return {
        "n_pairs": int(len(raw_gemma)),
        "median_match_cosine": float(raw_gemma["match_cosine"].median()),
        "mean_match_cosine": float(raw_gemma["match_cosine"].mean()),
        "frac_below_0_1": float((raw_gemma["match_cosine"] < 0.1).mean()),
        "base_pool_articles": int(len(pool)),
        "base_pool_overlap_frac": float(hits.mean()),
        "base_pool_overlap_n": int(hits.sum()),
        "overlap_by_writer": {
            row["Writer"]: {
                "frac": float(row["mean"]),
                "n_overlap": int(row["sum"]),
                "n": int(row["count"]),
            }
            for _, row in by_writer.iterrows()
        },
    }


def build_cache() -> pd.DataFrame:
    features, bin_model, bin_scaler, mul_model, mul_scaler, mul_le = load_models()
    records = []
    join_stats = {}
    exclusions = pd.read_csv(EXCLUSIONS)
    exclusions["text_norm"] = exclusions["text"].map(normalize_text)
    excluded_keys = set(zip(exclusions["paraphraser"], exclusions["writer"], exclusions["text_norm"]))
    n_excluded = 0

    for cfg in DATA_CONFIGS:
        name = cfg["name"]
        print(f"\n[{name}] joining raw and processed by exact text...", flush=True)
        raw = pd.read_csv(ROOT / cfg["raw"])
        proc = pd.read_csv(ROOT / cfg["processed"])
        joined = build_exact_join(raw, proc, name)

        # Explicitly drop the 18 mis-recovered meta-refusal rows by
        # (paraphraser, writer, text).  This is not a cross-writer recovery.
        keep = []
        for _, row in joined.iterrows():
            key = (name, row["writer_norm"], row["text_norm"])
            if key in excluded_keys:
                n_excluded += 1
                keep.append(False)
            else:
                keep.append(True)
        joined = joined.loc[keep].reset_index(drop=True)
        print(f"[{name}] rows={len(joined)}; excluded={int((~pd.Series(keep)).sum())}", flush=True)

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
                hyps, refs, lang="en", model_type="roberta-large",
                batch_size=64, device="cuda", verbose=False,
            )
            f1[valid.to_numpy()] = f1_valid.cpu().numpy()
        n_missing = int((~valid).sum())
        if n_missing:
            print(f"[{name}] {n_missing} rows have no Original_Text and are excluded", flush=True)

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
                    "pair_match_cosine": joined.loc[i, "pair_match_cosine"],
                }
            )

        join_stats[name] = {
            "raw_rows": int(len(raw)),
            "processed_rows": int(len(proc)),
            "joined_rows": int(len(joined)),
            "excluded_misrecovered": int((~pd.Series(keep)).sum()),
            "missing_original_text": n_missing,
        }

    cache = pd.DataFrame(records)
    cache.to_csv(CACHE_FILE, index=False)

    raw_gemma = pd.read_csv(ROOT / "data/processed/raw_paraphrased_gemma27b.csv")
    full = cache
    restricted = cache[cache["is_gemma27b"] == 0]
    summary = {
        "join_stats": join_stats,
        "n_rows": int(len(cache)),
        "n_valid_rows": int(cache["bertscore"].notna().sum()),
        "n_missing_original_text": int(cache["bertscore"].isna().sum()),
        "n_clusters_full_extended": int(cache["article_id"].nunique()),
        "n_clusters_restricted": int(restricted["article_id"].nunique()),
        "expected_rows": 6 * 2100,
        "missing_rows_vs_expected": int(6 * 2100 - len(cache)),
        "excluded_misrecovered_rows": int(n_excluded),
        "bertscore_mean": float(cache["bertscore"].mean()),
        "bertscore_std": float(cache["bertscore"].std()),
        "gemma27b_bertscore_mean": float(cache.loc[cache.is_gemma27b == 1, "bertscore"].mean()),
        "gemma27b_bertscore_std": float(cache.loc[cache.is_gemma27b == 1, "bertscore"].std()),
        "pairing_validation": pairing_validation(raw_gemma),
        "notes": {
            "regime_labels": {
                "restricted": "Five paraphrasers sharing the 2,100-article pool; primary clean comparison.",
                "full_extended": "Six paraphrasers including Gemma-27B; Cross-Sample / Extended-Pool Sensitivity Analysis.",
            },
            "excluded_rows_file": str(EXCLUSIONS.relative_to(ROOT)),
        },
    }
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[cache] {len(cache)} rows, {cache['article_id'].nunique()} clusters -> {CACHE_FILE}", flush=True)
    return cache


def _fit(formula: str, data: pd.DataFrame):
    model = smf.logit(formula, data=data).fit(
        disp=False, cov_type="cluster", cov_kwds={"groups": data["article_id"]}
    )
    return model, bool(model.mle_retvals.get("converged", False))


def fit_model(df: pd.DataFrame, target: str, extra: str = "", label: str = "") -> dict:
    data = df.dropna(subset=[target, "bertscore", "article_id", "paraphraser"]).copy()
    formula = f"{target} ~ bertscore + C(paraphraser, Treatment(reference='GPT-4o-mini'))"
    if extra:
        formula += " + " + extra
    model, converged = _fit(formula, data)
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
        "Converged": converged,
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
        ("1A", "multi_correct", full, "", "Extended-Pool Sensitivity | Multi-Class | Paraphraser controls"),
        ("1A-WFE", "multi_correct", full, "C(writer)", "Extended-Pool Sensitivity | Multi-Class | + Writer FE"),
        ("1B", "multi_correct", restricted, "", "Restricted (5 paraphrasers) | Multi-Class | Paraphraser controls"),
        ("1B-WFE", "multi_correct", restricted, "C(writer)", "Restricted (5 paraphrasers) | Multi-Class | + Writer FE"),
        ("2A", "binary_correct", cache, "", "Extended-Pool Sensitivity | Binary | Paraphraser controls"),
        ("2A-WFE", "binary_correct", cache, "C(writer)", "Extended-Pool Sensitivity | Binary | + Writer FE"),
        ("2B", "binary_correct", cache[cache.is_gemma27b == 0], "", "Restricted (5 paraphrasers) | Binary | Paraphraser controls"),
        ("2B-WFE", "binary_correct", cache[cache.is_gemma27b == 0], "C(writer)", "Restricted (5 paraphrasers) | Binary | + Writer FE"),
    ]
    rows = []
    for key, target, data, extra, label in specs:
        print(f"Fitting {key}: N={len(data)}...", flush=True)
        row = fit_model(data, target, extra=extra, label=label)
        row["Specification"] = "Writer FE" if "Writer FE" in label else "Paraphraser controls"
        row["Regime"] = "Extended-Pool Sensitivity" if "Extended-Pool" in label else "Restricted"
        row["Model_Key"] = key
        rows.append(row)
    out = pd.DataFrame(rows)
    cols = [
        "Model_Key", "Model", "Regime", "Specification", "Dependent_Variable",
        "N_Observations", "N_Clusters", "Converged", "Pseudo_R2", "BERTScore_Beta",
        "Clustered_SE", "z_statistic", "p_value", "Odds_Ratio_per_0.1",
        "Beta_CI_Lower", "Beta_CI_Upper", "OR_CI_Lower", "OR_CI_Upper",
    ]
    out = out[cols]
    out.to_csv(OUT_RESULTS_CSV, index=False)
    print(f"\n[results] saved {len(out)} specifications -> {OUT_RESULTS_CSV}", flush=True)
    print(out.to_string(index=False), flush=True)
    return out


def fit_within_class(cache: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("AI-only | Extended-Pool", cache[cache.binary_true == 1], "Extended-Pool Sensitivity"),
        ("Human-only | Extended-Pool", cache[cache.binary_true == 0], "Extended-Pool Sensitivity"),
        ("AI-only | Restricted", cache[(cache.is_gemma27b == 0) & (cache.binary_true == 1)], "Restricted"),
        ("Human-only | Restricted", cache[(cache.is_gemma27b == 0) & (cache.binary_true == 0)], "Restricted"),
    ]
    rows = []
    for label, data, regime in specs:
        row = fit_model(data, "binary_correct", extra="", label=label)
        row["Regime"] = regime
        row["Specification"] = "AI-only" if "AI-only" in label else "Human-only"
        row["Reported"] = bool(row["Converged"])
        row["Note"] = (
            "Quasi-separated; point estimate not reported"
            if not row["Converged"] and "AI-only" in label
            else ""
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(WITHIN_CSV, index=False)
    print(f"\n[within-class] saved -> {WITHIN_CSV}", flush=True)
    print(out[["Model", "N_Observations", "Converged", "BERTScore_Beta", "Clustered_SE", "p_value", "Reported"]].to_string(index=False), flush=True)
    return out


def fit_per_paraphraser_slopes(cache: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["multi_correct", "binary_correct"]:
        data = cache.dropna(subset=[target, "bertscore", "article_id", "paraphraser"]).copy()
        formula = f"{target} ~ bertscore * C(paraphraser, Treatment(reference='GPT-4o-mini'))"
        model, converged = _fit(formula, data)
        base = float(model.params["bertscore"])
        cov = model.cov_params()
        paraphrasers = sorted(data["paraphraser"].unique())
        for p in paraphrasers:
            if p == "GPT-4o-mini":
                beta, se = base, float(model.bse["bertscore"])
            else:
                key = f"C(paraphraser, Treatment(reference='GPT-4o-mini'))[T.{p}]"
                inter = f"bertscore:{key}"
                beta = base + float(model.params[inter])
                var = float(cov.loc["bertscore", "bertscore"]) + float(cov.loc[inter, inter]) + 2 * float(cov.loc["bertscore", inter])
                se = float(np.sqrt(max(var, 0.0)))
            z = beta / se if se else np.nan
            pval = float(2 * (1 - __import__("scipy.stats", fromlist=["norm"]).norm.cdf(abs(z)))) if se else np.nan
            rows.append({
                "Outcome": target,
                "Paraphraser": p,
                "BERTScore_Beta": round(beta, 4),
                "Clustered_SE": round(se, 4),
                "z_statistic": round(float(z), 2) if se else np.nan,
                "p_value": pval,
                "N_Observations": int((data.paraphraser == p).sum()),
                "Converged": converged,
            })
        # Joint Wald test that all interaction terms are zero (equal slopes).
        inter_terms = [k for k in model.params.index if k.startswith("bertscore:")]
        R = np.zeros((len(inter_terms), len(model.params)))
        for i, term in enumerate(inter_terms):
            R[i, list(model.params.index).index(term)] = 1
        wald = model.wald_test(R, scalar=True)
        rows.append({
            "Outcome": target,
            "Paraphraser": "__joint_wald_equal_slopes__",
            "BERTScore_Beta": np.nan,
            "Clustered_SE": np.nan,
            "z_statistic": np.nan,
            "p_value": float(wald.pvalue),
            "N_Observations": int(len(data)),
            "Converged": converged,
        })
    out = pd.DataFrame(rows)
    out.to_csv(SLOPES_CSV, index=False)
    print(f"\n[per-paraphraser] saved -> {SLOPES_CSV}", flush=True)
    print(out.to_string(index=False), flush=True)
    return out


def fit_interactions(cache: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in ["multi_correct", "binary_correct"]:
        data = cache.dropna(subset=[target, "bertscore", "article_id", "paraphraser"]).copy()
        data["gemma"] = data["is_gemma27b"].astype(int)
        specs = [
            ("Pooled interaction (no paraphraser controls)", f"{target} ~ bertscore * gemma"),
            ("Paraphraser controls (Gemma-27B vs GPT-4o-mini reference)", f"{target} ~ bertscore + C(paraphraser, Treatment(reference='GPT-4o-mini')) + bertscore:gemma"),
            ("Paraphraser controls + writer FE", f"{target} ~ bertscore + C(paraphraser, Treatment(reference='GPT-4o-mini')) + C(writer) + bertscore:gemma"),
        ]
        for label, formula in specs:
            model, converged = _fit(formula, data)
            term = [k for k in model.params.index if k.startswith("bertscore:gemma")][0]
            rows.append({
                "Outcome": target,
                "Specification": label,
                "N_Observations": int(len(data)),
                "N_Clusters": int(data["article_id"].nunique()),
                "Interaction_Beta": round(float(model.params[term]), 4),
                "Clustered_SE": round(float(model.bse[term]), 4),
                "z_statistic": round(float(model.tvalues[term]), 2),
                "p_value": float(model.pvalues[term]),
                "Converged": converged,
                "Reference_Slope": round(float(model.params["bertscore"]), 4),
            })
    out = pd.DataFrame(rows)
    out.to_csv(INTERACTION_CSV, index=False)
    print(f"\n[interaction] saved -> {INTERACTION_CSV}", flush=True)
    print(out.to_string(index=False), flush=True)
    return out


def main():
    os.makedirs(ROOT / "results/phase3", exist_ok=True)
    cache = build_cache()
    fit_all(cache)
    fit_within_class(cache)
    fit_per_paraphraser_slopes(cache)
    fit_interactions(cache)


if __name__ == "__main__":
    main()
