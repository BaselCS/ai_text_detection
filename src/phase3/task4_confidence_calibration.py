"""Task 4: Confidence Calibration Analysis on Retrained 6-Class Model.

Compares confidence calibration metrics between:
1. Baseline 7-class hybrid model (with Gemma-2-9B)
2. Retrained 6-class hybrid model (without Gemma-2-9B)

Datasets:
- Gemma-27B paraphrases (data/processed/processed_paraphrased_articles.csv)
- Mistral-7B paraphrases (data/processed/processed_paraphrased_mistral7b.csv)

Metrics computed:
- Mean and Std of confidence for Correct vs Wrong predictions
- Confidence Gap (Mean_Correct - Mean_Wrong)
- Multi-class Brier Score: (1/N) * sum_i sum_c (p_ic - y_ic)^2
- Expected Calibration Error (ECE) with 10 equal-width bins:
  ECE = sum_m (|B_m| / N) * |acc(B_m) - conf(B_m)|
- Maximum Calibration Error (MCE)
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def compute_ece_mce(confidences, predictions, ground_truth, n_bins=10):
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(confidences)

    correctness = (predictions == ground_truth).astype(float)

    bin_stats = []
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Upper boundary inclusive on the last bin
        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(correctness[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            abs_diff = np.abs(bin_acc - bin_conf)
            ece += (bin_size / n_samples) * abs_diff
            mce = max(mce, abs_diff)
            bin_stats.append({
                "bin": f"[{bin_lower:.1f}, {bin_upper:.1f}]",
                "size": int(bin_size),
                "acc": float(bin_acc),
                "conf": float(bin_conf),
                "diff": float(abs_diff)
            })

    return float(ece), float(mce), bin_stats


def compute_multiclass_brier(proba, y_indices, n_classes):
    """Compute standard multi-class Brier score."""
    y_one_hot = np.zeros_like(proba)
    for i, label in enumerate(y_indices):
        y_one_hot[i, label] = 1.0
    return float(np.mean(np.sum((proba - y_one_hot) ** 2, axis=1)))


def main():
    root = Path(__file__).resolve().parent.parent.parent
    features_path = root / "results/models/Bin_BERT_features.json"
    with open(features_path) as f:
        features = json.load(f)

    # 1. Load models, scalers, label encoders
    print("Loading 7-class baseline model...")
    m7 = joblib.load(root / "results/models/Multi_BERT_model.joblib")
    sc7 = joblib.load(root / "results/models/Multi_BERT_scaler.joblib")
    le7 = joblib.load(root / "results/models/Multi_BERT_label_encoder.joblib")
    classes_7 = list(le7.classes_)
    print("7-class labels:", classes_7)

    print("Loading 6-class retrained model (No-Gemma)...")
    m6 = joblib.load(root / "results/phase2/T21_NoGemma_model.joblib")
    sc6 = joblib.load(root / "results/phase2/T21_NoGemma_scaler.joblib")
    le6 = joblib.load(root / "results/phase2/T21_NoGemma_label_encoder.joblib")
    classes_6 = list(le6.classes_)
    print("6-class labels:", classes_6)

    datasets = {
        "Gemma-27B": root / "data/processed/processed_paraphrased_articles.csv",
        "Mistral-7B": root / "data/processed/processed_paraphrased_mistral7b.csv",
    }

    summary_rows = []
    detailed_rows = []

    for name, path in datasets.items():
        print(f"\nProcessing {name} from {path}...")
        df = pd.read_csv(path, low_memory=True)
        print(f"Total rows in {name}: {len(df)}")
        print("Writer distribution in raw dataset:\n", df["Writer"].value_counts().to_dict())

        # Scorable subset for 6-class: strictly samples whose true Writer is in classes_6
        # To make apples-to-apples comparison, we evaluate:
        # Configuration A: 6-Class scorable subset (N ~ 1775-1800, excluding true Gemma-2-9B)
        # Configuration B: Full 7-class set (for baseline context)
        
        for eval_mode in ["scorable_6class", "full_7class"]:
            if eval_mode == "scorable_6class":
                sub_df = df[df["Writer"].isin(classes_6)].copy().reset_index(drop=True)
                models_to_test = [("6-Class (No-Gemma)", m6, sc6, le6, classes_6),
                                  ("7-Class Baseline (Matched Samples)", m7, sc7, le7, classes_7)]
            else:
                sub_df = df[df["Writer"].isin(classes_7)].copy().reset_index(drop=True)
                models_to_test = [("7-Class Baseline (Full Set)", m7, sc7, le7, classes_7)]

            for model_name, model, sc, le, class_list in models_to_test:
                y_true_str = sub_df["Writer"].values
                y_true_idx = le.transform(y_true_str)

                # Preprocess features
                X_df = sub_df.reindex(columns=features, fill_value=0).astype("float32")
                X_scaled = pd.DataFrame(sc.transform(X_df).astype("float32"), columns=features)

                # Inference
                proba = model.predict_proba(X_scaled)
                conf = proba.max(axis=1)
                pred_idx = proba.argmax(axis=1)
                pred_str = le.inverse_transform(pred_idx)

                is_correct = (pred_idx == y_true_idx)
                n_total = len(sub_df)
                n_correct = int(np.sum(is_correct))
                n_wrong = int(n_total - n_correct)
                acc = float(n_correct / n_total)

                conf_correct = conf[is_correct]
                conf_wrong = conf[~is_correct]

                mean_conf_corr = float(np.mean(conf_correct)) if len(conf_correct) > 0 else 0.0
                std_conf_corr = float(np.std(conf_correct)) if len(conf_correct) > 0 else 0.0

                mean_conf_wrong = float(np.mean(conf_wrong)) if len(conf_wrong) > 0 else 0.0
                std_conf_wrong = float(np.std(conf_wrong)) if len(conf_wrong) > 0 else 0.0

                conf_gap = mean_conf_corr - mean_conf_wrong

                brier = compute_multiclass_brier(proba, y_true_idx, len(class_list))
                ece, mce, bin_stats = compute_ece_mce(conf, pred_idx, y_true_idx, n_bins=10)

                print(f"--- {name} | {model_name} (N={n_total}) ---")
                print(f"Accuracy: {acc:.4f} ({n_correct}/{n_total})")
                print(f"Conf Correct: {mean_conf_corr:.4f} ± {std_conf_corr:.4f} (n={n_correct})")
                print(f"Conf Wrong:   {mean_conf_wrong:.4f} ± {std_conf_wrong:.4f} (n={n_wrong})")
                print(f"Conf Gap:     {conf_gap:+.4f}")
                print(f"Brier Score:  {brier:.4f}")
                print(f"ECE (10-bin): {ece:.4f} | MCE: {mce:.4f}")

                summary_rows.append({
                    "Dataset": name,
                    "Model": model_name,
                    "Evaluation_Subset": eval_mode,
                    "N_samples": n_total,
                    "N_correct": n_correct,
                    "N_wrong": n_wrong,
                    "Accuracy": round(acc, 4),
                    "Mean_Conf_Correct": round(mean_conf_corr, 4),
                    "Std_Conf_Correct": round(std_conf_corr, 4),
                    "Mean_Conf_Wrong": round(mean_conf_wrong, 4),
                    "Std_Conf_Wrong": round(std_conf_wrong, 4),
                    "Confidence_Gap": round(conf_gap, 4),
                    "Brier_Score": round(brier, 4),
                    "ECE_10bin": round(ece, 4),
                    "MCE": round(mce, 4)
                })

                # Detailed sample logging
                for idx in range(min(n_total, 500)):  # sample up to 500 per configuration for audit
                    detailed_rows.append({
                        "Dataset": name,
                        "Model": model_name,
                        "True_Writer": y_true_str[idx],
                        "Pred_Writer": pred_str[idx],
                        "Confidence": round(float(conf[idx]), 4),
                        "Is_Correct": int(is_correct[idx])
                    })

    # Save summary table
    out_dir = root / "results/phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "T4_confidence_calibration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to {summary_csv}")

    detail_df = pd.DataFrame(detailed_rows)
    detail_csv = out_dir / "T4_confidence_calibration_predictions_sample.csv"
    detail_df.to_csv(detail_csv, index=False)
    print(f"Saved predictions sample to {detail_csv}")

    print("\nSummary Results:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
