"""Add explicit chance and uniform-forecast calibration benchmarks to Task 4.

The reported Brier scores and ECE values are only interpretable against a
baseline.  For a K-class problem:

* chance accuracy = 1 / K
* uniform-forecast Brier = (K - 1) / K
* Brier skill score = 1 - Brier / uniform_Brier

This script does not alter the fitted predictions; it only writes a companion
benchmark table so the report cannot describe a worse-than-uniform model as
well calibrated.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY = ROOT / "results/phase3/T4_confidence_calibration_summary.csv"
OUT = ROOT / "results/phase3/T4_calibration_benchmarks.csv"


def main():
    df = pd.read_csv(SUMMARY)
    rows = []
    for _, row in df.iterrows():
        k = 6 if "6-Class" in row["Model"] else 7
        uniform_brier = (k - 1) / k
        chance_accuracy = 1 / k
        rows.append(
            {
                "Dataset": row["Dataset"],
                "Model": row["Model"],
                "Evaluation_Subset": row["Evaluation_Subset"],
                "N_samples": int(row["N_samples"]),
                "Accuracy": float(row["Accuracy"]),
                "Chance_Accuracy": round(chance_accuracy, 4),
                "Accuracy_Above_Chance": round(float(row["Accuracy"]) - chance_accuracy, 4),
                "Brier_Score": float(row["Brier_Score"]),
                "Uniform_Forecast_Brier": round(uniform_brier, 4),
                "Brier_Skill_Score": round(1 - float(row["Brier_Score"]) / uniform_brier, 4),
                "ECE_10bin": float(row["ECE_10bin"]),
                "Worse_Than_Uniform_Brier": bool(float(row["Brier_Score"]) > uniform_brier),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
