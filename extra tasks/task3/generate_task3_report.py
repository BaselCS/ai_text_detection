"""
Task3 Report Generator
Produces all CSV files and figures for results/extra/task3/
Following F1.png color style (White background, Blues palette, crisp dark typography)
Run: uv run python results/extra/task3/generate_task3_report.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Output directory ──────────────────────────────────────────────────────────
OUT = Path("results/extra/task3")
OUT.mkdir(parents=True, exist_ok=True)

# ── Colour palette matching F1.png (Blues Theme) ──────────────────────────────
DARK_BLUE   = "#08306B"
NAVY_BLUE   = "#185498"
MED_BLUE    = "#2171B5"
LIGHT_BLUE  = "#6BAED6"
PALE_BLUE   = "#C6DBEF"
ACCENT_RED  = "#C63939"
ACCENT_ORANGE = "#D95F02"
DARK_GRAY   = "#222222"

PLT_STYLE = {
    "axes.facecolor":  "#FFFFFF",
    "figure.facecolor":"#FFFFFF",
    "axes.edgecolor":  "#CCCCCC",
    "axes.labelcolor": "#111111",
    "axes.titlesize":  13,
    "axes.titleweight":"bold",
    "axes.labelsize":  11,
    "xtick.color":     "#222222",
    "ytick.color":     "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.facecolor":"#FFFFFF",
    "legend.edgecolor":"#CCCCCC",
    "legend.fontsize": 9,
    "grid.color":      "#E5E5E5",
    "grid.linestyle":  "--",
    "grid.alpha":      0.8,
    "text.color":      "#111111",
}
plt.rcParams.update(PLT_STYLE)

# ══════════════════════════════════════════════════════════════════════════════
#  RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

train_history = {
    "binary": {
        "epoch":    [1, 2, 3],
        "val_loss": [0.0079, 0.0054, 0.0039],
        "val_acc":  [0.9975, 0.9982, 0.9922],
        "val_f1":   [0.9975, 0.9982, 0.9922],
    },
    "multi": {
        "epoch":    [1, 2, 3],
        "val_loss": [0.1601, 0.0862, 0.0500],
        "val_acc":  [0.9575, 0.9761, 0.9922],
        "val_f1":   [0.9575, 0.9760, 0.9922],
    },
}

eval_data = [
    # Binary
    ("RoBERTa_Binary", "Clean_Test",         10237, 0.9989, 0.9978, 0.9989, 30.10),
    ("RoBERTa_Binary", "Misspell_5%",        10242, 0.9953, 0.9903, 0.9953, 29.14),
    ("RoBERTa_Binary", "Misspell_10%",       10242, 0.9969, 0.9936, 0.9969, 30.00),
    ("RoBERTa_Binary", "Misspell_15%",       10242, 0.9973, 0.9944, 0.9973, 30.98),
    ("RoBERTa_Binary", "Misspell_20%",       10242, 0.9980, 0.9960, 0.9980, 31.76),
    ("RoBERTa_Binary", "CrossDomain_Essay",    396, 0.5051, 0.3529, 0.3498, 24.36),
    ("RoBERTa_Binary", "CrossDomain_WP",       398, 0.5176, 0.3752, 0.3737, 21.50),
    ("RoBERTa_Binary", "Unseen_Reuters",       400, 0.5150, 0.3658, 0.3658, 26.20),
    ("RoBERTa_Binary", "Paraphrased_Gemma",   2072, 0.8591, 0.4786, 0.7967, 15.39),
    ("RoBERTa_Binary", "Translation_NLLB",    1729, 1.0000, 1.0000, 1.0000, 26.41),
    # Multi
    ("RoBERTa_Multi",  "Clean_Test",         10237, 0.9723, 0.9722, 0.9722, 29.62),
    ("RoBERTa_Multi",  "Paraphrased_Gemma",   2072, 0.1684, 0.0980, 0.0980, 15.51),
    ("RoBERTa_Multi",  "Translation_NLLB",    1729, 0.9751, 0.9751, 0.9751, 26.53),
]

cols = ["Model", "Dataset", "Count", "Accuracy", "Macro_F1", "Weighted_F1", "Latency_ms"]
df_eval = pd.DataFrame(eval_data, columns=cols)

df_bin   = df_eval[df_eval["Model"] == "RoBERTa_Binary"].copy()
df_multi = df_eval[df_eval["Model"] == "RoBERTa_Multi"].copy()

# ══════════════════════════════════════════════════════════════════════════════
#  CSV FILES
# ══════════════════════════════════════════════════════════════════════════════

df_eval.to_csv(OUT / "T15_full_evaluation.csv", index=False)
df_bin.to_csv( OUT / "T15_binary_results.csv",  index=False)
df_multi.to_csv(OUT / "T15_multi_results.csv",  index=False)

for task, hist in train_history.items():
    pd.DataFrame(hist).to_csv(OUT / f"training_history_{task}.csv", index=False)

print("✓ CSVs saved")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 – Binary Macro-F1 across all stress-test datasets (Blues palette)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
datasets = df_bin["Dataset"].tolist()
f1_vals  = df_bin["Macro_F1"].tolist()
colors   = [DARK_BLUE if v >= 0.95 else MED_BLUE if v >= 0.80 else LIGHT_BLUE if v >= 0.50 else ACCENT_RED for v in f1_vals]
bars = ax.bar(datasets, f1_vals, color=colors, edgecolor=DARK_BLUE, linewidth=0.8, zorder=3)

for bar, val in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="#111111", fontweight="bold")

ax.set_ylim(0, 1.12)
ax.set_ylabel("Macro F1-Score")
ax.set_title("RoBERTa Binary – Macro F1 Across All Stress-Test Datasets", pad=12)
ax.set_xticks(range(len(datasets)))
ax.set_xticklabels(datasets, rotation=30, ha="right")
ax.axhline(0.5,  color="#888888", linestyle=":", linewidth=1.2, label="Random baseline (50%)")
ax.axhline(0.85, color=ACCENT_ORANGE, linestyle=":", linewidth=1.2, label="85% Threshold")
ax.legend(loc="upper right")
ax.grid(axis="y", zorder=0)
plt.tight_layout()
fig.savefig(OUT / "F1_binary_stress_test.png", bbox_inches="tight")
plt.close(fig)
print("✓ F1_binary_stress_test.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 – Multi-class results (Blues grouped bar)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
datasets_m = df_multi["Dataset"].tolist()
acc_m = df_multi["Accuracy"].tolist()
f1_m  = df_multi["Macro_F1"].tolist()
x = np.arange(len(datasets_m))
w = 0.35
b1 = ax.bar(x - w/2, acc_m, w, label="Accuracy",  color=DARK_BLUE, edgecolor=DARK_BLUE)
b2 = ax.bar(x + w/2, f1_m,  w, label="Macro F1",  color=LIGHT_BLUE, edgecolor=DARK_BLUE)

for b, v in zip(b1, acc_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.4f}", ha="center", fontsize=8, color="#111111", fontweight="bold")
for b, v in zip(b2, f1_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.4f}", ha="center", fontsize=8, color="#111111", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(datasets_m, rotation=15, ha="right")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("RoBERTa Multi-Class – Accuracy & Macro F1 by Dataset", pad=12)
ax.legend()
ax.grid(axis="y", zorder=0)
plt.tight_layout()
fig.savefig(OUT / "F2_multi_class_results.png", bbox_inches="tight")
plt.close(fig)
print("✓ F2_multi_class_results.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 – Misspelling robustness (Blues line plot)
# ══════════════════════════════════════════════════════════════════════════════
misspell_rows = df_bin[df_bin["Dataset"].str.startswith("Misspell")].copy()
noise_levels = [5, 10, 15, 20]
f1_misspell  = misspell_rows["Macro_F1"].tolist()
acc_misspell = misspell_rows["Accuracy"].tolist()

noise_levels = [0] + noise_levels
f1_misspell  = [df_bin[df_bin["Dataset"]=="Clean_Test"]["Macro_F1"].values[0]] + f1_misspell
acc_misspell = [df_bin[df_bin["Dataset"]=="Clean_Test"]["Accuracy"].values[0]] + acc_misspell

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
ax.plot(noise_levels, acc_misspell, "o-", color=DARK_BLUE, label="Accuracy",  linewidth=2.5, markersize=7)
ax.plot(noise_levels, f1_misspell,  "s-", color=MED_BLUE,  label="Macro F1",  linewidth=2.5, markersize=7)

for x_val, a_val, f_val in zip(noise_levels, acc_misspell, f1_misspell):
    ax.annotate(f"{a_val:.4f}", (x_val, a_val), textcoords="offset points", xytext=(-5, 10),
                fontsize=8, color=DARK_BLUE, fontweight="bold")
    ax.annotate(f"{f_val:.4f}", (x_val, f_val), textcoords="offset points", xytext=(-5,-15),
                fontsize=8, color=MED_BLUE, fontweight="bold")

ax.set_xticks(noise_levels)
ax.set_xticklabels(["0% (Clean)", "5%", "10%", "15%", "20%"])
ax.set_ylim(0.95, 1.015)
ax.set_xlabel("Misspelling Noise Level")
ax.set_ylabel("Score")
ax.set_title("RoBERTa Binary – Robustness to Misspelling Attacks", pad=12)
ax.legend()
ax.grid(True, zorder=0)
plt.tight_layout()
fig.savefig(OUT / "F3_misspelling_robustness.png", bbox_inches="tight")
plt.close(fig)
print("✓ F3_misspelling_robustness.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 – Training curves: val_loss + val_f1 (Blues theme)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
for ax, (task, hist), title in zip(axes, train_history.items(),
                                   ["Binary (Human vs. AI)", "Multi-Class (7 Labels)"]):
    epochs = hist["epoch"]
    ax2 = ax.twinx()
    l1, = ax.plot(epochs,  hist["val_loss"], "o-", color=ACCENT_RED, linewidth=2.5, markersize=6, label="Val Loss")
    l2, = ax2.plot(epochs, hist["val_f1"],   "s--",color=DARK_BLUE,  linewidth=2.5, markersize=6, label="Val F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss", color=ACCENT_RED, fontweight="bold")
    ax2.set_ylabel("Validation Macro F1", color=DARK_BLUE, fontweight="bold")
    ax.tick_params(axis="y", labelcolor=ACCENT_RED)
    ax2.tick_params(axis="y", labelcolor=DARK_BLUE)
    ax.set_title(f"Training Curves – {title}", pad=10)
    ax.grid(True, zorder=0)
    ax.set_xticks(epochs)
    ax.legend(handles=[l1, l2], loc="center right")

plt.tight_layout()
fig.savefig(OUT / "F4_training_curves.png", bbox_inches="tight")
plt.close(fig)
print("✓ F4_training_curves.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 – Cross-domain vs Clean comparison (binary - Blues theme)
# ══════════════════════════════════════════════════════════════════════════════
cd_labels = ["Clean\nTest", "Essay", "Wikipedia", "Reuters"]
cd_f1     = [
    df_bin[df_bin["Dataset"]=="Clean_Test"]["Macro_F1"].values[0],
    df_bin[df_bin["Dataset"]=="CrossDomain_Essay"]["Macro_F1"].values[0],
    df_bin[df_bin["Dataset"]=="CrossDomain_WP"]["Macro_F1"].values[0],
    df_bin[df_bin["Dataset"]=="Unseen_Reuters"]["Macro_F1"].values[0],
]
colors_cd = [DARK_BLUE, ACCENT_RED, ACCENT_RED, ACCENT_RED]

fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
bars = ax.bar(cd_labels, cd_f1, color=colors_cd, edgecolor=DARK_BLUE, linewidth=0.8, zorder=3, width=0.5)
for bar, val in zip(bars, cd_f1):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10, color="#111111", fontweight="bold")
ax.axhline(0.5, color=ACCENT_ORANGE, linestyle="--", linewidth=1.2, label="Random baseline (50%)")
ax.set_ylim(0, 1.12)
ax.set_ylabel("Macro F1-Score")
ax.set_title("Cross-Domain Generalization Failure – RoBERTa Binary", pad=12)
ax.legend()
ax.grid(axis="y", zorder=0)
plt.tight_layout()
fig.savefig(OUT / "F5_cross_domain_failure.png", bbox_inches="tight")
plt.close(fig)
print("✓ F5_cross_domain_failure.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 – Binary overview Heatmap (Seaborn 'Blues' colormap - exact F1 style)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300)
heatmap_data = np.array([
    [0.9978, 0.9903, 0.9936, 0.9944, 0.9960, 0.3529, 0.3752, 0.3658, 0.4786, 1.0000],
])
datasets_heat = ["Clean", "Misspell\n5%", "Misspell\n10%", "Misspell\n15%", "Misspell\n20%",
                 "Essay", "Wikipedia", "Reuters", "Paraphrase\n(Gemma)", "Translation\n(NLLB)"]
models_heat = ["RoBERTa\nBinary"]

sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="Blues",
            xticklabels=datasets_heat, yticklabels=models_heat,
            cbar_kws={"label": "Macro F1"}, ax=ax, vmin=0.3, vmax=1.0)

ax.set_title("RoBERTa Binary – Macro F1 Heatmap Across All Attack Types", pad=12)
plt.tight_layout()
fig.savefig(OUT / "F6_binary_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("✓ F6_binary_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7 – Fingerprint-Overwrite Effect (Blues vs Accent Red)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
categories = ["Binary Detection", "Multi-Class\nAttribution"]
f1_clean   = [0.9978, 0.9722]
f1_para    = [0.4786, 0.0980]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, f1_clean, w, label="Clean Test",        color=DARK_BLUE,  edgecolor=DARK_BLUE)
b2 = ax.bar(x + w/2, f1_para,  w, label="Paraphrased (Gemma)", color=ACCENT_RED, edgecolor=ACCENT_RED)
for b, v in zip(b1, f1_clean):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.4f}", ha="center", fontsize=10, color="#111111", fontweight="bold")
for b, v in zip(b2, f1_para):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.015, f"{v:.4f}", ha="center", fontsize=10, color="#111111", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Macro F1-Score")
ax.set_title("Fingerprint-Overwrite Effect (Gemma-27B Paraphrasing)", pad=12)
ax.legend()
ax.grid(axis="y", zorder=0)
plt.tight_layout()
fig.savefig(OUT / "F7_fingerprint_overwrite.png", bbox_inches="tight")
plt.close(fig)
print("✓ F7_fingerprint_overwrite.png")

print("\n✅ All figures updated to F1.png color style and saved in:", OUT.resolve())
