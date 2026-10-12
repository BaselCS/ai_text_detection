"""Task 7: train+evaluate RoBERTa binary+multi for one seed.
Usage: python newtasks2/run_seed.py <seed>
Reuses train_roberta_baseline / evaluate_roberta_baseline logic.
Models -> results/models/RoBERTa_{Binary,Multi}_s<seed>/ (seed 42 dirs untouched).
Metrics -> newtasks2/T25_seed<seed>.csv
"""
import os
import sys
import shutil

SEED = int(sys.argv[1])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_roberta_baseline as tr
import evaluate_roberta_baseline as ev

tr.SEED = SEED
tr.set_seed(SEED)

for task in ["binary", "multi"]:
    tr.train_model(task, epochs=3, batch_size=8, max_length=tr.MAX_LENGTH)
    src = f"results/models/RoBERTa_{task.capitalize()}"
    dst = f"{src}_s{SEED}"
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    print(f"moved {src} -> {dst}", flush=True)

# restore seed-42 models if they were moved (only when SEED==42 this is a no-op copy)
# eval from the seed-specific dirs
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda")
rows = []
bin_dir = f"results/models/RoBERTa_Binary_s{SEED}"
multi_dir = f"results/models/RoBERTa_Multi_s{SEED}"

bin_sets = {
    "Clean_Test": f"{bin_dir}/test_split.csv",
    "Misspell_5%": "data/processed/processed_misspelled_5.csv",
    "Misspell_20%": "data/processed/processed_misspelled_20.csv",
    "Paraphrased_Gemma": "data/processed/processed_paraphrased_articles.csv",
    "Translation_NLLB": "data/processed/processed_translation.csv",
}
tok = AutoTokenizer.from_pretrained(bin_dir)
mod = AutoModelForSequenceClassification.from_pretrained(bin_dir).to(device)
for name, path in bin_sets.items():
    df = pd.read_csv(path, low_memory=True)
    if "label" not in df.columns:
        df["label"] = df["is_AI"].astype(int)
    m = ev.evaluate_dataset(mod, tok, df, device)
    m.update({"Model": "RoBERTa_Binary", "Dataset": name})
    rows.append(m)
    print(f"seed {SEED} binary {name}: Acc={m['accuracy']} F1={m['macro_f1']}", flush=True)
del mod, tok

with open(os.path.join(multi_dir, "label_mapping.json")) as f:
    label2id = json.load(f)["label2id"]
multi_sets = {
    "Clean_Test": os.path.join(multi_dir, "test_split.csv"),
    "Paraphrased_Gemma": "data/processed/processed_paraphrased_articles.csv",
    "Translation_NLLB": "data/processed/processed_translation.csv",
}
tok = AutoTokenizer.from_pretrained(multi_dir)
mod = AutoModelForSequenceClassification.from_pretrained(multi_dir).to(device)
for name, path in multi_sets.items():
    df = pd.read_csv(path, low_memory=True)
    if "label" not in df.columns:
        df = df[df["Writer"].isin(label2id)].copy()
        df["label"] = df["Writer"].map(label2id)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    m = ev.evaluate_dataset(mod, tok, df, device)
    m.update({"Model": "RoBERTa_Multi", "Dataset": name})
    rows.append(m)
    print(f"seed {SEED} multi {name}: Acc={m['accuracy']} F1={m['macro_f1']}", flush=True)

pd.DataFrame(rows).to_csv(f"newtasks2/T25_seed{SEED}.csv", index=False)
print(f"seed {SEED} DONE -> newtasks2/T25_seed{SEED}.csv", flush=True)
