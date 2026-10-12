"""Watcher: waits for all 4 background seeds, aggregates 5-seed mean+-std
(seed 42 taken from existing T15 reference), saves T25_5seed_summary.csv,
then pops a desktop window + notification. Run once with nohup.
"""
import os
import time
import subprocess
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [123, 999, 2024, 8080]

while True:
    done = [s for s in SEEDS if os.path.exists(os.path.join(DIR, f"T25_seed{s}.csv"))]
    if len(done) == len(SEEDS):
        break
    print(f"watcher: {len(done)}/{len(SEEDS)} seeds done {done}", flush=True)
    time.sleep(300)

# seed 42 reference (existing T15 evaluation of the seed-42 models)
ref = pd.read_csv(os.path.join(DIR, "..", "results", "tables", "T15_full_evaluation.csv"))
ref = ref.rename(columns={"Accuracy": "accuracy", "Macro_F1": "macro_f1", "Count": "count"})
ref["Seed"] = 42

frames = [ref[["Seed", "Model", "Dataset", "count", "accuracy", "macro_f1"]]]
for s in SEEDS:
    df = pd.read_csv(os.path.join(DIR, f"T25_seed{s}.csv"))
    df["Seed"] = s
    frames.append(df[["Seed", "Model", "Dataset", "count", "accuracy", "macro_f1"]])
all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(os.path.join(DIR, "T25_all_seeds.csv"), index=False)

summary = (all_df.groupby(["Model", "Dataset"])[["accuracy", "macro_f1"]]
           .agg(["mean", "std"]).round(4))
summary.to_csv(os.path.join(DIR, "T25_5seed_summary.csv"))
print(summary.to_string())

subprocess.run(["notify-send", "-u", "critical", "RoBERTa 5-seed complete"])
subprocess.run(["zenity", "--info", "--title=RoBERTa 5-seed",
                "--text=All 5 seeds finished.\nSee newtasks2/T25_5seed_summary.csv",
                "--width=400"])
print("watcher: notified", flush=True)
