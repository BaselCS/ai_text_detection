"""Task 4: Perturbation magnitude per paraphraser.
BERTScore-F1 (roberta-large) + sentence BLEU (sacrebleu) between
Original_Text (reference) and Paraphrased_Text (hypothesis).
Reports mean +- std per paraphraser. GPU.
NOTE: Gemma-27B has no raw pairs (only paraphrased text in
data/need_processing/paraphrased_articles.csv, originals lost) -> excluded.
"""
import pandas as pd
from bert_score import score as bert_score
from sacrebleu import sentence_bleu

CONFIGS = [
    ("Gemma-2-9B",  "data/processed/raw_paraphrased_gemma2.csv"),
    ("Qwen-2",      "data/processed/raw_paraphrased_qwen2.csv"),
    ("Mistral-7B",  "data/processed/raw_paraphrased_mistral7b.csv"),
    ("LLaMA-3",     "data/processed/raw_paraphrased_llama3.csv"),
    ("GPT-4o-mini", "data/processed/raw_paraphrased_gpt4o_mini.csv"),
]

rows = []
for pname, csv in CONFIGS:
    df = pd.read_csv(csv)
    refs = df["Original_Text"].astype(str).tolist()
    hyps = df["Paraphrased_Text"].astype(str).tolist()
    print(f"{pname}: {len(df)} pairs -> BERTScore...", flush=True)
    _, _, F1 = bert_score(hyps, refs, lang="en", model_type="roberta-large",
                          batch_size=64, device="cuda", verbose=False)
    f1 = F1.cpu().numpy()
    print(f"{pname}: BLEU...", flush=True)
    bleu = [sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)]
    import numpy as np
    bleu = np.array(bleu)
    rows.append({"Paraphraser": pname, "N": len(df),
                 "BERTScore_F1_mean": round(float(f1.mean()), 4),
                 "BERTScore_F1_std": round(float(f1.std()), 4),
                 " sentBLEU_mean": round(float(bleu.mean()), 2),
                 "sentBLEU_std": round(float(bleu.std()), 2)})
    print(f"{pname}: BERTScore-F1={f1.mean():.4f}±{f1.std():.4f} BLEU={bleu.mean():.2f}±{bleu.std():.2f}")

out = pd.DataFrame(rows)
out.to_csv("results/phase2/T19_Perturbation_Magnitude.csv", index=False)
print("\nSaved -> results/phase2/T19_Perturbation_Magnitude.csv")
print(out.to_string(index=False))
