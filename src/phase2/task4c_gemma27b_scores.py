"""Task 4c: BERTScore + sentBLEU for reconstructed Gemma-27B pairs (same protocol as task4)."""
import pandas as pd
import numpy as np
from bert_score import score as bert_score
from sacrebleu import sentence_bleu

df = pd.read_csv("data/processed/raw_paraphrased_gemma27b.csv")
refs = df["Original_Text"].astype(str).tolist()
hyps = df["Paraphrased_Text"].astype(str).tolist()
print(f"Gemma-27B: {len(df)} pairs -> BERTScore (roberta-large, cuda)...", flush=True)
_, _, F1 = bert_score(hyps, refs, lang="en", model_type="roberta-large",
                      batch_size=64, device="cuda", verbose=True)
f1 = F1.cpu().numpy()
print("BLEU...", flush=True)
bleu = np.array([sentence_bleu(h, [r]).score for h, r in zip(hyps, refs)])
print(f"BERTScore-F1={f1.mean():.4f}±{f1.std():.4f} BLEU={bleu.mean():.2f}±{bleu.std():.2f}")
# save small summary for appending
pd.DataFrame([{"Paraphraser": "Gemma-27B", "N": len(df),
               "BERTScore_F1_mean": round(float(f1.mean()), 4),
               "BERTScore_F1_std": round(float(f1.std()), 4),
               "sentBLEU_mean": round(float(bleu.mean()), 2),
               "sentBLEU_std": round(float(bleu.std()), 2)}]).to_csv(
    "results/phase2/T19_Gemma27B_row.csv", index=False)
print("Saved -> results/phase2/T19_Gemma27B_row.csv")
