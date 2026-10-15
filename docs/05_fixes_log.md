# Recomputed Accuracy and Macro F1 — All Six Paraphrasers (Alt-GT)

Recomputed 2026-10-14 via `src/phase2/task1_alt_gt.py` (saved `Multi_BERT` XGBoost, no retraining).
Numbers match the previously saved tables exactly — verified.

Alt-GT: every document in a set is relabelled to the paraphraser's nearest training class.

| Paraphraser | Alt_Class | N | Orig_Acc | Orig_MacroF1 | Alt_Acc (attr-rate) | Alt_MacroF1 (7-class) | F1_paraphraser_class |
|---|---|---|---|---|---|---|---|
| Gemma-27B | Gemma-2-9B | 2072 | 0.2457 | 0.1692 | 0.8147 | 0.1283 | 0.8979 |
| Gemma-2-9B | Gemma-2-9B | 2098 | 0.2688 | 0.1991 | 0.7703 | 0.1243 | 0.8702 |
| Qwen-2 | Qwen-2-72B | 2099 | 0.6379 | 0.6293 | 0.0462 | 0.0126 | 0.0883 |
| Mistral-7B | Mistral-7B | 2100 | 0.4100 | 0.3854 | 0.0395 | 0.0109 | 0.0760 |
| LLaMA-3 | Llama-8B | 2100 | 0.3448 | 0.3210 | 0.0348 | 0.0096 | 0.0672 |
| GPT-4o-mini | GPT-4o | 2100 | 0.4633 | 0.4351 | 0.0710 | 0.0189 | 0.1325 |

Top-3 predicted per set: Gemma-27B → Gemma-2-9B 1688, Human 311, GPT-4o 43;
Gemma-2-9B → Gemma-2-9B 1616, Human 317, GPT-4o 136;
Qwen-2 → Gemma-2-9B 750, Mistral-7B 316, GPT-4o 315;
Mistral-7B → Gemma-2-9B 1349, GPT-4o 338, Yi-Large 130;
LLaMA-3 → Gemma-2-9B 890, Mistral-7B 455, Yi-Large 341;
GPT-4o-mini → Yi-Large 847, Gemma-2-9B 621, Mistral-7B 223.

## AI-only variant (Human-origin rows excluded)

| Paraphraser | Alt_Class | N_AI | Orig_Acc | Orig_MacroF1 | Alt_Acc | Alt_MacroF1 |
|---|---|---|---|---|---|---|
| Gemma-27B | Gemma-2-9B | 1776 | 0.1785 | 0.0814 | 0.8981 | 0.1352 |
| Gemma-2-9B | Gemma-2-9B | 1798 | 0.2136 | 0.1194 | 0.8398 | 0.1304 |
| Qwen-2 | Qwen-2-72B | 1800 | 0.6172 | 0.5166 | 0.0472 | 0.0129 |
| Mistral-7B | Mistral-7B | 1800 | 0.4228 | 0.3274 | 0.0461 | 0.0126 |
| LLaMA-3 | Llama-8B | 1800 | 0.3778 | 0.3025 | 0.0378 | 0.0104 |
| GPT-4o-mini | GPT-4o | 1800 | 0.5156 | 0.4883 | 0.0822 | 0.0253 |

## Verified identities (all six rows)

1. `F1_paraphraser = 2·Acc / (1+Acc)` — Precision of the target class = 1
   (no false positives when all ground truth labels are the same class),
   Recall = Acc. Example: `2×0.8147/1.8147 = 0.8979`.
2. `MacroF1_7class = F1_paraphraser / 7` — the other 6 classes have zero
   support, so sklearn gives them F1 = 0 (zero_division=0).
   Example: `0.8979/7 = 0.1283`.

Both identities checked programmatically on the recomputed CSV — hold for all six rows.

## Footnote to add under Table Item 1 in the DOCX

> *Note: Since relabelling assigns all documents in a set to a single target class, class-specific F1 for the paraphraser class is reported instead of unweighted 7-class Macro F1, where 6 classes have zero support (hence Macro F1 ≈ F1/7).*

## Notes

- Keep column 6 as-is (paraphraser-class F1 — scientifically the correct choice).
- The repo's `results/Over All Methodology.docx` contains no tables (33 text paragraphs only), so the footnote must be pasted manually under Table Item 1 in the supervisor's copy.
- `results/tables/T16_Alternative_GT.csv` was rewritten in clean form (single `F1_paraphraser_class` column).
- Source script: `src/phase2/task1_alt_gt.py`.

---

# Item 4 fix — Gemma-27B perturbation row (BERTScore + sentBLEU)

**Problem:** `T19_Perturbation_Magnitude.csv` had only 5 rows — Gemma-27B was missing
because `data/need_processing/paraphrased_articles.csv` stores paraphrased text only
(`Writer` + `Article`, N=2094, no `Original_Text`).

**Fix:** reconstructed Gemma-27B pairs by TF-IDF top-1 matching of each paraphrase
against the base corpus `data/need_processing/articles.csv` (N=51247, 7321 per Writer),
within the same Writer. Match quality: mean top-1 cosine 0.638, mean margin over
runner-up 0.311, only 5.8% with margin < 0.05. Then BERTScore-F1 (`roberta-large`,
same protocol as `task4_perturb.py`) + sacrebleu sentence-BLEU on the 2094 pairs.

**New Gemma-27B row:** N=2094, BERTScore-F1 0.8879±0.0247, sentBLEU 10.72±6.44 —
virtually identical to Gemma-2-9B (0.8869 / 10.86), confirming both Gemma models
are the most aggressive rewriters.

## Corrected 6-row table (`results/phase2/T19_Perturbation_Magnitude.csv`)

| Paraphraser | N | BERTScore-F1 mean±std | sentBLEU mean±std |
|---|---|---|---|
| Gemma-27B | 2094 | 0.8879 ±0.0247 | 10.72 ±6.44 |
| Gemma-2-9B | 2100 | 0.8869 ±0.0251 | 10.86 ±7.37 |
| Qwen-2 | 2100 | 0.9457 ±0.0323 | 51.53 ±15.95 |
| Mistral-7B | 2100 | 0.9268 ±0.0357 | 38.09 ±18.51 |
| LLaMA-3 | 2100 | 0.9130 ±0.0358 | 36.84 ±18.18 |
| GPT-4o-mini | 2100 | 0.9384 ±0.0298 | 36.26 ±12.97 |

Artifacts: `data/processed/raw_paraphrased_gemma27b.csv` (2094 matched pairs),
`results/phase2/T19_Gemma27B_row.csv`, scripts `task4b_gemma27b_match.py` +
`task4c_gemma27b_scores.py`. Also fixed a leading-space typo in the
`sentBLEU_mean` column header of T19.

---

# Item 7 fix — missing RoBERTa_Multi Misspell_5% / Misspell_20% rows

**Problem:** in the Item 7 table, Binary showed Misspell 5% + 20% but Multi showed
only Clean / Paraphrased_Gemma / Translation_NLLB.

**Root cause (verified, not a deletion):** `src/phase2/run_seed.py` (lines 62–66)
evaluates Multi on 3 sets only — Multi-Misspell was never in the protocol, so the
rows exist in no seed file (`T25_all_seeds.csv`, `T25_seed*.csv` checked).

**Fix executed 2026-10-14:** scored the 2 misspelled sets (N=10242 each, `Writer`
labels) with the surviving Multi seed models via `src/phase2/task7b_multi_misspell.py`.
Math identical to `evaluate_roberta_baseline.predict_one` (same chunk function,
mean chunk logits → argmax, eval mode — deterministic, same predictions), only
batched for speed. **Limitation:** seed-42 Multi model dir is deleted from disk
(only `s123/s999/s2024/s8080` remain), so these two rows are **4-seed**, not 5-seed —
flagged as such, existing T25 files untouched; per-seed rows saved separately.

## Per-seed results (`results/phase2/T25b_multi_misspell_4seed.csv`)

| Seed | Dataset | N | Acc | MacroF1 |
|---|---|---|---|---|
| 123 | Misspell_5% | 10242 | 0.4383 | 0.4231 |
| 123 | Misspell_20% | 10242 | 0.3898 | 0.3728 |
| 999 | Misspell_5% | 10242 | 0.7035 | 0.6917 |
| 999 | Misspell_20% | 10242 | 0.7076 | 0.6939 |
| 2024 | Misspell_5% | 10242 | 0.4580 | 0.4459 |
| 2024 | Misspell_20% | 10242 | 0.4186 | 0.4123 |
| 8080 | Misspell_5% | 10242 | 0.6199 | 0.6152 |
| 8080 | Misspell_20% | 10242 | 0.6203 | 0.5934 |

## The two rows to add (4-seed mean±std — label as 4-seed, NOT 5-seed)

| Model | Dataset | Acc mean±std | MacroF1 mean±std |
|---|---|---|---|
| RoBERTa_Multi | Misspell_5% | 0.5549 ±0.1282 | 0.5440 ±0.1305 |
| RoBERTa_Multi | Misspell_20% | 0.5341 ±0.1546 | 0.5181 ±0.1515 |

## Honest finding to disclose

Seed variance is extreme and bimodal: seeds 999/8080 stay robust under misspelling
(~0.62–0.71), seeds 123/2024 collapse (~0.39–0.46) — unlike Binary, which is stable
across all seeds (Misspell_5% 0.9830±0.0227, Misspell_20% 0.9902±0.0120). So the
mean±std above has large std; report per-seed values alongside, and note seed-42
is missing (model dir deleted — would need retraining to complete a true 5-seed mean).
