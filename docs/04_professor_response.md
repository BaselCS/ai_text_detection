# Response to Professor — Second Phase of Experiments (10 Items + 2 Clarifications)

Dear Professor,

As requested, please find below the complete response to the **Second Phase (10 items + 2 clarifications)**. All tables and figures are finalized and ready to attach. Priority 1 is detailed first.

---

## PRIORITY 1 (no new training)

### 1. Alternative Ground Truth — Fingerprint-Overwrite test

Relabelled each paraphrased document with its **paraphraser** as author. Mapping as instructed:
Gemma-27B & Gemma-2-9B → `Gemma-2-9B`, Qwen-2 → `Qwen-2-72B`, Mistral-7B → `Mistral-7B`, LLaMA-3 → `Llama-8B`, GPT-4o-mini → `GPT-4o`.

**Hybrid `Multi_BERT` (XGBoost) inference (All documents, including Human-origin):**

| Paraphraser | Alt class | N | Orig Acc / MacroF1 | Alt Acc (attr-rate) | Alt MacroF1 (7-class) | F1 paraphraser-class | Top-3 predicted |
|---|---|---|---|---|---|---|---|
| **Gemma-27B** | Gemma-2-9B | 2072 | 0.2457 / 0.1692 | **0.8147** | 0.1283 | **0.8979** | Gemma-2-9B 1688, Human 311, GPT-4o 43 |
| **Gemma-2-9B** | Gemma-2-9B | 2098 | 0.2688 / 0.1991 | **0.7703** | 0.1243 | **0.8702** | Gemma-2-9B 1616, Human 317, GPT-4o 136 |
| Qwen-2 | Qwen-2-72B | 2099 | 0.6379 / 0.6293 | 0.0462 | 0.0126 | 0.0883 | Gemma-2-9B 750, Mistral-7B 316, GPT-4o 315 |
| Mistral-7B | Mistral-7B | 2100 | 0.4100 / 0.3854 | 0.0395 | 0.0109 | 0.0760 | Gemma-2-9B 1349, GPT-4o 338, Yi-Large 130 |
| LLaMA-3 | Llama-8B | 2100 | 0.3448 / 0.3210 | 0.0348 | 0.0096 | 0.0672 | Gemma-2-9B 890, Mistral-7B 455, Yi-Large 341 |
| GPT-4o-mini | GPT-4o | 2100 | 0.4633 / 0.4351 | 0.0710 | 0.0189 | 0.1325 | Yi-Large 847, Gemma-2-9B 621, Mistral-7B 223 |

*Note: Since relabelling assigns all documents in a set to a single target class, both the unweighted 7-class Macro F1 (where 6 classes have zero support, hence Macro F1 = F1_target / 7) and the target class-specific F1_paraphraser = 2 * Acc / (1 + Acc) are reported.*

**AI-only variant (Human-origin rows excluded):**

| Paraphraser | Alt class | N_AI | Orig Acc / MacroF1 | Alt Acc | Alt MacroF1 (7-class) |
|---|---|---|---|---|---|
| **Gemma-27B** | Gemma-2-9B | 1776 | 0.1785 / 0.0814 | **0.8981** | 0.1352 |
| **Gemma-2-9B** | Gemma-2-9B | 1798 | 0.2136 / 0.1194 | **0.8398** | 0.1304 |
| Qwen-2 | Qwen-2-72B | 1800 | 0.6172 / 0.5166 | 0.0472 | 0.0129 |
| Mistral-7B | Mistral-7B | 1800 | 0.4228 / 0.3274 | 0.0461 | 0.0126 |
| LLaMA-3 | Llama-8B | 1800 | 0.3778 / 0.3025 | 0.0378 | 0.0104 |
| GPT-4o-mini | GPT-4o | 1800 | 0.5156 / 0.4883 | 0.0822 | 0.0253 |

**Interpretation:** Fingerprint-Overwrite is **confirmed for Gemma paraphrasers only**. All other paraphrasers collapse toward `Gemma-2-9B` (or `Yi-Large`), not toward their own class. This supports framing: Gemma is an aggressive rewriter that overwrites style; others only partially perturb.

> Files: `results/phase2/T16_Alternative_GT.csv`, `T16_Alternative_GT_AIonly.csv`, `task1_alt_gt.py`

---

### 2. Where did the six AI versions of the BBC articles come from?

**Correction: the set is not BBC — it is CNN/DailyMail.** The old "BBC" label has been renamed everywhere to `Second_Corpus_CNN` (`processed_second_corpus_cnn.csv`, `T14_Second_Corpus_CNN_Validation.csv`, `F15_...`).

**Human source (verified):**
`tools/second_corpus/prepare_cnn_dailymail.py` → HuggingFace `abisee/cnn_dailymail` v3.0.0, `test` split, 1000 random samples (`seed=42`). `highlights` → `Prompt` (mean 313 chars), `article` → `Human_story` (mean 3924 chars, all starting with `(CNN)` dateline).

**Generation (verified):** `tools/second_corpus/generate_from_apis.py`

Prompt template:
```text
Write a complete, professional news article based on the following summary/abstract. Do not add any commentary or meta-text — only output the article itself.

Summary:
{prompt}
```

Parameters: `temperature=0.7`, `max_tokens=1024`, async retry (tenacity exponential backoff).

Exact endpoints:
- `gemma_2_9b`: OpenRouter `google/gemma-3-12b-it` (proxy for Gemma-2-9B)
- `qwen_2_72b`: `qwen/qwen-2.5-72b-instruct`
- `llama_3_8b`: `meta-llama/llama-3.1-8b-instruct`
- `mistral_7b`: `mistralai/mistral-nemo`
- `deepseek_pro`: `deepseek/deepseek-v4-pro` (substitute for retired Yi-Large → mapped to `Yi-Large`)
- `claude_haiku`: `anthropic/claude-3-haiku:beta` (dropped at multi-class)
- `gpt-4o-mini`: OpenAI `gpt-4o-mini`

Artifact: `results/extra/task2/generated_full1000.csv` (1000×9, zero nulls) → processed via `helper/cross_corpus_feature_eval.py` to `data/processed/processed_second_corpus_cnn.csv` (7999 rows).

**Zero-shot hybrid results (T14):** Binary Acc 0.8466 / MacroF1 0.7014; Multi Acc 0.2656 / MacroF1 0.1992.

Quality flags to disclose: `llama_3_8b` row 474 refusal, 53/1000 <200 chars; `deepseek_pro` row 915 truncated `READING,`; `qwen_2_72b` row 682 truncated; 1 duplicated Prompt.

> Files: `docs/06_cnn_provenance.md`, `results/tables/T14_Second_Corpus_CNN_Validation.csv`, `results/figures/F15_...png`

---

### 3. RoBERTa attribution confusion matrix under Gemma paraphrase

Model `RoBERTa_Multi`, Gemma-27B set, N=2072, **Acc 0.1684, MacroF1 0.0980, 17.9 ms/doc (RTX 3060)**.

Counts (rows = true, cols = pred):

| True \ Pred | GPT-4o | Gemma-2-9B | Human | Llama-8B | Mistral-7B | Qwen-2-72B | Yi-Large |
|---|---|---|---|---|---|---|---|
| **GPT-4o** | 57 | 207 | 0 | 2 | 0 | 3 | 27 |
| **Gemma-2-9B** | 30 | 250 | 0 | 0 | 0 | 6 | 10 |
| **Human** | 22 | 156 | 1 | 1 | 0 | 77 | 39 |
| **Llama-8B** | 22 | 235 | 0 | 1 | 0 | 25 | 13 |
| **Mistral-7B** | 17 | 251 | 0 | 0 | 0 | 15 | 13 |
| **Qwen-2-72B** | 24 | 252 | 0 | 0 | 0 | 7 | 13 |
| **Yi-Large** | 44 | 214 | 0 | 0 | 1 | 4 | 33 |

75.5% (1565/2072) are predicted as `Gemma-2-9B` — showing the exact same collapse as the hybrid model and confirming Item 1 independently.

> Files: `T17_RoBERTa_Gemma_CM_counts.csv`, `T17_RoBERTa_Gemma_CM_rownorm.csv`, `T17_RoBERTa_Gemma_preds.csv`, `T17_RoBERTa_Gemma_summary.csv`, `F17_RoBERTa_Gemma_CM.png`

---

## PRIORITY 2

### 4. Perturbation magnitude: BERTScore + self-BLEU

Paired original vs. paraphrase across all six paraphrasers, BERTScore-F1 (`roberta-large`) + sacrebleu sentence-BLEU:

| Paraphraser | N | BERTScore-F1 mean±std | sentBLEU mean±std |
|---|---|---|---|
| **Gemma-27B*** | 2094 | 0.8879 ± 0.0247 | 10.72 ± 6.44 |
| **Gemma-2-9B** | 2100 | 0.8869 ± 0.0251 | 10.86 ± 7.37 |
| LLaMA-3 | 2100 | 0.9130 ± 0.0358 | 36.84 ± 18.18 |
| Mistral-7B | 2100 | 0.9268 ± 0.0357 | 38.09 ± 18.51 |
| GPT-4o-mini | 2100 | 0.9384 ± 0.0298 | 36.26 ± 12.97 |
| Qwen-2 | 2100 | 0.9457 ± 0.0323 | 51.53 ± 15.95 |

*\*Note on Gemma-27B: Because `paraphrased_articles.csv` stores paraphrased text only (N=2094), original pairs were reconstructed via within-author TF-IDF top-1 matching against the base corpus `articles.csv` (mean top-1 cosine 0.638, mean margin over runner-up 0.311).*

Both Gemma models (27B and 9B) are virtually identical in rewrite magnitude and represent the most aggressive rewriters (lowest BLEU ~10.7–10.9), whereas Qwen-2 is the most conservative (BLEU 51.53). This directly explains why only Gemma paraphrasers completely overwrite the authorial fingerprint.

> Files: `T19_Perturbation_Magnitude.csv`, `T19_Gemma27B_row.csv`, `raw_paraphrased_gemma27b.csv`, `task4_perturb.py`, `task4b_gemma27b_match.py`, `task4c_gemma27b_scores.py`

---

### 5. Predicted-class distribution (hybrid, column totals)

| Attack | Total | Predicted-Class Distribution (Column Totals) |
|---|---|---|
| **Misspell 5%** | 10242 | Human 5252 (51.3%), GPT-4o 2303, Gemma-2-9B 1786, Yi-Large 386, Llama-8B 207, Mistral-7B 203, Qwen-2-72B 104 |
| **Misspell 20%** | 10242 | **Human 9314 (90.9%)**, GPT-4o 872, Gemma-2-9B 48, rest ~0 |
| **Translation NLLB round-trip** | 1729 | Gemma-2-9B 251, Llama-8B 249, Qwen-2-72B 249, Yi-Large 249, Human 247, Mistral-7B 246, GPT-4o 238 (~14% each, balanced) |

Character-level misspelling creates a strong **Human attractor** (noise → Human), whereas NLLB round-trip translation preserves a balanced class distribution.

> Files: `T20_PredDist_Misspell_5%.csv`, `T20_PredDist_Misspell_20%.csv`, `T20_PredDist_Translation_NLLB.csv`

---

### 6. Retrain hybrid without Gemma-2-9B (6 classes) + re-run

New XGBoost attribution model trained on 6 classes (`Gemma-2-9B` excluded from the label set). Below are the summary metrics and full confusion matrices for the three sets:

#### A. Gemma-27B Paraphrase Set (`T21_CM_Gemma-27B.csv`)
- **Scorable N = 1776** (296 Gemma-origin rows unscorable; those split primarily into Human 93 and GPT-4o 92).
- **Acc: 0.3468 | MacroF1: 0.2893**
- **Column Totals:** Human 579, GPT-4o 529, Yi-Large 317, Mistral-7B 254, Qwen-2-72B 72, Llama-8B 25.

| True \ Pred | GPT-4o | Human | Llama-8B | Mistral-7B | Qwen-2-72B | Yi-Large |
|---|---|---|---|---|---|---|
| **GPT-4o** | 165 | 41 | 1 | 43 | 10 | 36 |
| **Human** | 28 | 249 | 2 | 1 | 2 | 14 |
| **Llama-8B** | 93 | 87 | 19 | 32 | 22 | 43 |
| **Mistral-7B** | 80 | 75 | 0 | 68 | 23 | 50 |
| **Qwen-2-72B** | 69 | 92 | 2 | 50 | 12 | 71 |
| **Yi-Large** | 94 | 35 | 1 | 60 | 3 | 103 |

#### B. Mistral-7B Paraphrase Set (`T21_CM_Mistral-7B.csv`)
- **Scorable N = 1800** | **Acc: 0.4983 | MacroF1: 0.4658**
- **Column Totals:** GPT-4o 637, Mistral-7B 515, Yi-Large 266, Human 189, Llama-8B 122, Qwen-2-72B 71.

| True \ Pred | GPT-4o | Human | Llama-8B | Mistral-7B | Qwen-2-72B | Yi-Large |
|---|---|---|---|---|---|---|
| **GPT-4o** | 285 | 6 | 1 | 3 | 1 | 4 |
| **Human** | 91 | 151 | 18 | 13 | 16 | 11 |
| **Llama-8B** | 63 | 12 | 80 | 87 | 16 | 42 |
| **Mistral-7B** | 54 | 7 | 4 | 202 | 5 | 28 |
| **Qwen-2-72B** | 61 | 12 | 16 | 143 | 33 | 35 |
| **Yi-Large** | 83 | 1 | 3 | 67 | 0 | 146 |

#### C. Second Corpus — CNN/DailyMail (`T21_CM_CNN.csv`)
- **Scorable N = 5999** (6 scorable classes × 1000 minus 1 dropped row; Gemma and Claude excluded).
- **Acc: 0.2430 | MacroF1: 0.2263**
- **Column Totals:** Human 1467, Mistral-7B 1352, Llama-8B 1098, Qwen-2-72B 974, Yi-Large 835, GPT-4o 273.

| True \ Pred | GPT-4o | Human | Llama-8B | Mistral-7B | Qwen-2-72B | Yi-Large |
|---|---|---|---|---|---|---|
| **GPT-4o** | 130 | 10 | 48 | 429 | 20 | 363 |
| **Human** | 0 | 679 | 87 | 6 | 185 | 43 |
| **Llama-8B** | 4 | 11 | 290 | 191 | 417 | 87 |
| **Mistral-7B** | 47 | 58 | 337 | 247 | 228 | 83 |
| **Qwen-2-72B** | 35 | 8 | 254 | 418 | 69 | 216 |
| **Yi-Large** | 57 | 701 | 82 | 61 | 55 | 43 |

**Conclusion:** Without `Gemma-2-9B` in the label set, the Gemma-paraphrased probability mass does **not revert to the original author** — instead, it splits predominantly between `Human` (579) and `GPT-4o` (529).

> Files: `T21_NoGemma_summary.csv`, `T21_CM_Gemma-27B.csv`, `T21_CM_Mistral-7B.csv`, `T21_CM_CNN.csv`, `F21_*.png`, `T21_NoGemma_*.joblib`

---

### 7. RoBERTa 5-seed protocol (42, 123, 999, 2024, 8080)

All 5 seeds are fully trained and evaluated across both Binary and Multi-class suites:

| Model | Dataset | Accuracy (Mean ± Std) | Macro F1 (Mean ± Std) |
|---|---|---|---|
| **RoBERTa_Binary** | Clean_Test | 0.9973 ± 0.0013 | 0.9944 ± 0.0028 |
| **RoBERTa_Binary** | Misspell_5% | 0.9836 ± 0.0232 | 0.9613 ± 0.0583 |
| **RoBERTa_Binary** | Misspell_20% | 0.9904 ± 0.0122 | 0.9790 ± 0.0275 |
| **RoBERTa_Binary** | Paraphrased_Gemma | 0.8595 ± 0.0025 | 0.4778 ± 0.0175 |
| **RoBERTa_Binary** | Translation_NLLB | 0.9988 ± 0.0014 | 0.9976 ± 0.0028 |
| **RoBERTa_Multi** | Clean_Test | 0.9705 ± 0.0105 | 0.9706 ± 0.0104 |
| **RoBERTa_Multi** | Misspell_5% | 0.5981 ± 0.1466 | 0.5818 ± 0.1415 |
| **RoBERTa_Multi** | Misspell_20% | 0.5824 ± 0.1706 | 0.5614 ± 0.1654 |
| **RoBERTa_Multi** | Paraphrased_Gemma | 0.1850 ± 0.0152 | 0.1169 ± 0.0270 |
| **RoBERTa_Multi** | Translation_NLLB | 0.9746 ± 0.0073 | 0.9746 ± 0.0073 |

**Observations:**
1. **Binary vs. Multi under Gemma Paraphrase:** Binary detection remains stable across all 5 seeds (Acc 0.8595 ± 0.0025), whereas Multi-class attribution collapses across all 5 seeds (Acc 0.1850 ± 0.0152).
2. **Bimodal Seed Sensitivity under Misspelling (Multi-class):** Multi-class attribution under character misspelling exhibits strong bimodal seed variance — 3 seeds remain robust (`s42`: 0.7708 / 0.7757; `s999`: 0.7035 / 0.7076; `s8080`: 0.6199 / 0.6203 for 5% / 20%), whereas 2 seeds degrade sharply (`s2024`: 0.4580 / 0.4186; `s123`: 0.4383 / 0.3898), explaining the ~0.15–0.17 std. In contrast, Binary is highly invariant across seeds (note: a single `s2024` outlier at 0.9428 on Binary Misspell_5% slightly elevates its std).
3. **Cross-domain Binary (seed 42):** Essay 0.5051, WP 0.5176, Reuters 0.5150.

> Files: `T25_5seed_summary.csv`, `T25_all_seeds.csv`, `T25_seed*.csv`, `run_seed.py`

---

## PRIORITY 3

### 8. Centroid distances in Features+BERT space (791-d)

Euclidean (nearest) / Cosine (nearest, paraphraser rank):

- **Gemma-27B:** Euc nearest Yi-Large 9.125 (Gemma 9.713); Cos nearest Yi-Large 0.3355, paraphraser rank 3
- **Gemma-2-9B:** Euc Yi-Large 8.661; Cos Yi-Large 0.3265, paraphraser rank 2
- **Qwen-2:** Euc Gemma-2-9B 4.257; Cos Gemma-2-9B 0.5555, paraphraser rank 4
- **Mistral-7B:** Euc Gemma-2-9B 3.590; Cos Gemma-2-9B 0.7597, paraphraser rank 4
- **LLaMA-3:** Euc Mistral-7B 4.897; Cos Mistral-7B 0.7824, paraphraser rank 5
- **GPT-4o-mini:** Euc Yi-Large 7.102; Cos Yi-Large 0.3342, paraphraser rank 5

**Geometric interpretation:** Paraphrased centroids converge into the Gemma / Yi / Mistral subregion and never toward the true author. *(Note: Unweighted Euclidean/Cosine distances in the 791-d space are dominated by the 768 BERT dimensions—placing Yi-Large slightly closer to Gemma paraphrases geometrically—whereas XGBoost uses feature-weighted tree splits that heavily weight the 23 stylometric/structural features, driving actual predictions into `Gemma-2-9B`.)*

> Files: `T22_Centroid_Euclidean.csv`, `T22_Centroid_Cosine.csv`

---

### 9. Hybrid latency per document (measured the same way as RoBERTa)

Steady-state single-document (`batch=1`) latency on RTX 3060 (after excluding one-time `torch.compile` Inductor warmup and cold booster allocation):

- **Stylometric & Linguistic Features (~82.2 ms total):** perplexity, burstiness, TTR, stylometric, UID, empath, syntax, and semantic consistency.
- **BERT Embedding Extraction (GPU steady-state):** `5.0 ms/doc`
- **XGBoost Inference (7 classes, 184 MB ensemble):** `88.9 ms/doc`
- **Total Hybrid Pipeline:** **`176.1 ms/doc`** vs. **RoBERTa `17.9 ms/doc`** (~9.8× slower due to multi-stage feature extraction and large multi-class tree ensemble evaluation).

> Files: `T24_Latency.csv`, `task9_latency.py`

---

### 10. Mean confidence correct vs. wrong

| Set | Acc | Conf Correct (Mean ± Std) | Conf Wrong (Mean ± Std) |
|---|---|---|---|
| **Gemma-27B** | 0.2457 | 0.9003 ± 0.1483 | 0.9011 ± 0.1483 |
| **Gemma-2-9B** | 0.2688 | 0.8923 ± 0.1467 | 0.8825 ± 0.1523 |
| **Qwen-2** | 0.6379 | 0.9040 ± 0.1574 | 0.7615 ± 0.1912 |
| **Mistral-7B** | 0.4100 | 0.9079 ± 0.1492 | 0.8667 ± 0.1710 |
| **LLaMA-3** | 0.3448 | 0.8129 ± 0.1885 | 0.7313 ± 0.1945 |
| **GPT-4o-mini** | 0.4633 | 0.7830 ± 0.1885 | 0.6483 ± 0.1915 |

The model is overconfident and uncalibrated on Gemma paraphrases (correct ≈ wrong ~0.90), because overwritten texts sit deep inside the Gemma decision region; it remains better calibrated on milder paraphrasers (GPT-4o-mini and Qwen-2).

> Files: `T23_Confidence.csv`, `task10_confidence.py`

---

## Clarifications

- **WP = Writing Prompts** (Reddit `r/WritingPrompts` from the MGT Benchmark), not Wikipedia.
- **Essay 396 / WP 398 vs. 400:** During the 11-step feature extraction pipeline (`notebooks/data_engineer.ipynb`), `dropna()` removed 4 Essay and 2 WP rows that produced empty/NaN values or tokenization failures.


Best regards,
Basel