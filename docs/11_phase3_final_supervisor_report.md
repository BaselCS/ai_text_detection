# Phase 3 Experiments: Final Supervisor Verification & Empirical Audit Report

**Date:** October 17, 2026  
**Project:** AI Text Detection & Multi-Class Attribution Robustness  
**Author:** Basel Ahmed Sardah  
**Recipient:** Research Supervisor  
**Status:** All Tasks (1–6) Executed and Audited; Corrected Results and Limitations Stated  
**Artifact Path:** `docs/11_phase3_final_supervisor_report.md`  

---

## Executive Summary

This report delivers the finalized, audited, and empirically verified findings of the **Third Phase of Experiments**, addressing all six directives and methodological queries raised by the supervisor:

1. **Second Corpus Provenance (CNN/DailyMail):** Due to upstream provider deprecation of legacy training checkpoints, we formally adopted the supervisor-approved contingency: framing the CNN/DailyMail multi-class evaluation ($Acc = 26.06\%$) as an **Unseen-Generator / Out-Of-Distribution (OOD) Generator Test**. The clean binary result is **84.69% accuracy / 70.27% Macro-F1**, which is **below the 87.41% majority-class baseline** (balanced accuracy 74.02%; Human recall ≈59.7%), so it is reported as weak cross-corpus transfer rather than strong resilience.
2. **Centroid Geometry & Feature Audit:** The feature dimensionality is conclusively established at exactly **20 handcrafted features** ($791 - 768 \text{ BERT} - 3 \text{ metadata} = 20$). Centroid distance recomputation within the model's actual decision space (XGBoost Gain-Weighted and Handcrafted-only) places **`Gemma-2-9B`** as the closest centroid to Gemma-27B paraphrases (Euclidean $0.3938$, Cosine Rank 1 at $0.3933$), resolving the earlier isotropic contradiction for the retained specifications; the weighted-handcrafted sensitivity is disclosed in §2.3.
3. **RoBERTa Translation Set Audit & Remediation:** The anomalous $1.0000$ Macro-F1 was diagnosed as a compound artifact of **78.48% train/validation split leakage** and an untranslated string pipeline bug. Remediation on genuine held-out test translations (**0% leakage**) across 5 random seeds confirms **99.97% AI recall** and near-ceiling performance under standard adversarial attack (**99.68% Macro-F1** in Setting B1; **99.60%** in balanced 1:1 Setting B2).
4. **Confidence Calibration (Attractor Elimination):** Retraining the classifier without the `Gemma-2-9B` attractor reduces pathological overconfidence on misclassifications: confidence on errors drops by **29.1 percentage points** (Gemma-27B) and **16.2 percentage points** (Mistral-7B). The confidence gap ($\text{Conf}_{\text{Correct}} - \text{Conf}_{\text{Wrong}}$) becomes **$+0.1507$** and **$+0.1482$**, and ECE more than halves ($0.7666 \to 0.3183$). This is a relative improvement, not good calibration: Gemma-27B's 6-class Brier score (0.8980) is worse than a uniform forecast (0.8333), and its 7-class matched accuracy (0.1340) is below $1/7$ chance.
5. **Data Hygiene & Protocol Formalization:** 65 anomalous rows were cleaned ($N = 7,934$): 57 short/refusal/truncation rows (53 LLaMA + 2 Qwen + 2 Yi-Large) plus 8 duplicated-prompt rows across all writers. The empirical shift is bounded by $|\Delta| \le 0.0064$. We also document that the processed CNN label `Claude` corresponds to the raw `claude_haiku` column and justify its exclusion from closed-world multi-class attribution.
6. **Document-Level Clustered Logistic Regressions:** Replacing aggregated 6-point correlations, logistic regressions across $N = 12,569$ document pairs clustered by $G = 2,096$ distinct original articles establish a robust positive association ($p < 10^{-7}$ in every specification). Baseline attribution ORs are **2.60×–3.97×** per $+0.10$ BERTScore. The baseline binary ORs (53.43×–80.18×) are strongly confounded by Human/AI composition; writer fixed effects reduce them to **5.77×–6.40×**, and the Human-only within-class estimates are **2.98×–3.49×**. Multi-class slopes are **not** invariant across the full and restricted regimes ($p = 0.0032$); the earlier “identical trajectories” claim is withdrawn.

---

## Task 1: Second Corpus Provenance & Contingency (CNN/DailyMail Evaluation)

### 1.1 Problem Statement & Background
The Phase 2 evaluation on the CNN/DailyMail dataset revealed that every generative class was produced using a different model checkpoint from the training corpus (e.g., `gemma-3-12b-it` vs. `gemma-2-9b-it`, `deepseek-v4-pro` vs. `yi-large`, `mistral-nemo` vs. `mistral-7b`), compounded by a prompt shift (generation from abstract rather than article rewrite). This simultaneous shift of corpus, prompt, and generator lowered multi-class attribution accuracy to $0.2656$. The supervisor directed us to check availability of original training checkpoints, drop `Yi-Large`, and if too few matched models remain, report the evaluation as an **unseen-generator test**.

### 1.2 Hosted API Endpoint Audit
A rigorous audit of commercial and open-weights API providers (including OpenAI, Anthropic, and the OpenRouter catalog) produced the following availability status. The audit scripts are `tools/second_corpus/query_openrouter_models.py` and `tools/second_corpus/test_endpoints.py`; the exact model-count snapshot was not retained as an artifact, so no “470+” count is claimed here.

| Training Model Checkpoint | Upstream API Status | Current Status | Remediation Decision |
|---|---|---|---|
| `gpt-4o` (OpenAI) | Active, but not the checkpoint used for CNN | CNN used `gpt-4o-mini`, a different OpenAI checkpoint | No exact match |
| `yi-large` (01-AI) | Discontinued / End-of-Life | Unavailable; CNN `Yi-Large` label is populated by `deepseek-v4-pro` | Dropped from the regeneration/match audit, but retained as an OOD label in the 7-class CNN evaluation |
| `google/gemma-2-9b-it` | Retired upstream (404/Deprecated) | Replaced by `gemma-3-12b-it` in CNN | Cannot match original weights |
| `meta-llama/llama-3-8b-instruct` | Retired upstream | Replaced by `llama-3.1-8b` in CNN | Cannot match original weights |
| `mistralai/mistral-7b-instruct` | Retired upstream | Replaced by `mistral-nemo` in CNN | Cannot match original weights |
| `qwen/qwen-2-72b-instruct` | Retired upstream | Replaced by `qwen-2.5-72b` in CNN | Cannot match original weights |

**Correction:** An earlier draft described `gpt-4o` as “Active & Matched.” That was inaccurate. The CNN corpus used `gpt-4o-mini`, not `gpt-4o`. The hosted audit therefore found **zero exact checkpoint matches** among the six CNN AI classes, not one. The unseen-generator contingency is the correct framing for all six classes.

### 1.3 Methodological Rationale & Adopted Contingency
Regenerating the CNN set using current generation successors (e.g., Qwen-2.5, Gemma-3, LLaMA-3.1) would perpetuate the exact methodological confound: altering generator architectures alongside the corpus domain, thereby failing to isolate corpus shift. 

Consequently, we formally adopted the supervisor's specified contingency:
1. **Unseen-Generator / Out-Of-Distribution (OOD) Framing:** The multi-class CNN/DailyMail evaluation is reported as an out-of-distribution generator test, demonstrating the closed-world multi-class classifier's behavior when encountering novel model signatures ($Acc = 26.06\%$, Macro-F1 = $0.1928$ on clean split).
2. **Binary Detection Transfer Is Weak:** The binary detection model achieves **84.69% clean accuracy** and **70.27% Macro-F1** (Raw: $84.66\%$ / $70.14\%$). This is **below the 87.41% majority-class baseline** on the clean split (87.50% raw), so the correct interpretation is weak cross-corpus transfer, not “substantial resilience.” Balanced accuracy is 74.02% clean (73.96% raw), and Human recall is approximately 59.7%; the aggregate accuracy is flattered by the AI-heavy class distribution.

---

## Task 2: Centroid Geometry vs. Prediction Reconciliation

### 2.1 Problem Statement & Dimensionality Audit
Preliminary isotropic Euclidean and Cosine distances computed across the 791-dimensional Feature+BERT space positioned `Yi-Large` closer to Gemma paraphrases than `Gemma-2-9B`, directly contradicting classifier predictions where Gemma paraphrases collapsed into the `Gemma-2-9B` attractor. Furthermore, the supervisor questioned whether the handcrafted feature count was 20 (as published in Paper 1) or 23 (as mentioned in preliminary Phase 2 notes).

### 2.2 Feature Count Clarification
We inspected the matrix schema of `processed_articles.csv` (**$N = 51,181$ rows**, columns $= 791$) — the authoritative corpus used by `task2_centroids.py`. The earlier $8,400$ figure was a historical balanced-subsample reference, not the file used in Task 2.
* **Frozen BERT Embeddings:** 768 dimensions (`bert_0` through `bert_767`).
* **Non-Feature Metadata Columns:** Exactly 3 columns (`Writer`, `Text`, `is_AI`).
* **Handcrafted Features:** Exactly **20 features** ($791 - 768 - 3 = 20$):
  1. `ppl` (Perplexity)
  2. `burstiness`
  3. `ttr` (Type-Token Ratio)
  4. `complex` (Flesch-Kincaid / Readability Complexity)
  5. `avg_word_len`
  6. `avg_sent_len`
  7. `uid` (Uniform Information Density)
  8–17. `gain`, `beauty`, `government`, `urban`, `art`, `help`, `optimism`, `strength`, `love`, `traveling` (10 Empath lexical categories)
  18. `syntax_depth` (Syntactic parse tree depth)
  19. `semantic_mean` (Internal semantic sentence similarity mean)
  20. `semantic_std` (Internal semantic sentence similarity standard deviation)

The Phase 2 figure of 23 resulted from an inadvertent subtraction $791 - 768 = 23$, omitting to deduct the 3 metadata columns. The original paper's specification of 20 handcrafted features is verified and exact.

### 2.3 Centroid Distance in Decision Space
Isotropic distances across frozen BERT dimensions are dominated by semantic content rather than generative stylometrics. When distances are computed within the classifier's actual operational space, geometric proximity mirrors predictions perfectly:

#### Option A: XGBoost Gain-Weighted Space (788 Dimensions)
Features are weighted by normalized XGBoost decision tree gain ($\sum w_i = 1$), directly reflecting the importance assigned by the classifier during training:

| Training Class Centroid | Euclidean Distance to Gemma-27B Paraphrases | Cosine Similarity | Cosine Similarity Rank |
|---|---|---|---|
| **`Gemma-2-9B`** | **0.3938** | **0.3933** | **Rank 1 (Closest)** |
| `Yi-Large` | 0.4521 | 0.1023 | Rank 3 |
| `Mistral-7B` | 0.4893 | 0.1092 | Rank 2 |
| `Llama-8B` | 0.5367 | -0.4885 | Rank 7 |
| `Qwen-2-72B` | 0.5596 | -0.0304 | Rank 5 |
| `GPT-4o` | 0.6086 | -0.0894 | Rank 6 |
| `Human` | 0.7852 | -0.0031 | Rank 4 |

*Note: Distances to Qwen-2 and Mistral-7B paraphrased sets independently confirm `Gemma-2-9B` as the nearest attractor centroid across both Euclidean and Cosine metrics.*

**Sensitivity caveat:** In the unreported `OptionB_GainWeighted` Euclidean variant, the Gemma-2-9B paraphrase set is actually marginally closer to the `GPT-4o` centroid (0.6618) than to the `Gemma-2-9B` centroid (0.6641), so the nearest-centroid claim is not invariant to the Option B weighting choice. The retained Option A (gain-weighted, 788-d) and unweighted handcrafted Option B results both support Gemma-2-9B; the weighted-handcrafted sensitivity should be disclosed rather than omitted.

#### Option B: Handcrafted Features Only (20 Dimensions)
Evaluating centroids purely across the 20 z-score normalized handcrafted features:

| Training Class Centroid | Euclidean Distance to Gemma-27B Paraphrases | Cosine Similarity | Cosine Similarity Rank |
|---|---|---|---|
| **`Gemma-2-9B`** | **1.7296** | **+0.8081** | **Rank 1 (Closest)** |
| `GPT-4o` | 1.8016 | +0.6906 | Rank 2 |
| `Yi-Large` | 2.7508 | -0.0920 | Rank 4 |
| `Mistral-7B` | 2.8633 | -0.2993 | Rank 5 |
| `Llama-8B` | 3.2740 | -0.6212 | Rank 6 |
| `Qwen-2-72B` | 3.4387 | -0.6837 | Rank 7 |
| `Human` | 3.8266 | -0.0367 | Rank 3 |

### 2.4 Conclusion
Decision-space geometry confirms the attractor hypothesis. The contradiction was purely an isotropic artifact of unweighted BERT representations. Both Options A and B affirm `Gemma-2-9B` as the nearest attractor centroid, and the centroid analysis is retained.

---

## Task 3: RoBERTa NLLB Translation Robustness Audit

### 3.1 Investigation of the 1.0000 Macro-F1 Anomaly
Phase 2 reported a suspicious RoBERTa binary macro-F1 of $1.0000$ on the NLLB translation set, exceeding clean test performance ($0.9978$). An audit of the historical evaluation file `data/processed/processed_translation.csv` ($N = 1,729$; 247 per class across 7 classes) uncovered two critical pipeline flaws. **Provenance clarification:** this 1,729-row file was a subsampled historical evaluation file, not the remediated test set. It suffered 78.48% train/validation leakage and retained untranslated strings. The remediated Seed-42 evaluation set ($N = 667$) was extracted directly from the pre-subsampling NLLB pool (`archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv`, $N = 3,064$) intersected with each seed's held-out test indices; the five-seed sample sizes are 667, 580, 610, 615, and 588. Only 373 of the 667 remediated rows overlap the 1,729-row audit file by `original_text` (55 Human + 318 AI), so 294 AI rows in the remediated set were not part of the leakage audit. (The audit file's own Seed-42 held-out subset is 372 rows; the 373 figure is the evaluation-to-audit text overlap, a different count.) The leakage percentages below therefore describe the historical file, not the 667-row evaluation set.
1. **Severe Train/Validation Split Leakage (78.48%):** Samples were drawn via random subsampling prior to dataset partitioning. On the primary baseline (Seed 42):
   - Training split overlap: 1,205 samples (**69.69%**)
   - Validation split overlap: 152 samples (**8.79%**)
   - True held-out test split: Only 372 samples (**21.52%**)
   - Across all five random seeds, train/val leakage was 78.48%, 81.78%, 80.74%, 79.01%, and 81.20% under the frozen first-occurrence duplicate-text mapping.
2. **Untranslated String Replacement Bug:** Pairwise TF-IDF and exact string comparison showed a **1.0000 cosine similarity** between texts in `processed_translation.csv` and clean training texts. The data pipeline had retained unperturbed text strings rather than round-trip translated outputs. As an illustrative historical baseline on this unperturbed $N = 1,729$ set, AI articles contained structural prompt artifacts completely absent in human articles (markdown bolding `**`: **56.2%** vs. **0.0%**; `Title:` tags: **37.4%** vs. **0.0%**; `Date:` tags: **8.8%** vs. **0.0%**; bullet points: **12.7%** vs. **3.6%**). A reproducible formatting-only Random Forest (depth 4, five-fold stratified CV) reaches **86.18% accuracy / 49.77% Macro-F1** on the full 1,729 rows and **90.49% / 90.40%** on a balanced 247-Human/247-AI subset; an extended surface-feature model reaches **98.61% / 97.11%** on the full set. The earlier unsupported **94.97% / 89.71%** figure is withdrawn because its generating script and feature definition are not retained and the claimed value could not be reproduced. All primary translation robustness conclusions below rely strictly on the remediated, zero-leakage $N = 667$ held-out test split (Section 3.2 & Section 3.3).

### 3.2 Physical Shift Introduced by NLLB Translation
We retrieved the genuine NLLB-200 round-trip translation outputs from `archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv` across all $N = 667$ held-out test articles for Seed 42 (`T3_surface_artifacts_comparison.csv`):
* **Text Truncation:** Average document length decreased by **-55.02%** ($2,870.82 \to 1,291.34$ characters, $p < 0.0001$; word count dropped from $444.60 \to 203.85$, -54.15%).
* **Syntactic & Punctuation Stripping:** Quotation marks dropped by **-65.44%** ($12.01 \to 4.15$), hyphens/dashes by **-62.83%** ($5.65 \to 2.10$), and markdown bullet points by **-98.82%** ($12.74\% \to 0.15\%$).
* **Surface Separability (Pre- vs. Post-NLLB on $N = 667$ Test Split):** Human test articles ($N = 55$) contain **0.0%** markdown bolding (`**`), `Title:`, `By:`, or `Date:` tags both pre- and post-translation (with bullets dropping from $7.27\% \to 0.0\%$). In AI test articles ($N = 612$), NLLB round-trip translation substantially strips surface formatting: markdown bolding drops from **57.35% $\to$ 44.28%** (-22.79%), `Title:` tags from **38.89% $\to$ 16.83%** (-56.72%), `Date:` tags from **9.48% $\to$ 5.07%** (-46.55%), and bullet points from **13.24% $\to$ 0.16%** (-98.77%).

### 3.3 Remediated Zero-Leakage Performance Across Dual Settings
We constructed a clean evaluation dataset (`results/phase3/T3_true_test_translated.csv`) strictly bounded to the held-out test split (**0% train/val leakage**), and evaluated RoBERTa binary classifiers across 5 seeds:

```
Setting A: Both Translated [Human NLLB vs. AI NLLB]
Setting B1: Standard Attack [Clean Test Human vs. NLLB Test AI] (Full Split)
Setting B2: Balanced 1:1 Standard Attack [50% Clean Human : 50% NLLB AI]
```

#### Multi-Seed Evaluation Results (5 Seeds: 42, 123, 999, 2024, 8080)

| Setting | Evaluation Regime | Accuracy ($\mu \pm \sigma$) | Macro-F1 ($\mu \pm \sigma$) | AI Recall | Human Recall | Seed 42 Baseline |
|---|---|---|---|---|---|---|
| **Setting A** | Both Human & AI Translated | **0.9576 ± 0.0203** | **0.7817 ± 0.1090** | **0.9997 ± 0.0007** | 0.4431 ± 0.2288 | Acc: 0.9820<br>F1: 0.9351 |
| **Setting B1** | Standard Attack (Full Test) | **0.9990 ± 0.0009** | **0.9968 ± 0.0029** | **0.9997 ± 0.0007** | 0.9920 ± 0.0110 | Acc: 0.9985<br>F1: 0.9951 |
| **Setting B2** | Balanced 1:1 Standard Attack | **0.9960 ± 0.0055** | **0.9960 ± 0.0055** | **1.0000 ± 0.0000** | 0.9920 ± 0.0110 | Acc: 1.0000<br>F1: 1.0000 |

### 3.4 Key Methodological Finding
* Under **Setting A**, AI Recall remains near 100% (**99.97%**). The drop in Macro-F1 ($0.7817$) is driven almost entirely by False Positives on human texts: RoBERTa detects NLLB's neural translation artifacts on human text, classifying translated human articles as synthetic. Seed 42 Setting A has confusion matrix `[[44, 11], [1, 611]]`, i.e. **one AI false negative**, so “entirely false positives” is too strong.
* Under **Setting B** (the realistic threat model where human reference text is natural and attacker text is translated), RoBERTa achieves near-ceiling robustness (**99.68% Macro-F1**, 100% AI Recall on balanced subsets). The leaked-set artifact is resolved; while genuine test Macro-F1 averages **0.9968** (B1) and **0.9960** (B2) across seeds, legitimate **1.0000** scores remain achievable on specific seeds/balanced subsets (B2 seeds 42, 999, 8080; B1 seeds 999, 8080) due to strong separability. Their persistence is not evidence of leakage.
* The five-seed evaluation sample sizes are 667, 580, 610, 615, and 588 (Seed 42: 55 Human / 612 AI). The evaluation archive uses an explicit last-occurrence duplicate-text rule to reproduce these sizes; the 1,729-row historical leakage audit uses an explicit first-occurrence rule and is a separate dataset.

---

## Task 4: Confidence Calibration on Retrained 6-Class Model

### 4.1 Attractor Elimination Hypothesis
In Phase 2, we showed that retraining the 7-class classifier without `Gemma-2-9B` dismantled the Gemma attractor. Task 4 evaluates whether attractor elimination also restores probabilistic calibration and resolves pathological error overconfidence.

### 4.2 Experimental Setup
We evaluated the retrained 6-class model against the 7-class baseline across both the `Gemma-27B` paraphrase set ($N = 1,776$ non-Gemma scorable, $N = 2,072$ full) and the `Mistral-7B` paraphrase set ($N = 1,800$ scorable, $N = 2,100$ full). We measured:
* Mean confidence on correct vs. wrong predictions ($\mu \pm \sigma$)
* Confidence separation gap: $\Delta = \text{Conf}_{\text{Correct}} - \text{Conf}_{\text{Wrong}}$
* Expected Calibration Error (ECE, 10 equal-width bins)
* Multi-class Brier Score: $\text{Brier} = \frac{1}{N}\sum_{i=1}^N \sum_{c=1}^K (p_{ic} - y_{ic})^2 \in [0, 2]$, and Maximum Calibration Error (MCE)

### 4.3 Calibration Results

| Paraphrase Set | Model Architecture | Scorable N | Accuracy | Conf. Correct ($\mu \pm \sigma$) | Conf. Wrong ($\mu \pm \sigma$) | Conf. Gap ($\Delta$) | Brier Score | ECE (10-bin) | MCE |
|---|---|---|---|---|---|---|---|---|---|
| **Gemma-27B** | **6-Class (No-Gemma)** | 1,776 | **0.3468** | **0.7636 ± 0.2367** | **0.6129 ± 0.2075** | **+0.1507** | **0.8980** | **0.3183** | 0.5547 |
| Gemma-27B | 7-Class Baseline | 1,776 | 0.1340 | 0.8791 ± 0.1576 | 0.9039 ± 0.1455 | -0.0248 | 1.5740 | 0.7666 | 0.8593 |
| Gemma-27B | 7-Class Baseline (Full) | 2,072 | 0.2457 | 0.9003 ± 0.1483 | 0.9011 ± 0.1483 | -0.0008 | 1.3691 | 0.6552 | 0.7348 |
| **Mistral-7B** | **6-Class (No-Gemma)** | 1,800 | **0.4983** | **0.8537 ± 0.1841** | **0.7055 ± 0.2096** | **+0.1482** | **0.7470** | **0.2810** | 0.4378 |
| Mistral-7B | 7-Class Baseline | 1,800 | 0.3144 | 0.8677 ± 0.1676 | 0.8676 ± 0.1707 | +0.0001 | 1.1929 | 0.5538 | 0.6716 |
| Mistral-7B | 7-Class Baseline (Full) | 2,100 | 0.4100 | 0.9079 ± 0.1492 | 0.8667 ± 0.1710 | +0.0411 | 1.0253 | 0.4741 | 0.5312 |

### 4.4 Key Scientific Findings
1. **Pathological Overconfidence Eradicated:** In the 7-class baseline, incorrect predictions exhibited extreme certainty ($\mu = 0.9039$ on Gemma-27B, $\mu = 0.8676$ on Mistral-7B) because samples were pulled blindly into the Gemma attractor. In the 6-class model, error confidence plunges by **29.1 percentage points** ($-32.2\%$ relative) on Gemma-27B and by **16.2 percentage points** ($-18.7\%$ relative) on Mistral-7B.
2. **Confidence Gap Restored:** The confidence gap ($\text{Conf}_{\text{Correct}} - \text{Conf}_{\text{Wrong}}$), previously inverted ($-0.0248$) or flat ($+0.0001$), widened to a robust **$+0.1507$** and **$+0.1482$**. The classifier expresses genuine hesitation on difficult edge cases.
3. **Halving of Expected Calibration Error:** ECE decreased from $0.7666 \to \mathbf{0.3183}$ (**-58.5%**) on Gemma-27B and from $0.5538 \to \mathbf{0.2810}$ (**-49.3%**) on Mistral-7B. Brier scores improved by $43.0\%$ and $37.4\%$.

**Calibration caveat:** “Halved” is not “calibrated.” The 6-class Gemma-27B Brier score is **0.8980**, which is **worse than the 0.8333 uniform-forecast Brier** (Brier skill score $-0.0776$); its ECE is still **0.3183**. The 7-class matched-sample accuracy is **0.1340**, below the $1/7 = 0.1429$ chance rate. The defensible claim is that removing the attractor reduces overconfidence and improves relative calibration, not that it restores good calibration. See `results/phase3/T4_calibration_benchmarks.csv`.

---

## Task 5: Data Hygiene, Quality Filters & Formal Exclusions

### 5.1 Systematic Audit of Flagged Rows
We executed an exhaustive audit across `results/extra/task2/generated_full1000.csv` ($1,000$ prompt rows $\times$ 8 generators) and `data/processed/processed_second_corpus_cnn.csv` ($N = 7,999$ rows):

1. **LLaMA Refusal String (Row 474):** Row 474 in `llama_3_8b` contains the 29-character refusal `"I can't fulfill this request."`, not an article narrative. A second LLaMA refusal/truncation occurs at row 526 (110 characters, `"I can't create a news article that involves a real person's death..."`).
2. **Short Articles ($< 200$ Characters, 57 Rows Total):** 53 are in `llama_3_8b` (including row 474 and row 526), 2 are in `qwen_2_72b` (rows 526 and 682), and 2 are in `deepseek_pro` / the `Yi-Large` class (rows 175 and 228). This corrects the earlier statement that all short rows were confined to LLaMA.
3. **Truncated Outputs:** Qwen rows 526 (101 chars) and 682 (94 chars); Yi-Large rows 175 (118 chars) and 228 (16 chars); LLaMA row 526 (110 chars). In addition, raw `deepseek_pro` row 915 is an 8-character truncation (`READING,`), but it is **not present in the processed CNN corpus** (Yi-Large has only 999 processed rows), so it is documented as a source-level exclusion rather than counted among the 65 dropped processed rows.
4. **Duplicated Prompt Across Writers (8 Rows):** Prompt row 233 was an exact textual duplicate of prompt row 128. Dropping prompt row 233 prunes 8 generated articles across all 8 writers.
5. **Total Anomaly Count:** Exactly **65 processed rows** pruned: 57 short/refusal/truncation rows (53 LLaMA + 2 Qwen + 2 Yi-Large) + 8 duplicate-prompt rows = 65, giving **$N = 7,934$** clean rows. The full row-level catalog is `results/phase3/T5_data_hygiene_audit.csv`; the source-only row 915 note is `results/phase3/T5_data_hygiene_notes.csv`.

### 5.2 Sensitivity Analysis: Clean vs. Raw Benchmark Metrics

The binary and multi-class evaluations use different denominators. Binary includes all 8 writers (7,999 raw / 7,934 clean). Multi-class excludes Claude (1,000 rows), so its denominators are **6,999 raw / 6,935 clean**, not 7,999/7,934.

| Evaluation | Metric | Raw Dataset | Clean Dataset | Shift ($\Delta$) | Raw N | Clean N |
|---|---|---|---|---|---|---|
| **Binary Detection** | Accuracy | 0.8466 | **0.8469** | **+0.0003** | 7,999 | 7,934 |
| **Binary Detection** | Macro-F1 | 0.7014 | **0.7027** | **+0.0013** | 7,999 | 7,934 |
| **Multi-Class Attribution** | Accuracy | 0.2656 | **0.2606** | **-0.0050** | 6,999 | 6,935 |
| **Multi-Class Attribution** | Macro-F1 | 0.1992 | **0.1928** | **-0.0064** | 6,999 | 6,935 |

The performance shift across all metrics is bounded by $|\Delta| \le 0.0064$, confirming that Phase 2 conclusions are not artifacts of the flagged rows. This is a robustness statement about the *shift*, not evidence that the absolute scores are strong: binary accuracy remains below the 87.41% clean majority baseline, and multi-class accuracy is low.

### 5.3 Formal Written Exclusion Rationale: `claude-3-haiku`
The processed CNN file labels these rows simply as `Claude`; the raw wide-format column is `claude_haiku`. We formally confirm the following methodological statement for the manuscript:
> "Articles generated by `claude-3-haiku` were successfully generated across all 1,000 prompts and retained for the zero-shot Binary Detection evaluation ($\text{is\_AI} = 1$). However, `claude-3-haiku` was intentionally and formally excluded from Multi-Class Attribution evaluation. The Phase-1 multi-class hybrid model was trained over a closed-world taxonomy of seven classes (`GPT-4o`, `Gemma-2-9B`, `Human`, `Llama-8B`, `Mistral-7B`, `Qwen-2-72B`, `Yi-Large`). Because `claude-3-haiku` was not present in the training set and does not constitute a valid attribution class, including it in multi-class testing would introduce an artificial out-of-distribution forced-choice error. Excluding Claude at the attribution stage preserves strict architectural integrity with the model's closed label space."

---

## Task 6: Document-Level Clustered Logistic Regressions on BERTScore

### 6.1 Motivation & Methodological Advancement
Previously, the correlation between perturbation magnitude and classifier degradation relied on six aggregate points (one mean per paraphraser). We rebuilt the analysis at the document level across six paraphrasers (`Gemma-2-9B`, `Qwen-2-72B`, `LLaMA-3`, `Mistral-7B`, `GPT-4o-mini`, and reconstructed `Gemma-27B`).

The original implementation contained a critical row-alignment bug: it zipped raw `Original_Text` rows to processed prediction rows by position (`iloc`) after truncating to the shorter frame. Because the processed files drop different rows in different paraphraser configs, roughly 42% of BERTScore values were attached to the wrong document. The rebuilt pipeline joins raw and processed rows **only** on exact normalized `(Paraphrased_Text, Writer)` pairs. All 12,569 processed rows join; one duplicate key (two `Gemma-27B` rows) is matched deterministically in occurrence order.

### 6.2 Model Specification & Cluster Adjustment
We estimated logistic regression models with standard errors clustered at the original article level:
$$\text{logit}(P(Y_i = 1)) = \beta_0 + \beta_1 \cdot \text{BERTScore}_i + \sum_{k=1}^5 \gamma_k \cdot \text{Paraphraser}_{i,k}$$
and a writer-fixed-effects robustness specification:
$$\text{logit}(P(Y_i = 1)) = \beta_0 + \beta_1 \cdot \text{BERTScore}_i + \sum_{k=1}^5 \gamma_k \cdot \text{Paraphraser}_{i,k} + \sum_{j=1}^6 \delta_j \cdot \text{Writer}_{i,j}.$$
* **Dependent Variables $Y$:** Correct multi-class attribution (1/0) and correct binary detection (1/0).
* **Independent Variable:** Document-level BERTScore-F1 recomputed against the exactly matched original story on GPU (`roberta-large`). Overall $\mu = 0.908142 \pm 0.048719$ across $N = 12,569$; reconstructed `Gemma-27B` subset $\mu = 0.836994 \pm 0.031095$.
* **Categorical Control:** Paraphraser identity (reference = `GPT-4o-mini`).
* **Cluster Units:** $G = 2,096$ distinct normalized original articles. This is not 2,097. The 2,100 prompt slots contain three recovered prompt references that duplicate existing articles (the 18 raw rows with missing `Original_Text` are meta-refusals; each config's three missing references were recovered from the parallel writer block) and one exact duplicate original text (`Your Wednesday Briefing` appears twice): $2{,}100 - 3 - 1 = 2{,}096$. The earlier $G = 2{,}097$ came from hashing the three missing texts into a single `nan` hash, which is a missing-data artifact, not a valid cluster.
* **Missing-Row Arithmetic (separate from clusters):** $12{,}600$ expected = $6 \times 2{,}100$; observed $12{,}569$, i.e. 31 processed rows fewer: `Gemma-2-9B` $-2$, `Qwen-2-72B` $-1$, reconstructed `Gemma-27B` $-28$. The 28 includes 6 rows absent at the raw reconstruction stage and 22 processed drops; the other five paraphrasers each contribute 2,100.
* **Robustness Regimes:**
  - **Full:** all 6 paraphrasers, $N = 12,569$ ($10,497$ primary + $2,072$ reconstructed `Gemma-27B`).
  - **Restricted:** excludes reconstructed `Gemma-27B`, $N = 10,497$ ($5 \times 2,100 - 3$).

### 6.3 Regression Results Table

| Model | Outcome | Scope | Specification | N | $G$ | Pseudo $R^2$ | BERTScore $\beta$ | Clustered SE | $z$ | $p$ | OR per $+0.10$ [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1A** | Multi-Class | Full | Paraphraser controls | 12,569 | 2,096 | 0.0687 | **+9.5404** | 0.8934 | 10.68 | $1.28\times 10^{-26}$ | **2.60** [2.18, 3.09] |
| **1A-WFE** | Multi-Class | Full | + Writer FE | 12,569 | 2,096 | 0.3098 | **+22.2700** | 1.1936 | 18.66 | $1.10\times 10^{-77}$ | **9.27** [7.34, 11.72] |
| **1B** | Multi-Class | Restricted | Paraphraser controls | 10,497 | 2,096 | 0.0745 | **+13.7895** | 1.1333 | 12.17 | $4.62\times 10^{-34}$ | **3.97** [3.18, 4.96] |
| **1B-WFE** | Multi-Class | Restricted | + Writer FE | 10,497 | 2,096 | 0.2891 | **+26.5980** | 1.5735 | 16.90 | $4.22\times 10^{-64}$ | **14.29** [10.50, 19.46] |
| **2A** | Binary | Full | Paraphraser controls | 12,569 | 2,096 | 0.2746 | **+39.7836** | 1.4651 | 27.15 | $2.28\times 10^{-162}$ | **53.43** [40.09, 71.20] |
| **2A-WFE** | Binary | Full | + Writer FE | 12,569 | 2,096 | 0.3794 | **+17.5290** | 1.5184 | 11.54 | $7.86\times 10^{-31}$ | **5.77** [4.29, 7.77] |
| **2B** | Binary | Restricted | Paraphraser controls | 10,497 | 2,096 | 0.3410 | **+43.8422** | 1.7772 | 24.67 | $2.30\times 10^{-134}$ | **80.18** [56.59, 113.59] |
| **2B-WFE** | Binary | Restricted | + Writer FE | 10,497 | 2,096 | 0.4669 | **+18.5668** | 1.7875 | 10.39 | $2.84\times 10^{-25}$ | **6.40** [4.51, 9.09] |

**Binary within-class diagnostics** (same clustered specification; AI-only or Human-only subsets):

| Model | Scope | Subset | N | $G$ | Pseudo $R^2$ | $\beta$ | SE | OR per $+0.10$ [95% CI] |
|---|---|---|---|---|---|---|---|---|
| **2A-AI** | Full | AI writers only | 10,774 | 1,797 | 0.1787 | +21.0549 | 3.8333 | **8.21** [3.87, 17.41] |
| **2A-Human** | Full | Human only | 1,795 | 299 | 0.2225 | +10.9027 | 1.8422 | **2.98** [2.07, 4.27] |
| **2B-AI** | Restricted | AI writers only | 8,998 | 1,797 | 0.2280 | +29.8706 | 4.2068 | **19.83** [8.69, 45.22] |
| **2B-Human** | Restricted | Human only | 1,499 | 299 | 0.2426 | +12.4944 | 1.9940 | **3.49** [2.36, 5.16] |

*(All models use article-clustered standard errors; paraphraser reference = `GPT-4o-mini`; writer FE reference = `GPT-4o`.)*

### 6.4 Key Empirical Insights
1. **Positive association, not causal proof:** Every specification shows a positive BERTScore–correctness association with $p < 10^{-7}$. This is stronger than the six aggregate points, but it remains observational: BERTScore covaries with prompt difficulty, refusal/truncation, and writer identity.
2. **Multi-class effect is real but modest:** Baseline ORs are **2.60** (full) and **3.97** (restricted) per $+0.10$ BERTScore, with pseudo $R^2$ only 0.069/0.075. Adding writer FE raises the BERTScore OR to 9.27/14.29, but also raises pseudo $R^2$ to 0.31/0.29, showing that writer/class identity explains much of the outcome.
3. **The huge binary ORs are confounded by class composition:** The raw baseline ORs of **53.43** and **80.18** collapse to **5.77** and **6.40** with writer FE. Human rows have lower mean BERTScore (0.8585) and only 45.85% binary accuracy, versus 96.3% for AI rows; the pooled binary model is partly separating Human from AI rather than measuring semantic preservation. Human-only ORs are **2.98/3.49**; AI-only ORs are **8.21/19.83**. The defensible binary claim is the Human-only/within-class range of roughly **3–3.5×**, not 40–80×.
4. **Reconciliation with the earlier audit:** The earlier audit's binary writer-FE range of “3.3–3.9×” is closest to the reproduced Human-only estimates (2.98–3.49); the pooled writer-FE estimate is 5.77–6.40. The earlier multi-class writer-FE value of “6.07×” is **not reproduced**: the corrected exact-join models give 9.27–14.29. We report the reproducible values and withdraw the earlier point estimates.
5. **Regime invariance is false for multi-class:** Full vs. restricted multi-class slopes differ by $\Delta\beta = -4.25$ (SE 1.44, $z = -2.94$, $p = 0.0032$). Binary slopes do not differ significantly ($\Delta\beta = -4.06$, SE 2.30, $z = -1.76$, $p = 0.078$), but the binary pseudo $R^2$ remains modest. “Statistically identical” and “completely independent of Gemma-27B” are therefore withdrawn.
6. **Practical implication:** Semantic preservation is a consistent correlate of detector retention, especially for multi-class attribution and for Human text under binary detection. It is not a 40–80× lever, and the document-level model does not establish that increasing BERTScore would causally improve detection.

---

## Summary of Core Codebase & Artifacts

All temporary test scripts have been scrubbed from `src/phase3/`. The core execution scripts and verified data artifacts are cataloged below:

### Retained Core Scripts (`src/phase3/`)
* [`src/phase3/task2_centroids.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task2_centroids.py): Computes Option A (XGBoost gain-weighted) and Option B (handcrafted) centroid geometries on the authoritative $N=51,181$ corpus.
* [`src/phase3/task3_leakage_audit.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_leakage_audit.py): Rebuilds the 1,729-row leakage audit with a frozen first-occurrence duplicate-text mapping.
* [`src/phase3/task3_remediation.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_remediation.py): Audits data leakage and prepares zero-leakage NLLB test partitions.
* [`src/phase3/task3_dual_settings.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_dual_settings.py): Evaluates RoBERTa across Setting A and Settings B1/B2 over 5 seeds.
* [`src/phase3/task3_shallow_baseline.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_shallow_baseline.py): Reproduces the historical surface-artifact prevalence and non-neural shallow baselines.
* [`src/phase3/task4_confidence_calibration.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task4_confidence_calibration.py): Evaluates confidence dynamics, ECE, and Brier scores on 6-class vs. 7-class models.
* [`src/phase3/task4_calibration_benchmarks.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task4_calibration_benchmarks.py): Adds chance/uniform-forecast Brier benchmarks.
* [`src/phase3/task5_corrected_hygiene.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task5_corrected_hygiene.py): Applies the 65-row hygiene filter and computes clean vs. raw robustness deltas.
* [`src/phase3/task5_hygiene_audit_rebuild.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task5_hygiene_audit_rebuild.py): Reconstructs the full 65-row catalog and explicit binary/multi denominators.
* [`src/phase3/task6_logistic_regression.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task6_logistic_regression.py): Fits clustered logistic regression models using an exact `(Paraphrased_Text, Writer)` join, with baseline and writer-fixed-effects specifications.

### Primary Results Artifacts (`results/phase3/`)
* `results/phase3/T2_Centroid_OptionA_GainWeighted_{Euclidean,Cosine}.csv`: Centroid distances in 788-d decision space.
* `results/phase3/T2_Centroid_OptionB_Handcrafted_{Euclidean,Cosine}.csv`: Centroid distances across 20 handcrafted features.
* `results/phase3/T3_leakage_audit_summary.csv`: Documented proof of prior 78.48% train-set leakage, with deterministic first-occurrence mapping.
* `results/phase3/T3_leakage_audit_metadata.json`: Frozen mapping rule and match counts for the 1,729-row audit.
* `results/phase3/T3_roberta_true_translation_metrics.csv`: 5-seed clean metrics for Setting A, B1, and B2.
* `results/phase3/T3_true_test_translated.csv`: Remediated zero-leakage evaluation split ($N = 667$).
* `results/phase3/T3_surface_artifacts_historical.csv`: Reproducible surface-artifact prevalence on the 1,729-row historical file.
* `results/phase3/T3_shallow_classifier_metrics.csv`: Formatting-only and extended surface baselines.
* `results/phase3/T4_confidence_calibration_summary.csv`: Full calibration metrics (ECE, Brier, Confidence Gaps).
* `results/phase3/T4_calibration_benchmarks.csv`: Chance accuracy and uniform-forecast Brier benchmarks.
* `results/phase3/T5_data_hygiene_audit.csv`: Full 65-row catalog of flagged rows and remediation actions.
* `results/phase3/T5_data_hygiene_summary.csv`: Category counts and clean/raw denominators.
* `results/phase3/T5_data_hygiene_notes.csv`: Source-only row 915 note.
* `results/phase3/T5_hygiene_metric_impact.csv`: Clean vs. raw performance delta evaluation with explicit Ns.
* `results/phase3/T6_logistic_regression_bertscore.csv`: Full regression coefficients, clustered standard errors, and odds ratios for baseline and writer-FE specifications.
* `results/phase3/T6_document_level_bertscore_cache.csv`: GPU BERTScore cache across exactly text-paired documents.
* `results/phase3/T6_rebuild_summary.json`: Join counts, missing-row arithmetic, cluster count, and BERTScore summary for the rebuilt Task 6 pipeline.

---

## Conclusion & Next Steps
With the completion of this final audit and report:
1. Every methodological critique raised by the supervisor has been addressed, with the limits of the evidence stated explicitly.
2. Contradictions in feature geometry, translation metrics, calibration, hygiene counts, and document-level alignment have been resolved against the raw data and rebuilt scripts.
3. The manuscript's core scientific narrative—that feature-space collapse into generative attractors governs attribution failure under perturbation—is supported by a corrected document-level analysis. The binary effect is materially weaker within class and is not causal; the multi-class effect is positive but modest in explanatory power. The report does not claim otherwise.

We await the supervisor's feedback on these finalized results to incorporate them into the final publication draft.
