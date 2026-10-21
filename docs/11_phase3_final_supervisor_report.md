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

1. **Second Corpus Provenance (CNN/DailyMail):** The open-weight checkpoints used to train the detector (`google/gemma-2-9b-it`, `meta-llama/Meta-Llama-3-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`, `Qwen/Qwen2-72B-Instruct`) remain publicly available on Hugging Face, but the legacy hosted OpenRouter endpoints are deprecated and return HTTP 404. Full corpus regeneration through local/dedicated GPU hosting was not performed within the compute/time budget. The CNN evaluation is therefore an **Unseen Corpus + Prompt + Generator Test**, not an isolated generator test. Under the original 7-class label space the clean result is **26.06% accuracy / 19.28% Macro-F1**; dropping the `deepseek-v4-pro` slot and retraining a 6-class model gives **30.22% accuracy / 25.28% Macro-F1**. The clean binary result is **84.69% accuracy / 70.27% Macro-F1**, which is **below the 87.41% majority-class baseline** (balanced accuracy 74.02%; Human recall ≈59.7%), so it is weak cross-corpus transfer rather than strong resilience. The endpoint audit snapshot is retained at `results/phase3/T1_endpoint_audit_snapshot.txt`.
2. **Centroid Geometry & Feature Audit:** The feature dimensionality is exactly **20 handcrafted features** ($791$ total columns $= 768$ BERT $+ 20$ handcrafted $+ 3$ metadata), so the model's decision space is **788-dimensional**. Option A (XGBoost gain-weighted) places **`Gemma-2-9B`** closest to Gemma-27B paraphrases (Euclidean $0.3938$; Cosine Rank 1 at $0.3933$). This is not invariant across every handcrafted reweighting: on the Gemma-2-9B paraphrase set the unweighted Option B Euclidean distance is Gemma-2-9B $1.9630$ vs GPT-4o $1.9754$ (0.6% margin), while the weighted Option B variant gives GPT-4o $0.6618$ vs Gemma-2-9B $0.6641$. The centroid evidence supports, but does not “unequivocally” prove, the attractor interpretation; the classifier behaviour and Option A are the strongest evidence.
3. **RoBERTa Translation Set Audit & Remediation:** The anomalous $1.0000$ Macro-F1 was diagnosed as a compound artifact of **78.48% train/validation split leakage** and an untranslated-string pipeline bug. The historical 1,729-row file is an exact-string duplicate of training articles (1,729/1,729 exact matches); the “1.0000 TF-IDF cosine” wording is withdrawn because the audit assigned similarity $1.0$ by construction after exact matching. On the remediated Seed-42 test set ($N=667$), there are **0 exact string matches** to training and the mean nearest-neighbour TF-IDF cosine is **0.6948** (median **0.7165**, maximum **0.9797**, below $0.99$). Across five seeds, Setting A accuracy is **95.76% ± 2.03%** but balanced accuracy is only **72.14% ± 11.41%** because Human $N$ is 35–55 per seed and the 612:55 imbalance inflates accuracy. Setting B1 is **99.90% ± 0.09% accuracy / 99.68% ± 0.29% Macro-F1**; Setting B2 is **99.60% ± 0.55% / 99.60% ± 0.55%**.
4. **Confidence Calibration (Attractor Reduction):** Retraining the classifier without the `Gemma-2-9B` attractor **substantially reduces** pathological overconfidence on misclassifications: error confidence drops by **29.1 percentage points** ($-32.2\%$ relative) on Gemma-27B and **16.2 percentage points** ($-18.7\%$ relative) on Mistral-7B. The confidence gap ($\text{Conf}_{\text{Correct}} - \text{Conf}_{\text{Wrong}}$) becomes **$+0.1507$** and **$+0.1482$**, and ECE more than halves ($0.7666 \to 0.3183$). This is a relative improvement, not good calibration: Gemma-27B's 6-class Brier score (0.8980) is worse than a uniform forecast (0.8333), and its 7-class matched accuracy (0.1340) is below $1/7$ chance. “Eradicated” is therefore replaced by “substantially reduced.”
5. **Data Hygiene & Protocol Formalization:** 65 anomalous rows were cleaned ($N = 7,934$): 57 short/refusal/truncation rows (53 LLaMA, including row 474; 2 Qwen; 2 `deepseek-v4-pro`) plus 8 duplicated-prompt rows across all writers. The largest metric shift is the multi-class Macro-F1 change of **-0.0063**; all other shifts satisfy $|\Delta| \le 0.0050$. We also document that the processed CNN label `Claude` corresponds to the raw `claude_haiku` column and justify its exclusion from closed-world multi-class attribution.
6. **Document-Level Clustered Logistic Regressions:** Replacing aggregated six-point correlations, logistic regressions use **$N = 12,551$ retained rows (12,549 with a valid reference)** clustered by **$G = 2,096$** distinct original articles in the restricted five-paraphraser regime; the extended-pool regime including reconstructed Gemma-27B has **$G = 3,782$**. Baseline attribution ORs are **2.91× (extended)** and **3.92× (restricted)** per $+0.10$ BERTScore; writer fixed effects give **14.19×** and **14.27×**. The pooled binary ORs (58.29× extended, 89.07× restricted) are strongly confounded by Human/AI composition; writer FE reduces them to **6.10×** and **5.95×**. Human-only binary accuracy is **45.85%** (below chance), and the AI-only models are **quasi-separated and do not converge** (`mle_retvals['converged'] = False`), so their point estimates are not reported. The invalid independent-SE “regime invariance” test is withdrawn. A fully interacted model rejects equal BERTScore slopes across paraphrasers (multi-class joint Wald $p = 4.49\times10^{-61}$; binary $p = 1.26\times10^{-37}$), and the interaction test shows Gemma-27B has a materially lower slope than the other five paraphrasers (pooled multi-class interaction $-29.84$, clustered SE $2.34$, $p = 4.12\times10^{-37}$). Task 6 establishes a document-level BERTScore–correctness association and paraphraser-specific slope heterogeneity; it does **not** test attractor geometry, which remains Task 2's question.

---

## Task 1: Second Corpus Provenance & Contingency (CNN/DailyMail Evaluation)

### 1.1 Problem Statement & Background
The Phase 2 evaluation on the CNN/DailyMail dataset revealed that every generative class was produced using a different model checkpoint from the training corpus (e.g., `gemma-3-12b-it` vs. `gemma-2-9b-it`, `deepseek-v4-pro` vs. `yi-large`, `mistral-nemo` vs. `mistral-7b`), compounded by a prompt shift (generation from abstract rather than article rewrite). This simultaneous shift of corpus, prompt, and generator lowered 7-class attribution accuracy to $0.2656$. The supervisor directed us to check availability of original training checkpoints, drop/relabel `Yi-Large`, and if too few matched models remain, report the evaluation under the unseen-shift contingency. Because corpus, prompt, and generator all differ, we report it as an **Unseen Corpus + Prompt + Generator Test**, not an isolated unseen-generator test.

### 1.2 Hosted API Endpoint Audit
A rigorous audit of commercial and open-weight model availability (OpenAI, Anthropic, and the OpenRouter catalogue) produced the following status. The audit scripts are `tools/second_corpus/query_openrouter_models.py` and `tools/second_corpus/test_endpoints.py`; their output is retained at `results/phase3/T1_endpoint_audit_snapshot.txt` (generated 2026-10-21). The hosted probes returned HTTP 404 for the four legacy open-weight endpoints, while the public model weights remain available on Hugging Face.

| Training Model Checkpoint | Hosted API Status (2026-10-21) | Public Weights / Current Successor | Remediation Decision |
|---|---|---|---|
| `gpt-4o` (OpenAI) | Active, but not the checkpoint used for CNN | CNN used `gpt-4o-mini`, a different OpenAI checkpoint | No exact match |
| `yi-large` (01-AI) | Legacy endpoint unavailable | Processed `Yi-Large` slot is populated by `deepseek-v4-pro`; it is not a Yi-Large generation | Dropped from the 6-class evaluation; retained only as an explicitly relabelled `deepseek-v4-pro` OOD class in the 7-class evaluation |
| `google/gemma-2-9b-it` | OpenRouter endpoint HTTP 404 | Weights remain available on Hugging Face; CNN used `gemma-3-12b-it` | Hosted regeneration unavailable; local regeneration not performed |
| `meta-llama/Meta-Llama-3-8B-Instruct` | OpenRouter endpoint HTTP 404 | Weights remain available on Hugging Face; CNN used `llama-3.1-8b` | Hosted regeneration unavailable; local regeneration not performed |
| `mistralai/Mistral-7B-Instruct-v0.3` | OpenRouter endpoint HTTP 404 | Weights remain available on Hugging Face; CNN used `mistral-nemo` | Hosted regeneration unavailable; local regeneration not performed |
| `Qwen/Qwen2-72B-Instruct` | OpenRouter endpoint HTTP 404 | Weights remain available on Hugging Face; CNN used `qwen-2.5-72b` | Hosted regeneration unavailable; local regeneration not performed |

**Correction:** Earlier drafts described the open-weight checkpoints as “officially retired upstream” or “404 / weights cannot be matched.” That is false. The checkpoints remain publicly downloadable; it was the legacy hosted API endpoints that were unavailable. The accurate statement is: **no CNN AI class used the exact training checkpoint**, and full regeneration through local/dedicated GPU hosting was not performed because of compute/time constraints. The hosted audit therefore found **zero exact checkpoint matches** among the six CNN AI classes, not one.

### 1.3 Methodological Rationale & Adopted Contingency
Regenerating the CNN set using current generation successors (e.g., Qwen-2.5, Gemma-3, LLaMA-3.1) would perpetuate the exact methodological confound: altering generator architectures alongside the corpus domain, thereby failing to isolate corpus shift. Local regeneration with the original weights was technically possible but was not performed within the available compute/time budget.

Consequently, we formally adopted the supervisor's specified contingency:
1. **Unseen Corpus + Prompt + Generator Test:** The CNN/DailyMail evaluation is reported as an out-of-distribution test in which corpus, prompt, and generator all shift simultaneously. It should not be attributed solely to novel model signatures. Under the original 7-class closed world (with the `deepseek-v4-pro` slot retained), the clean metrics are **26.06% accuracy / 19.28% Macro-F1** ($N = 6{,}935$); raw metrics are **26.56% / 19.92%** ($N = 6{,}999$). When the `deepseek-v4-pro` slot is formally dropped and a 6-class model is retrained, the clean metrics are **30.22% accuracy / 25.28% Macro-F1** ($N = 5{,}939$); raw metrics are **30.80% / 26.01%** ($N = 6{,}000$). Both are above 6-class or 7-class chance but far below in-distribution performance.
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

**Sensitivity caveat:** The nearest-centroid result is not invariant to every handcrafted reweighting. On the **Gemma-2-9B paraphrase set**, the unweighted `OptionB_Handcrafted` Euclidean distances are Gemma-2-9B **1.9630** vs GPT-4o **1.9754** (a 0.6% margin in favour of Gemma-2-9B), while the `OptionB_GainWeighted` Euclidean variant gives GPT-4o **0.6618** vs Gemma-2-9B **0.6641** (a 0.35% margin in favour of GPT-4o). The Gemma-27B paraphrase set is closer to Gemma-2-9B under both Option B variants (1.7296 vs 1.8016 unweighted; 0.5845 vs 0.5987 weighted). The retained Option A (gain-weighted, 788-d) result is the primary specification; the near-tie must be disclosed rather than described as “unequivocal.”

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
Decision-space geometry supports the attractor hypothesis, but the evidence is specification-dependent. Option A and the unweighted handcrafted Option B place Gemma-2-9B nearest to Gemma-27B paraphrases; the weighted handcrafted Option B produces a near-tie on the Gemma-2-9B paraphrase set (GPT-4o 0.6618 vs Gemma-2-9B 0.6641). The contradiction with the isotropic 791-column analysis is therefore primarily an artifact of unweighted BERT dominance, not proof that every metric agrees. The centroid analysis is retained as supporting evidence, with the near-tie disclosed.

---

## Task 3: RoBERTa NLLB Translation Robustness Audit

### 3.1 Investigation of the 1.0000 Macro-F1 Anomaly
Phase 2 reported a suspicious RoBERTa binary macro-F1 of $1.0000$ on the NLLB translation set, exceeding clean test performance ($0.9978$). An audit of the historical evaluation file `data/processed/processed_translation.csv` ($N = 1,729$; 247 per class across 7 classes) uncovered two critical pipeline flaws. **Provenance clarification:** this 1,729-row file was a subsampled historical evaluation file, not the remediated test set. It suffered 78.48% train/validation leakage and retained untranslated strings. The remediated Seed-42 evaluation set ($N = 667$) was extracted directly from the pre-subsampling NLLB pool (`archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv`, $N = 3,064$) intersected with each seed's held-out test indices; the five-seed sample sizes are 667, 580, 610, 615, and 588. Only 373 of the 667 remediated rows overlap the 1,729-row audit file by `original_text` (55 Human + 318 AI), so 294 AI rows in the remediated set were not part of the leakage audit. (The audit file's own Seed-42 held-out subset is 372 rows; the 373 figure is the evaluation-to-audit text overlap, a different count.) The leakage percentages below therefore describe the historical file, not the 667-row evaluation set.
1. **Severe Train/Validation Split Leakage (78.48%):** Samples were drawn via random subsampling prior to dataset partitioning. On the primary baseline (Seed 42):
   - Training split overlap: 1,205 samples (**69.69%**)
   - Validation split overlap: 152 samples (**8.79%**)
   - True held-out test split: Only 372 samples (**21.52%**)
   - Across all five random seeds, train/val leakage was 78.48%, 81.78%, 80.74%, 79.01%, and 81.20% under the frozen first-occurrence duplicate-text mapping.
2. **Untranslated String Replacement Bug:** An exact-string audit showed that **all 1,729 texts** in `processed_translation.csv` are exact duplicates of training articles (1,729/1,729 exact matches). The earlier “1.0000 TF-IDF cosine” wording is withdrawn: the audit did not compute TF-IDF similarity; it exact-matched strings and then assigned `match_similarity = 1.0` by construction. The data pipeline had retained unperturbed text strings rather than round-trip translated outputs. As an illustrative historical baseline on this unperturbed $N = 1,729$ set, AI articles contained structural prompt artifacts completely absent in human articles (markdown bolding `**`: **56.2%** vs. **0.0%**; `Title:` tags: **37.4%** vs. **0.0%**; `Date:` tags: **8.8%** vs. **0.0%**; bullet points: **12.7%** vs. **3.6%**). A reproducible formatting-only Random Forest (depth 4, five-fold stratified CV) reaches **86.18% accuracy / 49.77% Macro-F1** on the full 1,729 rows and **90.49% / 90.40%** on a balanced 247-Human/247-AI subset; an extended surface-feature model reaches **98.61% / 97.11%** on the full set. The earlier unsupported **94.97% / 89.71%** figure is withdrawn because its generating script and feature definition are not retained and the claimed value could not be reproduced. All primary translation robustness conclusions below rely strictly on the remediated, zero-leakage $N = 667$ held-out test split (Section 3.2 & Section 3.3).

### 3.2 Physical Shift Introduced by NLLB Translation
We retrieved the genuine NLLB-200 round-trip translation outputs from `archive/legacy/old_things/paraphrasing/nllb/with_semantic_similarity.csv` across all $N = 667$ held-out test articles for Seed 42 (`T3_surface_artifacts_comparison.csv`):
* **Text Truncation:** Average document length decreased by **-55.02%** ($2,870.82 \to 1,291.34$ characters, $p < 0.0001$; word count dropped from $444.60 \to 203.85$, -54.15%).
* **Syntactic & Punctuation Stripping:** Quotation marks dropped by **-65.44%** ($12.01 \to 4.15$), hyphens/dashes by **-62.83%** ($5.65 \to 2.10$), and markdown bullet points by **-98.82%** ($12.74\% \to 0.15\%$).
* **Surface Separability (Pre- vs. Post-NLLB on $N = 667$ Test Split):** Human test articles ($N = 55$) contain **0.0%** markdown bolding (`**`), `Title:`, `By:`, or `Date:` tags both pre- and post-translation (with bullets dropping from $7.27\% \to 0.0\%$). In AI test articles ($N = 612$), NLLB round-trip translation substantially strips surface formatting: markdown bolding drops from **57.35% $\to$ 44.28%** (-22.79%), `Title:` tags from **38.89% $\to$ 16.83%** (-56.72%), `Date:` tags from **9.48% $\to$ 5.07%** (-46.55%), and bullet points from **13.24% $\to$ 0.16%** (-98.77%).

#### 3.2.1 Remediated-Set Leakage Check and Shallow Baseline
The remediated 667-row test set has **0 exact string matches** against the 51,181-row training corpus. Against the full training corpus, the mean nearest-neighbour TF-IDF cosine is **0.6948**, the median is **0.7165**, and the maximum is **0.9797** (no row reaches $0.99$; one row is at or above $0.95$). This confirms that the 667-row evaluation is not a re-used training sample. The audit is reproducible via `src/phase3/task3_667_leakage_audit.py`, which writes `T3_667_leakage_audit.csv` and `T3_667_leakage_audit.json` (TF-IDF fit on the combined train + test corpus: stop words removed, unigrams + bigrams, `min_df=2`, `max_features=50,000`).

A non-neural shallow baseline quantifies how much of the held-out signal is available from surface formatting alone (`src/phase3/task3_shallow_baseline_667.py`; results in `T3_shallow_classifier_667_metrics.csv`):

| Protocol | Features | N | Accuracy | Balanced Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| Formatting-only, natural imbalance | bold, `Title:`, `Date:`, bullets | 667 (612 AI / 55 Human) | 0.9175 | **0.5000** | 0.4785 |
| Extended surface, natural imbalance | formatting + length, punctuation, headings, case/digit ratios | 667 (612 AI / 55 Human) | 0.9340 | **0.6000** | 0.6493 |
| Formatting-only, balanced diagnostic | bold, `Title:`, `Date:`, bullets | 110 (55 AI / 55 Human) | 0.7455 | 0.7455 | 0.7278 |
| Extended surface, balanced diagnostic | formatting + length, punctuation, headings, case/digit ratios | 110 (55 AI / 55 Human) | 0.9091 | 0.9091 | 0.9091 |

The natural-imbalance accuracy is inflated by the 612:55 class ratio: formatting-only accuracy is 91.75% but balanced accuracy is exactly 50%, i.e. the model predicts the majority AI class for every row. The balanced diagnostic is the honest comparison. Surface artifacts remain sufficient to separate many AI from Human articles even after NLLB translation, so the RoBERTa result cannot be attributed to neural detection alone.

### 3.3 Remediated Zero-Leakage Performance Across Dual Settings
We constructed a clean evaluation dataset (`results/phase3/T3_true_test_translated.csv`) strictly bounded to the held-out test split (**0% train/val leakage**), and evaluated RoBERTa binary classifiers across 5 seeds:

```
Setting A: Both Translated [Human NLLB vs. AI NLLB]
Setting B1: Standard Attack [Clean Test Human vs. NLLB Test AI] (Full Split)
Setting B2: Balanced 1:1 Standard Attack [50% Clean Human : 50% NLLB AI]
```

#### Multi-Seed Evaluation Results (5 Seeds: 42, 123, 999, 2024, 8080)

| Setting | Evaluation Regime | Accuracy ($\mu \pm \sigma$) | Balanced Accuracy ($\mu \pm \sigma$) | Macro-F1 ($\mu \pm \sigma$) | AI Recall | Human Recall | Seed 42 Baseline |
|---|---|---|---|---|---|---|---|
| **Setting A** | Both Human & AI Translated | **0.9576 ± 0.0203** | **0.7214 ± 0.1141** | **0.7817 ± 0.1090** | **0.9997 ± 0.0007** | 0.4431 ± 0.2288 | Acc: 0.9820<br>F1: 0.9351 |
| **Setting B1** | Standard Attack (Full Test) | **0.9990 ± 0.0009** | **0.9958 ± 0.0054** | **0.9968 ± 0.0029** | **0.9997 ± 0.0007** | 0.9920 ± 0.0110 | Acc: 0.9985<br>F1: 0.9951 |
| **Setting B2** | Balanced 1:1 Standard Attack | **0.9960 ± 0.0055** | **0.9960 ± 0.0055** | **0.9960 ± 0.0055** | **1.0000 ± 0.0000** | 0.9920 ± 0.0110 | Acc: 1.0000<br>F1: 1.0000 |

**Setting A imbalance caveat:** Accuracy is 95.76% ± 2.03%, but balanced accuracy is only **72.14% ± 11.41%**. The per-seed Human sample is small ($N = 35$–$55$) while AI is $N = 532$–$612$, so a model that misses many Human articles still scores above 93% accuracy. The balanced accuracy is the appropriate headline for Setting A; the high accuracy is not evidence of balanced performance.

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
1. **Pathological Overconfidence Substantially Reduced:** In the 7-class baseline, incorrect predictions exhibited extreme certainty ($\mu = 0.9039$ on Gemma-27B, $\mu = 0.8676$ on Mistral-7B) because samples were pulled blindly into the Gemma attractor. In the 6-class model, error confidence drops by **29.1 percentage points** ($-32.2\%$ relative) on Gemma-27B and by **16.2 percentage points** ($-18.7\%$ relative) on Mistral-7B. This is a large relative improvement, not full elimination; 6-class error confidence remains 0.6129 and 0.7055.
2. **Confidence Gap Restored:** The confidence gap ($\text{Conf}_{\text{Correct}} - \text{Conf}_{\text{Wrong}}$), previously inverted ($-0.0248$) or flat ($+0.0001$), widened to a robust **$+0.1507$** and **$+0.1482$**. The classifier expresses genuine hesitation on difficult edge cases.
3. **Halving of Expected Calibration Error:** ECE decreased from $0.7666 \to \mathbf{0.3183}$ (**-58.5%**) on Gemma-27B and from $0.5538 \to \mathbf{0.2810}$ (**-49.3%**) on Mistral-7B. Brier scores improved by $43.0\%$ and $37.4\%$.

**Calibration caveat:** “Halved” is not “calibrated.” The 6-class Gemma-27B Brier score is **0.8980**, which is **worse than the 0.8333 uniform-forecast Brier** (Brier skill score $-0.0776$); its ECE is still **0.3183**. The 7-class matched-sample accuracy is **0.1340**, below the $1/7 = 0.1429$ chance rate. The defensible claim is that removing the attractor reduces overconfidence and improves relative calibration, not that it restores good calibration. See `results/phase3/T4_calibration_benchmarks.csv`.

---

## Task 5: Data Hygiene, Quality Filters & Formal Exclusions

### 5.1 Systematic Audit of Flagged Rows
We executed an exhaustive audit across `results/extra/task2/generated_full1000.csv` ($1,000$ prompt rows $\times$ 8 generators) and `data/processed/processed_second_corpus_cnn.csv` ($N = 7,999$ rows):

1. **LLaMA Refusal String (Row 474):** Row 474 in `llama_3_8b` contains the 29-character refusal `"I can't fulfill this request."`, not an article narrative. Row 474 is one of the 53 LLaMA short rows counted below; it is not an additional row. A second LLaMA refusal/truncation occurs at row 526 (110 characters, `"I can't create a news article that involves a real person's death..."`).
2. **Short Articles ($< 200$ Characters, 57 Rows Total):** 53 are in `llama_3_8b` (including row 474 and row 526), 2 are in `qwen_2_72b` (rows 526 and 682), and 2 are in `deepseek_pro` / the `deepseek-v4-pro` class (rows 175 and 228). This corrects the earlier statement that all short rows were confined to LLaMA.
3. **Truncated Outputs:** Qwen rows 526 (101 chars) and 682 (94 chars); `deepseek-v4-pro` rows 175 (118 chars) and 228 (16 chars); LLaMA row 526 (110 chars). In addition, raw `deepseek_pro` row 915 is an 8-character truncation (`READING,`), but it is **not present in the processed CNN corpus** (`deepseek-v4-pro` has only 999 processed rows), so it is documented as a source-level exclusion rather than counted among the 65 dropped processed rows.
4. **Duplicated Prompt Across Writers (8 Rows):** Prompt row 233 was an exact textual duplicate of prompt row 128. Dropping prompt row 233 prunes 8 generated articles across all 8 writers.
5. **Total Anomaly Count:** Exactly **65 processed rows** pruned: 57 short/refusal/truncation rows (53 LLaMA + 2 Qwen + 2 deepseek-v4-pro) + 8 duplicate-prompt rows = 65, giving **$N = 7,934$** clean rows. The full row-level catalog is `results/phase3/T5_data_hygiene_audit.csv`; the source-only row 915 note is `results/phase3/T5_data_hygiene_notes.csv`.

### 5.2 Sensitivity Analysis: Clean vs. Raw Benchmark Metrics

The binary and multi-class evaluations use different denominators. Binary includes all 8 writers (7,999 raw / 7,934 clean). Multi-class excludes Claude (1,000 rows), so its denominators are **6,999 raw / 6,935 clean**, not 7,999/7,934.

| Evaluation | Metric | Raw Dataset | Clean Dataset | Shift ($\Delta$) | Raw N | Clean N |
|---|---|---|---|---|---|---|
| **Binary Detection** | Accuracy | 0.8466 | **0.8469** | **+0.0003** | 7,999 | 7,934 |
| **Binary Detection** | Macro-F1 | 0.7014 | **0.7027** | **+0.0013** | 7,999 | 7,934 |
| **Multi-Class Attribution** | Accuracy | 0.2656 | **0.2606** | **-0.0050** | 6,999 | 6,935 |
| **Multi-Class Attribution** | Macro-F1 | 0.1992 | **0.1928** | **-0.0063** | 6,999 | 6,935 |

The largest performance shift is the multi-class Macro-F1 change of **-0.0063**; all other shifts satisfy $|\Delta| \le 0.0050$. This confirms that Phase 2 conclusions are not artifacts of the flagged rows. This is a robustness statement about the *shift*, not evidence that the absolute scores are strong: binary accuracy remains below the 87.41% clean majority baseline, and multi-class accuracy is low.

### 5.3 Formal Written Exclusion Rationale: `claude-3-haiku`
The processed CNN file labels these rows simply as `Claude`; the raw wide-format column is `claude_haiku`. We formally confirm the following methodological statement for the manuscript:
> "Articles generated by `claude-3-haiku` were successfully generated across all 1,000 prompts and retained for the zero-shot Binary Detection evaluation ($\text{is\_AI} = 1$). However, `claude-3-haiku` was intentionally and formally excluded from Multi-Class Attribution evaluation. The Phase-1 multi-class hybrid model was trained over a closed-world taxonomy of seven classes (`GPT-4o`, `Gemma-2-9B`, `Human`, `Llama-8B`, `Mistral-7B`, `Qwen-2-72B`, `Yi-Large`). Because `claude-3-haiku` was not present in the training set and does not constitute a valid attribution class, including it in multi-class testing would introduce an artificial out-of-distribution forced-choice error. Excluding Claude at the attribution stage preserves strict architectural integrity with the model's closed label space."

---

## Task 6: Document-Level Clustered Logistic Regressions on BERTScore

### 6.1 Motivation, Pairing Rebuild, and Row Alignment
The earlier correlation between perturbation magnitude and classifier degradation relied on six aggregate points (one mean per paraphraser). We rebuilt the analysis at the document level across six paraphrasers (`Gemma-2-9B`, `Qwen-2-72B`, `LLaMA-3`, `Mistral-7B`, `GPT-4o-mini`, and reconstructed `Gemma-27B`).

The original script attached raw `Original_Text` rows to processed prediction rows by position (`iloc`) and recovered missing references through a parallel writer block. That recovery is invalid: the raw files are stacked writer blocks, so identical prompt indices refer to different articles in different blocks. The rebuilt pipeline (`src/phase3/task6_logistic_regression.py`) joins only on exact normalized `(Paraphrased_Text, Writer)` pairs and uses the committed `src/phase2/task4b_gemma27b_match.py` reconstruction for Gemma-27B. The processed prediction files contain **2,098** Gemma-2-9B rows, **2,099** Qwen-2-72B rows, 2,100 rows each for LLaMA-3, Mistral-7B, and GPT-4o-mini, and 2,072 reconstructed Gemma-27B rows; these are the denominators before the 18 explicit exclusions. The original positional misalignment affected a substantial fraction of rows; an exact 42% count cannot be verified from the retained artifacts and is no longer claimed.

**Gemma-27B pairing validation.** The rebuilt raw file contains 2,094 pairs. Its TF-IDF top-1 match cosine has median **0.6561**, mean **0.6376**, and **0.0%** of pairs below 0.10. Recomputing a paired TF-IDF cosine directly gives median **0.651**, mean **0.635**, and 0.0% below 0.10. The previously retained file had median **0.151**, mean **0.254**, and **34.4%** below 0.10, confirming that it was mispaired. Only **18.1%** (381/2,094) of the reconstructed Gemma-27B originals overlap the 2,097-article pool used by the other five paraphrasers; GPT-4o-mini overlaps 96.6%, while the other writers overlap 2.3–8.0%. Gemma-27B is therefore **not** a same-article comparison. We label the six-paraphraser regime a **Cross-Sample / Extended-Pool Sensitivity Analysis** and keep the five-paraphraser restricted regime as the primary clean comparison.

**Explicit exclusion of 18 mis-recovered rows.** Eighteen rows had missing or invalid references and were previously “recovered” from the wrong writer block. Fifteen are meta-refusals from the five-configuration raw files; the three Gemma-27B rows are valid article texts whose original references could not be matched. All eighteen are listed in `results/phase3/T6_excluded_rows.csv` and dropped, not recovered. After this exclusion and the Gemma-27B rebuild, the cache contains **12,551 rows**; **12,549** have a valid BERTScore reference (two reconstructed Gemma-27B rows still lack a matched original and are excluded from model fitting). The restricted five-paraphraser regime is **N = 10,482**, all valid. After the 18 explicit exclusions, the six configurations contribute 2,069 (reconstructed Gemma-27B) + 2,095 + 2,096 + 2,097 + 2,097 + 2,097 = 12,551 rows; the restricted five-paraphraser subset is 10,482 rows.

### 6.2 Model Specification & Cluster Adjustment
We estimated logistic regression models with standard errors clustered at the original article level:
$$\text{logit}(P(Y_i = 1)) = \beta_0 + \beta_1 \cdot \text{BERTScore}_i + \sum_{k=1}^5 \gamma_k \cdot \text{Paraphraser}_{i,k}$$
and a writer-fixed-effects robustness specification:
$$\text{logit}(P(Y_i = 1)) = \beta_0 + \beta_1 \cdot \text{BERTScore}_i + \sum_{k=1}^5 \gamma_k \cdot \text{Paraphraser}_{i,k} + \sum_{j=1}^6 \delta_j \cdot \text{Writer}_{i,j}.$$
* **Dependent Variables $Y$:** Correct multi-class attribution (1/0) and correct binary detection (1/0).
* **Independent Variable:** Document-level BERTScore-F1 recomputed against the exactly matched original story on GPU (`roberta-large`). Across the 12,549 valid rows, $\mu = 0.916687 \pm 0.038219$; the reconstructed Gemma-27B subset is $\mu = 0.887848 \pm 0.024727$ ($N = 2{,}067$).
* **Categorical Control:** Paraphraser identity (reference = `GPT-4o-mini`).
* **Cluster Units:** $G = 2{,}096$ distinct original articles in the restricted five-paraphraser regime. The extended-pool regime has $G = 3{,}782$ because Gemma-27B contributes mostly different articles (18.1% overlap with the shared pool).
* **Missing-Row Arithmetic:** $12{,}600$ expected $= 6 \times 2{,}100$; 31 processed rows are absent (`Gemma-2-9B` $-2$, `Qwen-2-72B` $-1$, reconstructed `Gemma-27B` $-28$), and 18 mis-recovered rows are explicitly excluded. That gives $12{,}600 - 31 - 18 = 12{,}551$ retained rows, of which 12,549 have a valid BERTScore reference.
* **Robustness Regimes:**
  - **Extended-Pool Sensitivity:** all 6 paraphrasers, $N = 12{,}549$ valid rows, $G = 3{,}782$; not a same-article comparison.
  - **Restricted:** excludes reconstructed `Gemma-27B`, $N = 10{,}482$, $G = 2{,}096$; primary clean comparison.

### 6.3 Regression Results Table

| Model | Outcome | Scope | Specification | N | $G$ | Pseudo $R^2$ | BERTScore $\beta$ | Clustered SE | $z$ | $p$ | OR per $+0.10$ [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1A** | Multi-Class | Extended-Pool | Paraphraser controls | 12,549 | 3,782 | 0.0706 | **+10.6932** | 0.9794 | 10.92 | $9.42\times 10^{-28}$ | **2.91** [2.40, 3.53] |
| **1A-WFE** | Multi-Class | Extended-Pool | + Writer FE | 12,549 | 3,782 | 0.3171 | **+26.5219** | 1.4249 | 18.61 | $2.51\times 10^{-77}$ | **14.19** [10.73, 18.76] |
| **1B** | Multi-Class | Restricted | Paraphraser controls | 10,482 | 2,096 | 0.0737 | **+13.6601** | 1.1394 | 11.99 | $4.09\times 10^{-33}$ | **3.92** [3.14, 4.90] |
| **1B-WFE** | Multi-Class | Restricted | + Writer FE | 10,482 | 2,096 | 0.2883 | **+26.5795** | 1.5767 | 16.86 | $9.18\times 10^{-64}$ | **14.27** [10.47, 19.43] |
| **2A** | Binary | Extended-Pool | Paraphraser controls | 12,549 | 3,782 | 0.2844 | **+40.6549** | 1.3911 | 29.22 | $9.34\times 10^{-188}$ | **58.29** [44.38, 76.57] |
| **2A-WFE** | Binary | Extended-Pool | + Writer FE | 12,549 | 3,782 | 0.3823 | **+18.0784** | 1.5849 | 11.41 | $3.87\times 10^{-30}$ | **6.10** [4.47, 8.32] |
| **2B** | Binary | Restricted | Paraphraser controls | 10,482 | 2,096 | 0.3453 | **+44.8939** | 1.7770 | 25.26 | $7.97\times 10^{-141}$ | **89.07** [62.87, 126.18] |
| **2B-WFE** | Binary | Restricted | + Writer FE | 10,482 | 2,096 | 0.4686 | **+17.8342** | 1.8805 | 9.48 | $2.46\times 10^{-21}$ | **5.95** [4.12, 8.60] |

**Binary within-class diagnostics** (article-clustered; AI-only models are reported only as convergence diagnostics):

| Model | Scope | Subset | N | $G$ | Pseudo $R^2$ | $\beta$ | SE | OR per $+0.10$ [95% CI] | Status |
|---|---|---|---|---|---|---|---|---|---|
| **2A-AI** | Extended-Pool | AI writers only | 10,754 | 3,203 | — | +23.0645 | 4.1023 | not reported | **Not converged (quasi-separation)** |
| **2A-Human** | Extended-Pool | Human only | 1,795 | 579 | 0.2250 | +11.3245 | 1.7892 | **3.10** [2.19, 4.41] | Converged |
| **2B-AI** | Restricted | AI writers only | 8,983 | 1,797 | — | +27.7568 | 6.0880 | not reported | **Not converged (quasi-separation)** |
| **2B-Human** | Restricted | Human only | 1,499 | 299 | 0.2426 | +12.4944 | 1.9940 | **3.49** [2.36, 5.16] | Converged |

The AI-only models fail to converge because the binary detector is correct on **96.3%** of AI rows. Their point estimates are therefore not interpreted. Human-only binary accuracy is **45.85%**, below chance; the pooled binary model is partly separating Human from AI rather than measuring semantic preservation.

### 6.4 Paraphraser-Specific Slopes and Interaction Test

The pooled common-slope model is rejected. A fully interacted model (`Y ~ BERTScore * Paraphraser`, article-clustered SEs) gives the following slopes per $+1.0$ BERTScore:

| Paraphraser | Multi-class $\beta$ | SE | $p$ | Binary $\beta$ | SE | $p$ |
|---|---|---|---|---|---|---|
| GPT-4o-mini | **+28.7430** | 2.4398 | $<10^{-30}$ | **+99.8244** | 7.4442 | $<10^{-30}$ |
| Gemma-2-9B | **-5.2645** | 2.1290 | 0.0134 | **+23.3972** | 2.3440 | $<10^{-24}$ |
| Gemma-27B | **-12.5733** | 2.1906 | $9.48\times 10^{-9}$ | **+21.3184** | 2.2844 | $<10^{-19}$ |
| LLaMA-3 | **+21.4734** | 1.9690 | $<10^{-26}$ | **+62.0006** | 3.5832 | $<10^{-66}$ |
| Mistral-7B | **+14.0317** | 1.9261 | $3.21\times 10^{-13}$ | **+40.1554** | 2.6810 | $<10^{-49}$ |
| Qwen-2-72B | **+8.8765** | 1.4868 | $2.37\times 10^{-9}$ | **+35.2996** | 2.6925 | $<10^{-38}$ |
| **Joint Wald test of equal slopes** | — | — | **$4.49\times 10^{-61}$** | — | — | **$1.26\times 10^{-37}$** |

The scientific pattern is substantive: non-attractor paraphrasers have strongly positive BERTScore slopes, whereas the two Gemma paraphrasers have negative multi-class slopes. Higher-fidelity Gemma rewrites still collapse into the Gemma-2-9B attractor, so better semantic preservation does not rescue attribution for those families. Binary slopes remain positive for all paraphrasers.

The stacked interaction test confirms the same heterogeneity. The direct pooled interaction (Gemma-27B vs the other five) is:

| Outcome | Specification | Interaction $\beta$ | Clustered SE | $z$ | $p$ |
|---|---|---|---|---|---|
| Multi-class | Pooled interaction, no paraphraser controls | **-29.8417** | 2.3445 | -12.73 | $4.12\times 10^{-37}$ |
| Multi-class | Paraphraser controls (Gemma-27B vs GPT-4o-mini) | **-26.2334** | 2.4106 | -10.88 | $1.40\times 10^{-27}$ |
| Multi-class | Paraphraser controls + writer FE | **-36.0332** | 2.4020 | -15.00 | $7.22\times 10^{-51}$ |
| Binary | Pooled interaction, no paraphraser controls | **-16.8744** | 2.5592 | -6.59 | $4.29\times 10^{-11}$ |
| Binary | Paraphraser controls (Gemma-27B vs GPT-4o-mini) | **-23.5755** | 2.8190 | -8.36 | $6.11\times 10^{-17}$ |
| Binary | Paraphraser controls + writer FE | **-25.7111** | 2.7828 | -9.24 | $2.48\times 10^{-20}$ |

The previously reported independent-SE comparison ($\Delta\beta = -4.25$, SE $1.44$, $p = 0.0032$; binary $-4.06$, SE $2.30$, $p = 0.078$) is **withdrawn**. It applied $\sqrt{SE_{\text{full}}^2 + SE_{\text{restricted}}^2}$ to nested estimates that share most of the same rows, which overstates the variance of the difference. The interaction model above is the valid test.

### 6.5 Key Empirical Insights
1. **Positive association, not causal proof:** Every specification shows a positive BERTScore–correctness association with $p < 10^{-7}$. This is stronger than six aggregate points, but it remains observational: BERTScore covaries with prompt difficulty, refusal/truncation, and writer identity.
2. **Multi-class effect is real but modest:** Baseline ORs are **2.91** (extended) and **3.92** (restricted) per $+0.10$ BERTScore, with pseudo $R^2$ only 0.071/0.074. Writer FE raises the OR to 14.19/14.27 but also raises pseudo $R^2$ to 0.32/0.29, showing that writer/class identity explains much of the outcome.
3. **The huge binary ORs are confounded by class composition:** Raw baseline ORs of **58.29** and **89.07** collapse to **6.10** and **5.95** with writer FE. Human rows have lower mean BERTScore and only **45.85%** binary accuracy, versus **96.3%** for AI rows. The pooled binary model is partly separating Human from AI rather than measuring semantic preservation. Human-only ORs are **3.10/3.49**; AI-only models are quasi-separated and not reported. The defensible binary claim is the Human-only range of roughly **3.1–3.5×**, not 50–90×.
4. **Slope heterogeneity is the robust finding:** The common-slope restriction is rejected (multi-class joint Wald $p = 4.49\times 10^{-61}$; binary $p = 1.26\times 10^{-37}$). Gemma-2-9B and Gemma-27B have negative multi-class slopes; GPT-4o-mini, LLaMA-3, Mistral-7B, and Qwen-2-72B have positive slopes. This is consistent with the attractor interpretation but is not itself an attractor-geometry test.
5. **Task 6 does not test Task 2's attractor claim:** Task 6 estimates document-level BERTScore–correctness associations and paraphraser-specific slope heterogeneity. It does not measure centroid geometry, feature-space collapse, or causal attribution. The earlier sentence “the core narrative is supported by a corrected document-level analysis” is therefore replaced by the narrower claim that semantic preservation is a consistent, heterogeneous correlate of detector retention.

---

## Summary of Core Codebase & Artifacts

All temporary test scripts have been scrubbed from `src/phase3/`. The core execution scripts and verified data artifacts are cataloged below:

### Retained Core Scripts (`src/phase3/`)
* [`src/phase3/task2_centroids.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task2_centroids.py): Computes Option A (XGBoost gain-weighted) and Option B (handcrafted) centroid geometries on the authoritative $N=51,181$ corpus.
* [`src/phase3/task3_leakage_audit.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_leakage_audit.py): Rebuilds the 1,729-row leakage audit with a frozen first-occurrence duplicate-text mapping.
* [`src/phase3/task3_remediation.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_remediation.py): Audits data leakage and prepares zero-leakage NLLB test partitions.
* [`src/phase3/task3_dual_settings.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_dual_settings.py): Evaluates RoBERTa across Setting A and Settings B1/B2 over 5 seeds.
* [`src/phase3/task3_shallow_baseline.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_shallow_baseline.py): Reproduces the historical 1,729-row surface-artifact prevalence and non-neural shallow baselines.
* [`src/phase3/task3_shallow_baseline_667.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_shallow_baseline_667.py): Reproduces the 667-row pre/post surface-artifact table and the natural-imbalance and balanced shallow baselines.
* [`src/phase3/task3_667_leakage_audit.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task3_667_leakage_audit.py): Recomputes the exact-string and nearest-neighbour TF-IDF leakage audit for the 667-row test set.
* [`src/phase3/task4_confidence_calibration.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task4_confidence_calibration.py): Evaluates confidence dynamics, ECE, and Brier scores on 6-class vs. 7-class models.
* [`src/phase3/task4_calibration_benchmarks.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task4_calibration_benchmarks.py): Adds chance/uniform-forecast Brier benchmarks.
* [`src/phase3/task5_corrected_hygiene.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task5_corrected_hygiene.py): Applies the 65-row hygiene filter and computes clean vs. raw robustness deltas.
* [`src/phase3/task5_hygiene_audit_rebuild.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task5_hygiene_audit_rebuild.py): Reconstructs the full 65-row catalog and explicit binary/multi denominators.
* [`src/phase3/task6_logistic_regression.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task6_logistic_regression.py): Fits clustered logistic regression models using an exact `(Paraphrased_Text, Writer)` join, explicit 18-row exclusion, per-paraphraser slopes, and interaction tests.
* [`src/phase3/task1_cnn_multiclass_ood.py`](file:///home/basel/Work/python/ai_text_detection/src/phase3/task1_cnn_multiclass_ood.py): Evaluates the 7-class CNN model and trains/evaluates the 6-class model after dropping the `deepseek-v4-pro` slot.
* [`src/phase2/task4b_gemma27b_match.py`](file:///home/basel/Work/python/ai_text_detection/src/phase2/task4b_gemma27b_match.py): Reconstructs Gemma-27B paraphrase pairs by TF-IDF top-1 matching within writer.

### Primary Results Artifacts (`results/phase3/`)
* `results/phase3/T1_cnn_multiclass_ood_metrics.csv`: 6-class and 7-class CNN OOD metrics for raw and clean splits.
* `results/phase3/T1_cnn_{6class,7class}_confusion_{raw,clean}.csv`: Confusion matrices with `deepseek-v4-pro` displayed explicitly.
* `results/phase3/T1_endpoint_audit_snapshot.txt`: Retained OpenRouter catalogue/endpoint probe output.
* `results/phase3/T2_Centroid_OptionA_GainWeighted_{Euclidean,Cosine}.csv`: Centroid distances in 788-d decision space.
* `results/phase3/T2_Centroid_OptionB_Handcrafted_{Euclidean,Cosine}.csv`: Centroid distances across 20 handcrafted features.
* `results/phase3/T3_leakage_audit_summary.csv`: Documented proof of prior 78.48% train-set leakage, with deterministic first-occurrence mapping.
* `results/phase3/T3_leakage_audit_metadata.json`: Frozen mapping rule and match counts for the 1,729-row audit.
* `results/phase3/T3_roberta_true_translation_metrics.csv`: 5-seed clean metrics for Setting A, B1, and B2.
* `results/phase3/T3_true_test_translated.csv`: Remediated zero-leakage evaluation split ($N = 667$).
* `results/phase3/T3_surface_artifacts_comparison.csv`: 667-row pre/post NLLB surface-artifact comparison.
* `results/phase3/T3_surface_artifacts_historical.csv`: Reproducible surface-artifact prevalence on the 1,729-row historical file.
* `results/phase3/T3_shallow_classifier_667_metrics.csv`: 667-row shallow baselines (natural imbalance and balanced diagnostic).
* `results/phase3/T3_667_leakage_audit.csv`: Per-row exact-match flag and nearest-neighbour TF-IDF cosine for the 667-row set.
* `results/phase3/T3_667_leakage_audit.json`: Summary of the 667-row leakage audit (0 exact matches; mean 0.6948, median 0.7165, max 0.9797).
* `results/phase3/T3_shallow_classifier_metrics.csv`: Historical 1,729-row formatting-only and extended surface baselines.
* `results/phase3/T4_confidence_calibration_summary.csv`: Full calibration metrics (ECE, Brier, Confidence Gaps).
* `results/phase3/T4_calibration_benchmarks.csv`: Chance accuracy and uniform-forecast Brier benchmarks.
* `results/phase3/T5_data_hygiene_audit.csv`: Full 65-row catalog of flagged rows and remediation actions.
* `results/phase3/T5_data_hygiene_summary.csv`: Category counts and clean/raw denominators.
* `results/phase3/T5_data_hygiene_notes.csv`: Source-only row 915 note.
* `results/phase3/T5_hygiene_metric_impact.csv`: Clean vs. raw performance delta evaluation with explicit Ns.
* `results/phase3/T6_logistic_regression_bertscore.csv`: Full regression coefficients, clustered standard errors, and odds ratios for baseline and writer-FE specifications.
* `results/phase3/T6_within_class_diagnostics.csv`: Human-only converged estimates and AI-only quasi-separation diagnostics.
* `results/phase3/T6_per_paraphraser_slopes.csv`: Fully interacted per-paraphraser slopes and joint Wald tests.
* `results/phase3/T6_interaction_test.csv`: Pooled and paraphraser-controlled interaction tests.
* `results/phase3/T6_excluded_rows.csv`: The 18 explicitly excluded rows: 15 five-configuration meta-refusals and 3 unmatched Gemma-27B references.
* `results/phase3/T6_document_level_bertscore_cache.csv`: GPU BERTScore cache across exactly text-paired documents.
* `results/phase3/T6_rebuild_summary.json`: Join counts, missing-row arithmetic, cluster counts, pairing validation, and BERTScore summary for the rebuilt Task 6 pipeline.

---

## Conclusion & Next Steps
With the completion of this final audit and report:
1. Every methodological critique raised by the supervisor has been addressed, with the limits of the evidence stated explicitly.
2. Contradictions in feature geometry, translation metrics, calibration, hygiene counts, and document-level alignment have been resolved against the raw data and rebuilt scripts.
3. The manuscript's central narrative—feature-space collapse into generative attractors under perturbation—is supported by the classifier behaviour and the Option A centroid analysis, but the centroid evidence is specification-dependent (Section 2.3). Task 6 does **not** test attractor geometry: it establishes a document-level BERTScore–correctness association and strong paraphraser-specific slope heterogeneity. The binary effect is materially weaker within class and is not causal; the multi-class effect is positive but modest in explanatory power. The report does not claim otherwise.

We await the supervisor's feedback on these finalized results to incorporate them into the final publication draft.
