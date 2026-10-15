# Second Phase of Experiments — Action Plan & Technical Analysis

This document provides a detailed technical breakdown, current status, and execution plan for all items in [Second phase of experiments.md](../Second%20phase%20of%20experiments.md).

---

## 1. Executive Summary & Goals
The supervising professor has outlined specific requirements to elevate the paper for a Q1 journal submission. The primary scientific objectives are:
1. **Validating the "Fingerprint-Overwrite Effect"**: Demonstrating that when text is paraphrased by an LLM, the detector shifts attribution to the paraphraser rather than the original author.
2. **Provenance & Data Verification**: Explaining generation parameters for the second corpus (BBC) and clarifying dataset filtering discrepancies.
3. **Deep Comparative Baseline Analysis**: Benchmarking fine-tuned RoBERTa against the hybrid XGBoost model across multi-seed evaluations, confusion matrices, and latency.

---

## 2. Priority 1 (Urgent — No New Training Required)

### Item 1: Alternative Ground Truth for Paraphrased Sets
- **Objective:** Relabel each paraphrased sample with its paraphraser as the author (rather than original generator) and recompute Accuracy and Macro F1 across all 6 paraphrasers.
- **Label Mapping:**
  - `Gemma-27B` & `Gemma-2-9B` $\rightarrow$ `Gemma-2-9B`
  - `Qwen-2` $\rightarrow$ `Qwen-2-72B`
  - `Mistral-7B` $\rightarrow$ `Mistral-7B`
  - `LLaMA-3` $\rightarrow$ `Llama-8B`
  - `GPT-4o-mini` $\rightarrow$ `GPT-4o`
- **Data Status:** All 6 paraphraser datasets are already extracted and processed in [data/processed/](../data/processed):
  - [processed_paraphrased_articles.csv](../data/processed/processed_paraphrased_articles.csv) (Gemma-27B)
  - [processed_paraphrased_gemma2.csv](../data/processed/processed_paraphrased_gemma2.csv)
  - [processed_paraphrased_gpt4o_mini.csv](../data/processed/processed_paraphrased_gpt4o_mini.csv)
  - [processed_paraphrased_llama3.csv](../data/processed/processed_paraphrased_llama3.csv)
  - [processed_paraphrased_mistral7b.csv](../data/processed/processed_paraphrased_mistral7b.csv)
  - [processed_paraphrased_qwen2.csv](../data/processed/processed_paraphrased_qwen2.csv)
- **Execution Plan:** Run an inference script using saved XGBoost model [Multi_BERT.joblib](../results/models/Multi_BERT.joblib), apply the alternative ground truth mapping, and generate a new summary table comparing original vs. alternative ground truth.

### Item 2: BBC Dataset Provenance
- **Objective:** Detail origins, models, prompts, and sampling settings for the 6 AI versions in the BBC test corpus.
- **Data Status:** The multi-generator outputs reside in [results/extra/task2/generated_full1000.csv](../results/extra/task2/generated_full1000.csv).
- **Details:**
  - Evaluated in [helper/cross_corpus_feature_eval.py](../helper/cross_corpus_feature_eval.py).
  - Models: `claude_haiku`, `gemma_2_9b`, `qwen_2_72b`, `llama_3_8b`, `mistral_7b`, `gpt-4o-mini`, `deepseek_pro`.
  - Prompts and generation scripts document exact prompt templates used to convert human news summaries into full synthetic articles.

### Item 3: RoBERTa Gemma Paraphrase Confusion Matrix
- **Objective:** Extract and provide the multi-class confusion matrix for fine-tuned RoBERTa under the Gemma-27B paraphrasing attack.
- **Data Status:** Already completed in Task 3. Documented in [results/extra/task3/Task3_Results_Report.md](../results/extra/task3/Task3_Results_Report.md) and plotted in [results/extra/task3/F7_fingerprint_overwrite.png](../results/extra/task3/F7_fingerprint_overwrite.png).
- **Execution Plan:** Export the exact numerical confusion matrix into a clean CSV and formatted figure for paper submission.

---

## 3. Priority 2 (Medium Effort & Retraining)

### Item 4: Perturbation Magnitude (BERTScore & Self-BLEU)
- **Objective:** Quantify stylistic and semantic distortion caused by each paraphraser by calculating BERTScore (semantic similarity) and Self-BLEU (n-gram overlap) between original articles and paraphrased versions.
- **Execution Plan:** Use `evaluate` / `bert_score` with `roberta-large` and NLTK BLEU across pairs in `raw_paraphrased_*.csv`. Report mean $\pm$ std per paraphraser.

### Item 5: Predicted-Class Distribution under Attacks
- **Objective:** Compute marginal distributions (column totals of confusion matrix) for hybrid model predictions under:
  - Misspelling attacks at 5% and 20% noise.
  - Round-trip translation (NLLB-200).
- **Data Status:** Metrics and confusion matrices exist in [results/tables/T4.csv](../results/tables/T4.csv), [results/tables/T7.csv](../results/tables/T7.csv), and figures `F7`, `F8`, `F12`.
- **Execution Plan:** Sum predictions across predicted classes to illustrate which classes absorb false positives under perturbation.

### Item 6: Retrain Hybrid Model without Gemma (6-Class Ablation)
- **Objective:** Retrain the hybrid multi-class XGBoost model with `Gemma-2-9B` excluded from the training label set (reducing from 7 to 6 AI classes). Then evaluate on:
  - Gemma-27B paraphrased set
  - Mistral-7B paraphrased set
  - BBC cross-corpus set
- **Scientific Goal:** Test where Gemma-paraphrased samples are attributed when the detector cannot attribute them to Gemma (does it revert to original author, human, or another model?).

### Item 7: RoBERTa 5-Seed Reproducibility Protocol
- **Objective:** Train and evaluate `roberta-base` for binary detection and multi-class attribution across five fixed seeds (`42`, `123`, `999`, `2024`, `8080`).
- **Data Status:** Seed `42` is finished ([src/roberta/train_roberta_baseline.py](../src/roberta/train_roberta_baseline.py) & [src/roberta/evaluate_roberta_baseline.py](../src/roberta/evaluate_roberta_baseline.py)).
- **Execution Plan:** Automate the seed loop, save checkpoints, and report mean $\pm$ standard deviation matching the XGBoost protocol in [results/tables/T9_Statistical_Significance_Detailed.csv](../results/tables/T9_Statistical_Significance_Detailed.csv).

---

## 4. Priority 3 (Analytical Metrics & Latency)

### Item 8: Centroid Distances in Hybrid Feature Space
- **Objective:** Calculate the Euclidean and Cosine distances in 791-dimensional space between the centroid of each paraphrased dataset and the centroids of all 8 training classes.
- **Scientific Goal:** Prove geometric convergence of paraphrased texts toward the paraphraser's cluster.

### Item 9: End-to-End Latency Benchmark
- **Objective:** Measure per-document inference latency for the hybrid model (11-step feature extraction + XGBoost inference) versus RoBERTa inference (~15–30 ms/sample).

### Item 10: Attribution Confidence Analysis
- **Objective:** Measure mean predicted probability (softmax confidence) for correct attributions versus incorrect attributions across paraphrased sets.
- **Scientific Goal:** Assess calibration and model uncertainty under adversarial attacks.

---

## 5. Answers to Clarifying Questions

### Question A: "Is the cross-domain WP set Writing Prompts or Wikipedia?"
- **Answer:** **Writing Prompts**. Specifically Reddit `r/WritingPrompts` from the MGT Benchmark dataset (referenced in [notebooks/Main.ipynb:L502](../notebooks/Main.ipynb#L502) and [agents/Agent Handoff Document.md:L59](../agents/Agent%20Handoff%20Document.md#L59)).

### Question B: "Why do the Essay and WP counts show 396 and 398 rather than 400?"
- **Answer:** In [notebooks/data_engineer.ipynb:L697](../notebooks/data_engineer.ipynb#L697), preprocessing step 10 executes `train_df.dropna(inplace=True)`. Four samples in Essay and two samples in WP contained empty strings, NaNs, or failed tokenization during extraction and were dropped.
