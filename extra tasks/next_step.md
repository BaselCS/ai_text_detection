# PHASE 2 IMPLEMENTATION ROADMAP: STEP-BY-STEP ACTION PLAN

**Project Title:** A Hybrid Stylometric and Contextual-Embedding Approach to AI-Generated News Detection and Model Attribution  
**Target:** Journal Extension (Q1 Journal Submission Requirements)  
**Package Manager Requirement:** `uv` (`uv run python ...`)  

---

## Task 1: Multiple Paraphrasers Attack (Fingerprint Overwrite Generalization)

### Objective
Determine if the **"Fingerprint-Overwrite Effect"** discovered with Gemma-27B is universal across other LLM families (e.g., OpenAI GPT family and Meta/Qwen LLaMA family).

### Step-by-Step Execution Plan

1. **Step 1.1: Setup Paraphrasing Script for New Models**
   - Create script `helper/paraphrase_multi_models.py` using `uv`.
   - Implement API / Generation calls for:
     - **Family A (OpenAI):** `gpt-4o-mini` (via OpenAI API).
     - **Family B (Meta / Qwen):** `llama-3-8b-instruct` or `qwen-2.5-7b-instruct` (via vLLM / HuggingFace local inference or API).
   - Use identical persona prompt template as Gemma-27B (`helper/paraphrasing_attacks.ipynb`).

2. **Step 1.2: Generate Paraphrased Attack Datasets**
   - Paraphrase 500+ original articles from `data/processed/processed_articles.csv` for each paraphraser model.
   - Save outputs to:
     - `data/processed/raw_paraphrased_gpt4o_mini.csv`
     - `data/processed/raw_paraphrased_llama3.csv`

3. **Step 1.3: Run Feature Extraction Pipeline**
   - Run 11-step feature extraction pipeline via `uv run python` script or `data_engineer.ipynb`.
   - Extract 19 stylometric/statistical/semantic features + 768 BERT embeddings for both new datasets.
   - Save feature-engineered datasets:
     - `data/processed/processed_paraphrased_gpt4o_mini.csv`
     - `data/processed/processed_paraphrased_llama3.csv`

4. **Step 1.4: Run Inference & Evaluation**
   - Evaluate pre-trained Phase 1 models (`Bin_BERT`, `Multi_BERT`) on the new paraphrased datasets.
   - Measure:
     - Binary Detection degradation.
     - Multi-Class Attribution misclassification distribution (check if GPT-4o-mini or LLaMA-3 paraphrasing forces predictions into GPT-4o or LLaMA classes).

5. **Step 1.5: Generate Comparative Findings & Plots**
   - Add Table `T7.1` to `results/tables/T7.1_Expanded_Paraphrasers.csv`.
   - Generate confusion matrices `results/figures/F12.1_Paraphrased_GPT.png` and `results/figures/F12.1_Paraphrased_LLaMA.png`.

---

## Task 2: Second Dataset Validation (Cross-Corpus Generalization)

### Objective
Validate pre-trained Phase 1 XGBoost models on a completely independent human news corpus + newly generated AI articles across all 7 LLM classes to prove out-of-corpus robustness.

### Step-by-Step Execution Plan

1. **Step 2.1: Acquire Second Human News Dataset**
   - Download or extract a clean English Human News Corpus (e.g., BBC News Corpus or Reuters Corpus).
   - Filter and clean 1,000+ human news articles.

2. **Step 2.2: Generate Parallel AI Text Corpus (7 LLM Classes)**
   - Generate corresponding AI news articles matching the BBC/Reuters news prompts across all 7 target AI LLMs:
     - `GPT-4o`
     - `Gemma-2-9B`
     - `Qwen-2-72B`
     - `LLaMA-3-8B`
     - `Mistral-7B`
     - `Yi-Large` replaced with `deepseek_pro` 
     - `GPT4All` / `Claude`
   - Combine human + 7 AI classes into `data/raw_second_corpus.csv`.

3. **Step 2.3: Run Feature Engineering**
   - Execute feature engineering pipeline via `uv run python`:
     - Extract Stylometrics, Perplexity, Burstiness, UID, spaCy Syntax Depth, Empath categories, MiniLM Semantic Consistency, and BERT 768-dim embeddings.
   - Save processed dataset to `data/processed/processed_second_corpus_bbc.csv`.

4. **Step 2.4: Zero-Shot Cross-Corpus Testing**
   - Load pre-trained models from Phase 1 (`results/models/Bin_BERT_model.joblib`, `Multi_BERT_model.joblib`).
   - Run inference directly on `processed_second_corpus_bbc.csv` **without any re-training or fine-tuning**.

5. **Step 2.5: Document Cross-Corpus Results**
   - Output summary table `results/tables/T14_Second_Corpus_Validation.csv`.
   - Generate confusion matrix `results/figures/F15_Second_Corpus_Confusion_Matrix.png`.

---

## Task3: Fine-Tuned Neural Baseline (RoBERTa)

### Objective
Build a direct deep learning baseline by fine-tuning `roberta-base` end-to-end on text splits, comparing its performance against XGBoost + Extracted Hybrid Features.

### Step-by-Step Execution Plan

1. **Step 3.1: Create PyTorch Training Script**
   - Create `train_roberta_baseline.py` using `uv`.
   - Use `transformers.Trainer` or PyTorch training loop with AdamW optimizer, linear warmup schedule, and mixed precision (`fp16`).

2. **Step 3.2: Fine-Tune Binary Detection Model**
   - Model: `roberta-base` (SequenceClassification, 2 classes: Human vs. AI).
   - Training parameters: Batch size 16, Learning rate 2e-5, Epochs 3-5, Seed 999.
   - Save trained weights to `results/models/RoBERTa_Binary/`.

3. **Step 3.3: Fine-Tune Multi-Class Attribution Model**
   - Model: `roberta-base` (SequenceClassification, 8 classes: Human + 7 LLMs).
   - Training parameters: Batch size 16, Learning rate 2e-5, Epochs 5, Seed 999.
   - Save trained weights to `results/models/RoBERTa_Multi/`.

4. **Step 3.4: Comprehensive Stress Testing**
   - Evaluate fine-tuned RoBERTa models across all Phase 1 evaluation suites:
     - Clean Test Split.
     - Misspelling Attacks (5%, 10%, 15%, 20%).
     - Cross-Domain Datasets (Essays, WebText, Reuters).
     - Paraphrasing & Translation Attacks.

5. **Step 3.5: Comparative Analysis & Final Documentation**
   - Create summary table `results/tables/T15_Neural_vs_Hybrid_XGBoost.csv` comparing Accuracy, F1-Score, Training Time, and Robustness under attacks.
   - Summarize key findings in journal manuscript draft.

---

## Summary Checklist

| Task | Core Command / Action | Key Output File | Status |
| :--- | :--- | :--- | :--- |
| **Task 1** | `uv run python helper/paraphrase_multi_models.py` | `processed_paraphrased_gpt4o_mini.csv`, `T7_Expanded.csv` | `[ ] Pending` |
| **Task 2** | `uv run python scripts/process_second_corpus.py` | `processed_second_corpus_bbc.csv`, `T14_Second_Corpus.csv` | `[ ] Pending` |
| **Task3** | `uv run python train_roberta_baseline.py` | `RoBERTa_Binary/`, `T15_Neural_vs_Hybrid.csv` | `[ ] Pending` |
