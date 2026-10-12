# Task 2 — Second News Corpus (CNN/DailyMail) Provenance & Evaluation Report

> Naming note: this set was previously labeled "BBC" across the repo. All code, data,
> tables, and figures have been renamed to CNN/DailyMail (`processed_second_corpus_cnn.csv`,
> `Second_Corpus_CNN` in T14). Only the frozen `Archive/.../Item2_Cross_Corpus_BBC_News/`
> package keeps the old name.

**Artifact:** `results/extra/task2/generated_full1000.csv` (+ pilot `generated_sample50.csv`)  
**Downstream Dataset:** `data/processed/processed_second_corpus_cnn.csv` via `helper/cross_corpus_feature_eval.py`  
**Status:** Provenance fully recovered, verified from repo generation scripts (`extra tasks/task2/`), and zero-shot results verified (§5–§6).

---

## 1. File identity (verified from artifact)

| Property | Value |
|---|---|
| Path | `results/extra/task2/generated_full1000.csv` |
| Shape | 1000 rows × 9 columns, zero nulls |
| Columns | `Prompt, Human_story, claude_haiku, gemma_2_9b, qwen_2_72b, llama_3_8b, mistral_7b, gpt-4o-mini, deepseek_pro` |
| Created | 2026-08-06 (pilot `generated_sample50.csv`: 2026-08-05, 7 cols — no `gpt-4o-mini`, no `deepseek_pro`) |
| `Prompt` | 999/1000 unique, mean 313 chars — short news summaries/headlines |
| `Human_story` | 1000/1000 unique, mean 3924 chars (629–10735) — full articles, each starting with `(CNN)` dateline |

## 2. Human source — Verified CNN/DailyMail

The origin of the human articles is confirmed by `extra tasks/task2/prepare_cnn_dailymail.py`:
- Dataset: Hugging Face `abisee/cnn_dailymail` (version `3.0.0`, `test` split).
- 1,000 articles sampled with `seed=42`.
- `highlights` mapped to `Prompt`, and `article` mapped to `Human_story`.
- Every `Human_story` opens with a `(CNN)` dateline.
- The old "BBC" label (from early pipeline drafts) has been removed everywhere; the few frozen Archive copies are noted in §6.
- **Recommendation:** Explicitly document in the paper that the secondary news benchmark is built upon CNN/DailyMail highlights and articles.

## 3. AI columns → training-class mapping (from `helper/cross_corpus_feature_eval.py:43-52`)

| CSV column (= model used, per column name) | Mapped `Writer` | In Multi_BERT label set? |
|---|---|---|
| `gpt-4o-mini` | `GPT-4o` | yes |
| `gemma_2_9b` | `Gemma-2-9B` | yes |
| `mistral_7b` | `Mistral-7B` | yes |
| `llama_3_8b` | `Llama-8B` | yes |
| `qwen_2_72b` | `Qwen-2-72B` | yes |
| `deepseek_pro` | `Yi-Large` (zero-shot compat alias) | yes |
| `claude_haiku` | `Claude` | NO — dropped at multi-class inference |
| `Human_story` | `Human` | yes |

Processed corpus: 7999 rows (8×1000 minus 1 `Yi-Large` row lost to `dropna`).
T14 zero-shot results: Binary Acc 0.8466 / F1 0.7014; Multi Acc 0.2656 / F1 0.1992.

Mean output lengths (chars): claude_haiku 1568, gemma_2_9b 1361, qwen_2_72b 2268,
llama_3_8b 1702, mistral_7b 1606, gpt-4o-mini 1822, deepseek_pro 2425.

## 4. Quality flags found in the artifact (should be disclosed)

- `llama_3_8b` row 474 = `"I can't fulfill this request."` (refusal, 29 chars); 53/1000 llama texts < 200 chars.
- `deepseek_pro` row 915 = `"READING,"` (truncated, 8 chars); 3/1000 < 200 chars.
- `qwen_2_72b` row 682 truncated mid-sentence (94 chars); 2/1000 < 200 chars.
- `claude_haiku`, `gemma_2_9b`, `mistral_7b`, `gpt-4o-mini`: no texts < 200 chars.
- 1 duplicated `Prompt` (999 unique / 1000 rows).

## 5. Recovered Generation Scripts & Parameters (Found in repo history)

The generation and dataset preparation scripts are verified from `extra tasks/task2/`:

- **Human Dataset Preparation:** `extra tasks/task2/prepare_cnn_dailymail.py`
  - **Source Dataset:** Hugging Face `abisee/cnn_dailymail` (version `3.0.0`, `test` split).
  - **Sampling:** 1,000 random samples, `seed=42`.
  - **Mapping:** `highlights` -> `Prompt`, `article` -> `Human_story`.
  - **Intermediate Output:** `results/extra/task2/cnn_dailymail_1k.csv`.

- **Generation Script:** `extra tasks/task2/generate_from_apis.py`
  - **Prompt Template:**
    ```text
    Write a complete, professional news article based on the following summary/abstract. Do not add any commentary or meta-text — only output the article itself.

    Summary:
    {prompt}
    ```
  - **Sampling / Decoding Parameters:**
    - `temperature`: `0.7`
    - `max_tokens`: `1024`
    - Async retry: Tenacity with exponential backoff (`multiplier=1, min=4, max=60, stop=4`).
  - **Exact Model Endpoints & IDs:**
    - `gemma_2_9b`: OpenRouter `google/gemma-3-12b-it` (closest available to Gemma-2-9B)
    - `qwen_2_72b`: OpenRouter `qwen/qwen-2.5-72b-instruct`
    - `llama_3_8b`: OpenRouter `meta-llama/llama-3.1-8b-instruct`
    - `mistral_7b`: OpenRouter `mistralai/mistral-nemo`
    - `deepseek_pro`: OpenRouter `deepseek/deepseek-v4-pro` (substitute for retired Yi-Large; mapped to `Yi-Large` downstream)
    - `claude_haiku`: OpenRouter `anthropic/claude-3-haiku:beta` (dropped in multi-class)
    - `gpt-4o-mini`: OpenAI API `gpt-4o-mini`
  - **Output File:** `results/extra/task2/generated_full1000.csv` (1000 rows × 9 cols).

## 6. Downstream Feature Pipeline & Zero-Shot Results

- **Feature Pipeline:** `helper/cross_corpus_feature_eval.py`
  - Generated long-format dataset: `data/processed/processed_second_corpus_cnn.csv` (7,999 rows).
  - 11-step feature extraction (Perplexity, Burstiness, UID, Syntax Depth, Empath, Semantic Consistency, BERT 768-dim).
- **Validation Results Table (T14):**
  - Path: `results/tables/T14_Second_Corpus_CNN_Validation.csv` (frozen copy under the old name in `Archive/evaluation_results_package/Item2_Cross_Corpus_BBC_News/`)
  - **Binary Detection:**
    - Accuracy: **84.66%** (`0.8466`)
    - Macro Precision: **67.94%** (`0.6794`)
    - Macro Recall: **73.96%** (`0.7396`)
    - Macro F1: **70.14%** (`0.7014`)
  - **Multi-Class Attribution:**
    - Accuracy: **26.56%** (`0.2656`)
    - Macro Precision: **19.95%** (`0.1995`)
    - Macro Recall: **26.56%** (`0.2656`)
    - Macro F1: **19.92%** (`0.1992`)
- **Confusion Matrix Figure (F15):**
  - Path: `results/figures/F15_Second_Corpus_CNN_Confusion_Matrix.png` (frozen copy under the old name in `Archive/evaluation_results_package/Item2_Cross_Corpus_BBC_News/`).

