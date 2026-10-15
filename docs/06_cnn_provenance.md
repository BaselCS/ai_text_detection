# T18 — Second-Corpus (CNN/DailyMail) Provenance

Corresponds to Item 2 of the Second Phase of Experiments:
"Tell me where the six AI versions of the BBC articles came from.
If you generated them, send the models, prompts, and settings used."

## 1. Correction: not BBC — CNN/DailyMail

The set previously labelled "BBC" is in fact **CNN/DailyMail**.
The old label has been renamed everywhere to `Second_Corpus_CNN`:

- `data/processed/processed_second_corpus_cnn.csv`
- `results/tables/T14_Second_Corpus_CNN_Validation.csv`
- `results/figures/F15_Second_Corpus_CNN_Confusion_Matrix.png`

## 2. Human source (verified)

Script: `tools/second_corpus/prepare_cnn_dailymail.py`

- HuggingFace dataset: `abisee/cnn_dailymail`, config `3.0.0`, split `test`
- Sampling: 1000 random samples, `seed=42`
  (`dataset.shuffle(seed=42).select(range(1000))`)
- Column mapping:
  - `highlights` → `Prompt` (mean ~313 chars)
  - `article` → `Human_story` (mean ~3924 chars, all starting with `(CNN)` dateline)
- Text cleaning: `\n` → space + strip
- Output: `results/extra/task2/cnn_dailymail_1k.csv` (1000 rows)

## 3. Generation (verified)

Script: `tools/second_corpus/generate_from_apis.py`

Prompt template (identical for all models):

```text
Write a complete, professional news article based on the following summary/abstract. Do not add any commentary or meta-text — only output the article itself.

Summary:
{prompt}
```

Parameters:

- `temperature=0.7`, `max_tokens=1024`
- Async requests with tenacity exponential backoff
  (`wait_exponential(multiplier=1, min=4, max=60)`, `stop_after_attempt(4)`)

Exact endpoints:

| Column | Provider / endpoint | Model id |
|---|---|---|
| `gemma_2_9b` | OpenRouter | `google/gemma-3-12b-it` (proxy for Gemma-2-9B) |
| `qwen_2_72b` | OpenRouter | `qwen/qwen-2.5-72b-instruct` |
| `llama_3_8b` | OpenRouter | `meta-llama/llama-3.1-8b-instruct` |
| `mistral_7b` | OpenRouter | `mistralai/mistral-nemo` |
| `deepseek_pro` | OpenRouter | `deepseek/deepseek-v4-pro` (substitute for retired Yi-Large → mapped to `Yi-Large`) |
| `claude_haiku` | OpenRouter | `anthropic/claude-3-haiku:beta` (dropped at multi-class) |
| `gpt-4o-mini` | OpenAI | `gpt-4o-mini` |

## 4. Artifacts

- `results/extra/task2/generated_full1000.csv` — 1000 rows × 9 cols
  (`Prompt, Human_story, claude_haiku, gemma_2_9b, qwen_2_72b, llama_3_8b, mistral_7b, gpt-4o-mini, deepseek_pro`), zero nulls
- Processed via `helper/cross_corpus_feature_eval.py` →
  `data/processed/processed_second_corpus_cnn.csv` (7999 rows)

## 5. Zero-shot hybrid results (T14)

`results/tables/T14_Second_Corpus_CNN_Validation.csv`:

- Binary: Acc 0.8466 / MacroF1 0.7014
- Multi: Acc 0.2656 / MacroF1 0.1992

Figure: `results/figures/F15_Second_Corpus_CNN_Confusion_Matrix.png`

## 6. Quality flags to disclose

- `llama_3_8b` row 474 refusal; 53/1000 <200 chars
- `deepseek_pro` row 915 truncated (`READING,`)
- `qwen_2_72b` row 682 truncated
- 1 duplicated Prompt

## 7. Related files

- This file: `docs/06_cnn_provenance.md`
- `results/tables/T14_Second_Corpus_CNN_Validation.csv`
- `results/figures/F15_Second_Corpus_CNN_Confusion_Matrix.png`
- `tools/second_corpus/prepare_cnn_dailymail.py`
- `tools/second_corpus/generate_from_apis.py`
