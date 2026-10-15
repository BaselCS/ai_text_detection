# 01 — Project overview (2-minute context)

## Tasks
- **Binary:** Human vs AI (`is_AI` 0/1).
- **Multi-class attribution:** Human + 7 AI generators (GPT-4o, Gemma-2-9B, Qwen-2-72B, Llama-8B, Mistral-7B, Yi-Large, GPT4All/Claude family). Human baseline: NYT news corpus.

## Method
- **Features Only:** 19 stylometric/statistical/syntactic/psycholinguistic features (perplexity via GPT-2, burstiness, TTR, UID, syntax depth via spaCy, Empath categories, semantic consistency).
- **Features + BERT (hybrid):** 19 features + 768-d `bert-base-uncased` CLS embeddings (sliding window 512/overlap 50) = 791-d. XGBoost (`hist`, early stopping).
- **Baseline:** fine-tuned `roberta-base` end-to-end (see `docs/07_task3_roberta_report.md`).

## Reproducibility
Seed `999` for main split/training; 5-seed protocol `[42, 123, 999, 2024, 8080]` for variance (see T25 in `docs/04_professor_response.md`).

## Key Phase-1 finding
**Fingerprint-Overwrite Effect:** Gemma-27B paraphrasing drops attribution F1 ~85% → ~48%; detector re-attributes to the paraphraser. Phase 2 (see `docs/02_professor_request.md`) tests this across 6 paraphrasers.

## Entry points
- Training/eval: `notebooks/Main.ipynb` (XGBoost), `src/roberta/train_roberta_baseline.py` + `src/roberta/evaluate_roberta_baseline.py` (RoBERTa).
- Feature pipeline: `notebooks/data_engineer.ipynb` (11 steps).
- Attacks: `helper/paraphrasing_attacks.ipynb`, `helper/spelling_errors.ipynb`.
- Long architecture notes (historical, untracked): `agents/PROJECT_BREAKDOWN.md`, `agents/Agent Handoff Document.md`.
