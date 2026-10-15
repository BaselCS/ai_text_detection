# AI Text Detection Project

Binary human-vs-AI detection + multi-class LLM attribution (7 AI generators + Human), using XGBoost on hybrid Features+BERT (791-d) with a fine-tuned RoBERTa baseline.

## Docs — start here

All project notes live in `docs/` in reading order — see `docs/00_START_HERE.md` (navigation, 10-item status table, layout map):

- `01_project_overview.md` — task, method, seeds, entry points
- `02_professor_request.md` — supervisor spec (10 items + 2 clarifications)
- `03_action_plan.md` — execution plan per item
- `04_professor_response.md` — final answers sent to professor
- `05_fixes_log.md` — post-hoc fixes (read before citing T19/T25)
- `06_cnn_provenance.md` — second-corpus (CNN/DailyMail) provenance
- `07_task3_roberta_report.md` — RoBERTa baseline report
- `08_explanation_ar.md` — Arabic walkthrough

## Project Structure

- `src/phase2/`: Phase-2 scripts (items 1–10). Run from repo root; outputs go to `results/phase2/`.
- `tools/second_corpus/`: CNN/DailyMail acquisition + AI-version generation scripts.
- `helper/`: attack/feature-eval helpers (paraphrasing, misspelling, cross-corpus).
- `data/`: `processed/` (feature CSVs) and `need_processing/` (raw inputs).
- `results/`: `tables/` + `figures/` (Phase-1 T1–T14, F1–F15), `phase2/` (Phase-2 finals T16–T25), `extra/` (task2/task3 work packages), `logs/`, `models/`.
- `archive/` (local only, not in git): finished/frozen work — `legacy/`, `phase2_work/` (logs), `submission/` (professor docx/pdf), `detectgpt/`, `misc/`.
- `notebooks/Main.ipynb`: core XGBoost training/eval. `notebooks/data_engineer.ipynb`: 11-step feature pipeline.
- `src/roberta/train_roberta_baseline.py` / `src/roberta/evaluate_roberta_baseline.py`: RoBERTa baseline. `src/case_study.py`: paraphrase case study.

## How to Use

```bash
git clone https://github.com/BaselCS/ai_text_detection

# To set up the environment
uv init
```

## Reproducing Results

1. Open `notebooks/Main.ipynb` and run cells sequentially (trains XGBoost Binary/Multi, writes to `results/`).
2. For RoBERTa: `uv run python src/roberta/train_roberta_baseline.py --epochs 3 --batch_size 8 --max_length 512`, then `uv run python src/roberta/evaluate_roberta_baseline.py`.
3. For Phase-2 items: `uv run python src/phase2/task1_alt_gt.py` etc. (see `docs/00_START_HERE.md` for the script→table map).

## Notes on Methodology

- **Reproducibility:** fixed global seed (`SEED = 999`); 5-seed protocol `[42, 123, 999, 2024, 8080]` for variance.
- **Adversarial Resilience:** translation, paraphrasing (6 paraphrasers), and misspelling (5–20%) attacks; see `docs/04_professor_response.md`.
