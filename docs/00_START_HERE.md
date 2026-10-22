# START HERE — what happened in this project, in order

Read these in number order. Everything else is code or results.

| # | File | What it is |
|---|------|------------|
| 01 | `01_project_overview.md` | Binary vs multi-class task, hybrid Features+BERT, seed 999. 2-min context. |
| 02 | `02_professor_request.md` | Original supervisor request: 10 items (P1/P2/P3) + 2 clarifications. **The spec — read first.** |
| 03 | `03_action_plan.md` | Technical breakdown + execution plan per item (data status, label mapping, scripts to run). |
| 04 | `04_professor_response.md` | **Final answers sent to professor.** All 10 tables + interpretations. |
| 05 | `05_fixes_log.md` | Two post-hoc fixes: Gemma-27B perturbation row (TF-IDF rematch) + RoBERTa-Multi misspell rows (4-seed, seed-42 model deleted). Read before citing T19/T25. |
| 06 | `06_cnn_provenance.md` | Item 2 detail: CNN/DailyMail (not BBC) source, prompts, endpoints, quality flags. |
| 07 | `07_task3_roberta_report.md` | RoBERTa baseline full report (seed 42). Copy of `results/extra/task3/Task3_Results_Report.md` (that copy stays with its figures). |
| 09 | `09_third_phase_experiments.md` | Phase-3 supervisor spec + per-task resolutions (English/Arabic record). |
| 11 | `11_phase3_final_supervisor_report.md` | Final Phase-3 supervisor report: corrected numbers, caveats, artifacts. |
| 12 | `12_phase3_complete_arabic_retrospective.md` | **Single Arabic retrospective of Phase 2 (10 tasks) + Phase 3 (6 tasks).** |

## Status at a glance (all 10 done)

| Item | Result (one line) | Table |
|------|-------------------|-------|
| 1. Alt-GT fingerprint-overwrite | Confirmed for Gemma only (81.5% / 77.0%); others collapse to Gemma-2-9B | T16 |
| 2. Second-corpus provenance | CNN/DailyMail, 7 endpoints, temp 0.7 | T14 / T18 |
| 3. RoBERTa Gemma CM | Acc 16.8%, 75.5% → Gemma-2-9B | T17 |
| 4. Perturbation magnitude | Gemma most aggressive (BLEU ~10.8); Qwen most conservative (51.5) | T19 |
| 5. Pred-class distribution | Misspell-20% → Human 90.9%; translation balanced | T20 |
| 6. No-Gemma retrain | Mass splits Human 579 / GPT-4o 529, not back to author | T21 |
| 7. RoBERTa 5-seed | Binary stable 0.997±0.001; Multi Gemma-collapse 0.185±0.015 | T25 |
| 8. Centroid distances | Paraphrases converge to Gemma/Yi/Mistral region | T22 |
| 9. Latency | Hybrid 176 ms/doc vs RoBERTa 18 ms/doc (~10×) | T24 |
| 10. Confidence | Overconfident on Gemma (~0.90 correct ≈ wrong) | T23 |
| Q: WP? Essay/WP counts? | WritingPrompts; 4+2 rows dropped by `dropna()` | — |

Details + numbers: `04_professor_response.md`. Caveats: `05_fixes_log.md`.

## Where do code and results live?

- Phase-2 scripts: `src/phase2/` (`task1_alt_gt.py`→T16, `task3_roberta_cm.py`→T17, `task4*.py`→T19, `task5_preddist.py`→T20, `task6_nogemma_eval.py`→T21, `run_seed.py`+`watch_seeds.py`→T25, `task8_centroids.py`→T22, `task9_latency.py`→T24, `task10_confidence.py`→T23). Run from repo root; outputs go to `results/phase2/`.
- Phase-2 finals (single copy): `results/phase2/` (`T16`–`T25`, `F17`, `F21_*`, NoGemma joblibs).
- Phase-1 pipeline: `notebooks/Main.ipynb`, `notebooks/data_engineer.ipynb`, `src/roberta/train_roberta_baseline.py`, `src/roberta/evaluate_roberta_baseline.py`; outputs in `results/tables/`, `results/figures/` (T1–T13, F1–F14; T14/F15 written by `helper/cross_corpus_feature_eval.py`).
- Second-corpus generation: `tools/second_corpus/` (`prepare_cnn_dailymail.py`, `generate_from_apis.py`); raw output `results/extra/task2/`.
- RoBERTa work package (report + F1–F7 + T15): `results/extra/task3/`.
- `archive/` (local only, NOT in git): `legacy/` (old Archive/), `phase2_work/` (logs, meeting brief), `submission/` (`Prof_Response.docx/.pdf`), `detectgpt/` (one-off DetectGPT scripts), `misc/` (old zips/docx).

## Cleanup history (why paths in old messages may differ)

- 2026-10-15 (1): loose root `.md` files consolidated into `docs/` with numbered names.
- 2026-10-15 (2): `newtasks2/` split into `src/phase2/` (scripts) + `results/phase2/` (outputs); `extra tasks/task2/` → `tools/second_corpus/`; `Second_Phase_Results_Basel/` deleted (every file verified byte-identical to the kept copy); 14 duplicate figures, duplicate T16 tables and duplicate `Prof_Response` copies removed; finished work moved to `archive/` and untracked from git (`archive/`, `*.joblib`, `*.log` ignored).
