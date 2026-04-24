### **Phase 1 Deliverables Checklist**

**I. Formatting & Organization Rules**
* [ ] Maintain a fixed random seed for every experiment and document it.
* [ ] Use an exact 80/20 train/test split across all experiments.
* [ ] Save all result tables as CSV files in the `results/tables/` folder (named T1.csv through T8.csv).
* [ ] Save all figures as high-resolution PNG files (minimum 300 DPI) in the `results/figures/` folder (named F1.png through F13.png).
* [ ] Save all model files (.pkl or .joblib) with metadata (seed, split, feature list) in the `results/models/` folder.
* [ ] Save training logs showing hyperparameters used in the `results/logs/` folder.

---

**II. The 8 Required Tables**
* [ ] **Table T1:** Binary detection results (features vs. features+BERT).
* [ ] **Table T2:** Multi-class attribution results (features vs. features+BERT).
* [ ] **Table T3:** Per-class F1 Scores for multi-class attribution.
* [ ] **Table T4:** Misspelling attack degradation at 5%, 10%, 15%, and 20% perturbation rates.
* [ ] **Table T5:** Length sensitivity analysis (short/medium/long bins).
* [ ] **Table T6:** Cross-domain generalization results (including Essay and WP benchmarks).
* [ ] **Table T7:** Adversarial attack summary (all attacks, binary).
* [ ] **Table T8:** Feature ablation study results (impact of removing each feature group).

---

**III. The 13 Required Figures**
* [ ] **Figure F1:** Confusion matrix: binary detection (features+BERT).
* [ ] **Figure F2:** Confusion matrix: multi-class attribution (features+BERT).
* [ ] **Figure F3:** Feature importance bar chart: binary, features only.
* [ ] **Figure F4:** Feature importance bar chart: binary, features+BERT.
* [ ] **Figure F5:** Feature importance bar chart: multi-class, features only.
* [ ] **Figure F6:** Feature importance bar chart: multi-class, features+BERT.
* [ ] **Figure F7:** Degradation curve: misspelling vs. accuracy (binary task, both configs).
* [ ] **Figure F8:** Degradation curve: misspelling vs. accuracy (multi-class attribution).
* [ ] **Figure F9:** Bar chart: length sensitivity (both tasks, both configs across length bins).
* [ ] **Figure F10:** Confusion matrices: Essay and WP benchmarks (features+BERT).
* [ ] **Figure F11:** Confusion matrix: Reuters/Claude unseen model.
* [ ] **Figure F12:** Confusion matrix: paraphrasing attack (multi-class, features+BERT).
* [ ] **Figure F13:** Bar chart: feature ablation impact (showing F1 drop when each feature group is removed).
