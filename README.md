original dataset :https:/huggingface.co/datasets/gsingh1-py/train?library=datasets

MTG_Benchmark dataset : https://github.com/xinleihe/MGTBench?tab=readme-ov-file

missplled dataset: https://www.kaggle.com/datasets/fazilbtopal/misspelled-words?select=misspelled.csv


all numbers are collected from `macro avg`
BINARY TBINARY Taskask :

| **Model **                  | **Accuracy ** | **Precision** | **Recall ** | **F1-Score** |
| --------------------------- | ------------- | ------------- | ----------- | ------------ |
| Base model                  | 0.98          | 0.97          | 0.96        | 0.97         |
| Misspelled                  | 0.196         | 0.867         | 0.196       | 0.138        |
| Misspelled + BERT           | 0.390         | 0.884         | 0.390       | 0.430        |
| Cross-Domain (Essay)        | 0.705         | 0.804         | 0.705       | 0.679        |
| Cross-Domain (Essay) + BERT | 0.609         | 0.781         | 0.609       | 0.540        |
| Cross-Domain (WP)           | 0.927         | 0.928         | 0.927       | 0.927        |
| Cross-Domain (WP) + BERT    | 0.824         | 0.859         | 0.824       | 0.820        |
| Unseen Models               | 0.672         | 0.735         | 0.672       | 0.649        |
| Unseen Models + BERT        | 0.835         | 0.835         | 0.835       | 0.835        |
| Paraphrasing                | 0.802         | 0.887         | 0.802       | 0.832        |
| Paraphrasing + BERT         | 0.889         | 0.853         | 0.889       | 0.857        |
| Translation                 | 0.996         | 0.996         | 0.996       | 0.996        |
| Translation + BERT          | 1.000         | 1.000         | 1.000       | 1.000        |

Classifction Task :

| **Model**           | **Accuracy ** | **Precision ** | **Recall ** | **F1-Score ** |
| ------------------- | ------------- | -------------- | ----------- | ------------- |
| Base Model          | 0.72          | 0.72           | 0.72        | 0.72          |
| Misspelled          | 0.164         | 0.246          | 0.164       | 0.073         |
| Misspelled + BERT   | 0.285         | 0.579          | 0.285       | 0.211         |
| Paraphrasing        | 0.204         | 0.176          | 0.204       | 0.126         |
| Paraphrasing + BERT | 0.164         | 0.237          | 0.164       | 0.160         |