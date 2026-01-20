original dataset : <https:/huggingface.co/datasets/gsingh1-py/train?library=datasets>

MTG_Benchmark dataset : <https://github.com/xinleihe/MGTBench?tab=readme-ov-file>

misspelled dataset : <https://www.kaggle.com/datasets/fazilbtopal/misspelled-words?select=misspelled.csv>

In this project we used `uv` as package manager and `python 3.11` as the python version.

Files map :

- `baseline_consolidated.ipynb` : contains the Training and evaluation code for the baseline models for both binary and classification tasks.
- `building_with_BERT.ipynb` : contains the Training and evaluation code for the models with BERT embeddings for both binary and classification tasks.
- `data_processing.ipynb` : contains the code for data processing for both binary and classification tasks.
- `data_engineer.ipynb` : contains the code for calculating the base features like `perplexity`, `burstiness`, `Type-Token Ratio` etc.
- `data_engineer_with_BERT.ipynb` : contains the code for adding BERT embeddings to the base features.
- `data` folder : contains all the datasets used in the experiments.
- `Evaluation` folder : contains the code for evaluating models under different scenarios like misspelling attack, cross-domain, unseen models, paraphrasing and translation.
- `Evaluation/evaluation_utils.ipynb` : contains the utility functions for evaluation Accuracy, Precision, Recall and F1-Score for both binary and classification tasks under different scenarios like misspelling attack, cross-domain, unseen models, paraphrasing and translation.
- `models` folder : contains the saved models for both binary and classification tasks.

**Key Info:**

- **For misspelling attack:** I replaced some letters with their surrounding letters in the QWERTY keyboard layout, swapped two adjacent characters, removed random characters, removed or added punctuation at the end of words, and replaced some common misspelling words with their correct versions. I performed one or more of these for 20% of the test text.
- **For Cross-Domain:** I chose the `MTG_Benchmark` dataset as a benchmark. This dataset has three sets. The first is `Essay`, which consists of students writing essays. The second is `WP`, which consists of stories written by Reddit users. For these two, I compared `GPT4All` and `Human results`. Since our model is trained on `GPT-4o` and `Human_story`, we can test the generalization of the model.
- **For the unseen model in `MTG_Benchmark`:** There is a third set which consists of articles from `Reuters` news and its `Claude` generated version i chose it since it same filed of out  original dataset but completely not seen and not seen Model .
- **For the Paraphrasing:** I used the `gemma-2-27b-it` model and asked it to paraphrase 500 samples from the original test dataset.
- **For Translation:** I used the `NLLB` model and had it translate 3,136 samples from the original test dataset from English to French and then back t English.

All numbers are collected from the `macro avg`.
**Binary Task:**

| **Model**                  | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
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

**Classification Task:**

| **Model**           | **Accuracy** | **Precision** | **Recall** | **F1-Score** |
| ------------------- | ------------- | -------------- | ----------- | ------------- |
| Base Model          | 0.72          | 0.72           | 0.72        | 0.72          |
| Misspelled          | 0.164         | 0.246          | 0.164       | 0.073         |
| Misspelled + BERT   | 0.285         | 0.579          | 0.285       | 0.211         |
| Paraphrasing        | 0.204         | 0.176          | 0.204       | 0.126         |
| Paraphrasing + BERT | 0.164         | 0.237          | 0.164       | 0.160         |
