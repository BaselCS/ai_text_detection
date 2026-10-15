"""Task 9: hybrid end-to-end latency per document (11-step features + XGB)
vs RoBERTa (~17.9 ms/sample measured in Task 3). Steady-state GPU inference.
"""
import json, time, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
import torch

import helper.paraphrase_feature_eval as pfe

DEVICE = pfe.DEVICE

df = pd.read_csv("data/processed/processed_paraphrased_articles.csv", usecols=["Text"], nrows=10)
texts = df["Text"].tolist()[:3]

with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
model = joblib.load("results/models/Multi_BERT_model.joblib")
scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")

# 1. Load models once outside timing
from transformers import GPT2LMHeadModel, GPT2Tokenizer
gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
if DEVICE == "cuda":
    gpt_mod = gpt_mod.half()
gpt_mod.eval()

from empath import Empath
empath_obj = Empath()

import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

from sentence_transformers import SentenceTransformer
sm = SentenceTransformer("all-MiniLM-L6-v2").to(DEVICE)

from transformers import AutoTokenizer, AutoModel
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_mod = AutoModel.from_pretrained("bert-base-uncased").to(DEVICE).half()
bert_mod.eval()

# 2. Warmup all models to eliminate first-call JIT / allocation latency
if DEVICE == "cuda":
    torch.cuda.synchronize()

_ = pfe.process_perplexity([texts[0]], gpt_mod, gpt_tok, DEVICE)
_ = pfe.process_burstiness([texts[0]])
_ = pfe.process_ttr([texts[0]])
_ = pfe.process_stylometric([texts[0]])
_ = pfe.process_uid([texts[0]], gpt_mod, gpt_tok, DEVICE)
_ = pfe.process_empath([texts[0]], empath_obj)
_ = pfe.process_syntax_depth([texts[0]], nlp)
_ = sm.encode([texts[0]], device=DEVICE)

dummy_bert_input = bert_tok([texts[0]], max_length=512, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    _ = bert_mod(**dummy_bert_input)

dummy_X = np.zeros((1, len(features)), dtype=np.float32)
dummy_Xs = scaler.transform(dummy_X)
_ = model.predict(dummy_Xs)

if DEVICE == "cuda":
    torch.cuda.synchronize()

# 3. Steady-State Timing per document
stages = {}

# (a) Perplexity (GPT-2)
if DEVICE == "cuda": torch.cuda.synchronize()
t = time.perf_counter()
ppl = pfe.process_perplexity(texts, gpt_mod, gpt_tok, DEVICE)
if DEVICE == "cuda": torch.cuda.synchronize()
stages["perplexity"] = (time.perf_counter() - t)

# (b) Burstiness
t = time.perf_counter()
b = pfe.process_burstiness(texts)
stages["burstiness"] = (time.perf_counter() - t)

# (c) TTR
t = time.perf_counter()
tr = pfe.process_ttr(texts)
stages["ttr"] = (time.perf_counter() - t)

# (d) Stylometric
t = time.perf_counter()
cx, awl, asl = pfe.process_stylometric(texts)
stages["stylometric"] = (time.perf_counter() - t)

# (e) UID
if DEVICE == "cuda": torch.cuda.synchronize()
t = time.perf_counter()
u = pfe.process_uid(texts, gpt_mod, gpt_tok, DEVICE)
if DEVICE == "cuda": torch.cuda.synchronize()
stages["uid"] = (time.perf_counter() - t)

# (f) Empath
t = time.perf_counter()
em = pfe.process_empath(texts, empath_obj)
stages["empath"] = (time.perf_counter() - t)

# (g) Syntax Depth (Spacy)
t = time.perf_counter()
sd = pfe.process_syntax_depth(texts, nlp)
stages["syntax"] = (time.perf_counter() - t)

# (h) Semantic similarity (SentenceTransformer)
if DEVICE == "cuda": torch.cuda.synchronize()
t = time.perf_counter()
mn, st_ = pfe.process_semantic(texts, sm, DEVICE)
if DEVICE == "cuda": torch.cuda.synchronize()
stages["semantic"] = (time.perf_counter() - t)

# (i) BERT Embeddings (Steady state GPU)
if DEVICE == "cuda": torch.cuda.synchronize()
t = time.perf_counter()
b_enc = bert_tok(texts, max_length=512, truncation=True, padding=True, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    b_out = bert_mod(**b_enc)
    emb = b_out.last_hidden_state[:, 0, :].float().cpu().numpy()
if DEVICE == "cuda": torch.cuda.synchronize()
stages["bert"] = (time.perf_counter() - t)

# (j) XGBoost Inference
# Build feature matrix for the 3 texts
X_rows = []
for idx in range(len(texts)):
    row = {"ppl": ppl[idx], "burstiness": b[idx], "ttr": float(tr[idx]), "complex": float(cx[idx]),
           "avg_word_len": float(awl[idx]), "avg_sent_len": float(asl[idx]), "uid": u[idx]}
    row.update({c: float(v) for c, v in zip(pfe.EMPATH_CATS, em.iloc[idx].tolist())})
    row.update({"syntax_depth": float(sd[idx]), "semantic_mean": float(mn[idx]), "semantic_std": float(st_[idx])})
    row.update({f"bert_{k}": float(v) for k, v in enumerate(emb[idx])})
    X_rows.append([row.get(feat, 0.0) for feat in features])

X_mat = np.array(X_rows, dtype=np.float32)

t = time.perf_counter()
Xs = scaler.transform(X_mat)
model.predict(Xs)
stages["xgb_infer"] = (time.perf_counter() - t)

n = len(texts)
print(f"device={DEVICE} N={n}")
per_doc = {}
for k, v in stages.items():
    per_doc[k] = round(v / n * 1000, 1)
    print(f"  {k:12s} {v / n * 1000:9.1f} ms/doc")
total = sum(per_doc.values())
print(f"  {'TOTAL':12s} {total:9.1f} ms/doc  vs RoBERTa ~17.9 ms/doc")
pd.DataFrame([per_doc]).to_csv("results/phase2/T24_Latency.csv", index=False)
print("Saved -> results/phase2/T24_Latency.csv")
