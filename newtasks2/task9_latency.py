"""Task 9: hybrid end-to-end latency per document (11-step features + XGB)
vs RoBERTa (~17.9 ms/sample measured in Task 3). Times 3 sample docs.
"""
import json, time, sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd
import torch

import helper.paraphrase_feature_eval as pfe

df = pd.read_csv("data/processed/processed_paraphrased_articles.csv", usecols=["Text"], nrows=3)
texts = df["Text"].tolist()

with open("results/models/Bin_BERT_features.json") as f:
    features = json.load(f)
model = joblib.load("results/models/Multi_BERT_model.joblib")
scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")

times = {}
t0 = time.time()
from transformers import GPT2LMHeadModel, GPT2Tokenizer
gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
gpt_mod = GPT2LMHeadModel.from_pretrained("gpt2").to(pfe.DEVICE)
if pfe.DEVICE == "cuda":
    gpt_mod = gpt_mod.half()
gpt_mod.eval()
times["load_gpt2"] = time.time() - t0

stages = {}
t = time.time(); ppl = pfe.process_perplexity(texts, gpt_mod, gpt_tok, pfe.DEVICE); stages["perplexity"] = time.time() - t
t = time.time(); b = pfe.process_burstiness(texts); stages["burstiness"] = time.time() - t
t = time.time(); tr = pfe.process_ttr(texts); stages["ttr"] = time.time() - t
t = time.time(); cx, awl, asl = pfe.process_stylometric(texts); stages["stylometric"] = time.time() - t
t = time.time(); u = pfe.process_uid(texts, gpt_mod, gpt_tok, pfe.DEVICE); stages["uid"] = time.time() - t
del gpt_mod, gpt_tok
t = time.time()
from empath import Empath
em = pfe.process_empath(texts, Empath()); stages["empath"] = time.time() - t
t = time.time()
import spacy
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
sd = pfe.process_syntax_depth(texts, nlp); stages["syntax"] = time.time() - t
del nlp
t = time.time()
from sentence_transformers import SentenceTransformer
sm = SentenceTransformer("all-MiniLM-L6-v2").to(pfe.DEVICE)
pfe.process_semantic([texts[0]], sm, pfe.DEVICE)  # warmup (model already loaded above)
mn, st_ = pfe.process_semantic(texts, sm, pfe.DEVICE); stages["semantic"] = time.time() - t
del sm
pfe._load_bert()  # load+compile outside timing
t = time.time(); pfe.get_bert_embeddings_batched([texts[0]]); stages["bert_warmup"] = time.time() - t
t = time.time(); emb = pfe.get_bert_embeddings_batched(texts); stages["bert"] = time.time() - t

row = {"ppl": ppl[0], "burstiness": b[0], "ttr": float(tr[0]), "complex": float(cx[0]),
       "avg_word_len": float(awl[0]), "avg_sent_len": float(asl[0]), "uid": u[0]}
row.update({c: float(v) for c, v in zip(pfe.EMPATH_CATS, em.iloc[0].tolist())})
row.update({"syntax_depth": float(sd[0]), "semantic_mean": float(mn[0]), "semantic_std": float(st_[0])})
row.update({f"bert_{i}": float(v) for i, v in enumerate(emb[0])})
t = time.time()
X = pd.DataFrame([{k: row.get(k, 0) for k in features}], columns=features).astype("float32")
Xs = pd.DataFrame(scaler.transform(X).astype("float32"), columns=features)
model.predict(Xs)
stages["xgb_infer"] = (time.time() - t)

n = len(texts)
print(f"device={pfe.DEVICE} N={n}")
per_doc = {}
for k, v in stages.items():
    per_doc[k] = round(v / n * 1000, 1)
    print(f"  {k:12s} {v / n * 1000:9.1f} ms/doc")
total = sum(per_doc.values())
print(f"  {'TOTAL':12s} {total:9.1f} ms/doc  vs RoBERTa ~17.9 ms/doc")
pd.DataFrame([per_doc]).to_csv("newtasks2/T24_Latency.csv", index=False)
print("Saved -> newtasks2/T24_Latency.csv")
