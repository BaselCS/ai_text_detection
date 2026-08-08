"""
Task 1.3 → 1.4 → 1.5: Multiple Paraphrasers Attack — Full Pipeline
====================================================================
Step 1.3: Feature extraction (11-step) on raw_paraphrased_gpt4o_mini.csv
          and raw_paraphrased_llama3.csv
Step 1.4: Inference using Phase-1 Bin_BERT & Multi_BERT XGBoost models
Step 1.5: Save T7.1 table + F12.1 confusion matrix figures

GPU target: NVIDIA RTX 3060 12 GB VRAM
Run with:   uv run python helper/paraphrase_feature_eval.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. ENV + IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, gc, re, json, warnings
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from tqdm import tqdm
from empath import Empath
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModel,
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
SEED = 999
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True       # faster matmul on Ampere
    torch.backends.cudnn.benchmark = True               # speed up conv
    torch.backends.cudnn.allow_tf32 = True

MAX_WORKERS = min(os.cpu_count() - 1, 4) if os.cpu_count() else 4

# BERT embedding config — large batch to saturate 12 GB VRAM
BERT_MODEL_NAME = "bert-base-uncased"
BERT_BATCH_SIZE = 64        # push GPU hard
BERT_MAX_LENGTH = 512
BERT_OVERLAP    = 50

# Models configuration
MODELS_CONFIG = [
    {"name": "Gemma-2-9B",  "raw": "data/processed/raw_paraphrased_gemma2.csv",       "out": "data/processed/processed_paraphrased_gemma2.csv",       "fig": "Gemma2"},
    {"name": "Qwen-2-72B",  "raw": "data/processed/raw_paraphrased_qwen2.csv",        "out": "data/processed/processed_paraphrased_qwen2.csv",        "fig": "Qwen2"},
    {"name": "LLaMA-3",     "raw": "data/processed/raw_paraphrased_llama3.csv",       "out": "data/processed/processed_paraphrased_llama3.csv",       "fig": "LLaMA"},
    {"name": "Mistral-7B",  "raw": "data/processed/raw_paraphrased_mistral7b.csv",    "out": "data/processed/processed_paraphrased_mistral7b.csv",    "fig": "Mistral7B"},
    {"name": "GPT-4o-mini", "raw": "data/processed/raw_paraphrased_gpt4o_mini.csv",   "out": "data/processed/processed_paraphrased_gpt4o_mini.csv",   "fig": "GPT"},
]

MODEL_DIR  = "results/models"
TABLE_DIR  = "results/tables"
FIGURE_DIR = "results/figures"

for d in [MODEL_DIR, TABLE_DIR, FIGURE_DIR]:
    os.makedirs(d, exist_ok=True)

# Writer name normalisation — remove _paraphrased suffix to match Phase-1 classes
WRITER_CLEAN = {
    "GPT-4o_paraphrased":      "GPT-4o",
    "Gemma-2-9B_paraphrased":  "Gemma-2-9B",
    "Mistral-7B_paraphrased":  "Mistral-7B",
    "Llama-8B_paraphrased":    "Llama-8B",
    "Qwen-2-72B_paraphrased":  "Qwen-2-72B",
    "Yi-Large_paraphrased":    "Yi-Large",
    "Human_paraphrased":       "Human",
}

# Empath categories used in Phase 1
EMPATH_CATS = [
    "gain", "beauty", "government", "urban", "art",
    "help", "optimism", "strength", "love", "traveling",
]

print(f"[CONFIG] Device : {DEVICE}")
print(f"[CONFIG] BERT batch size : {BERT_BATCH_SIZE}")
print(f"[CONFIG] Workers : {MAX_WORKERS}")
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"[CONFIG] GPU : {props.name}  VRAM={props.total_memory/1024**3:.1f} GB")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION HELPERS  (mirrors data_engineer.ipynb exactly)
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_text(text) -> bool:
    return isinstance(text, str) and len(text.strip()) > 10


# ── 2a. Perplexity (GPT-2 sliding-window) ────────────────────────────────────
def calculate_perplexity(text, model, tokenizer, device, stride=512):
    if not is_valid_text(text):
        return None
    orig_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    tokenizer.model_max_length = orig_max
    seq_len = enc.input_ids.size(1)
    max_len = model.config.n_positions
    nlls, prev_end = [], 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        trg_len = end - prev_end
        ids = enc.input_ids[:, begin:end].to(device)
        tgt = ids.clone()
        tgt[:, :-trg_len] = -100
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(ids, labels=tgt)
            nlls.append(out.loss * trg_len)
        prev_end = end
        if end == seq_len:
            break
    return torch.exp(torch.stack(nlls).sum() / seq_len).item()


def process_perplexity(texts, model, tokenizer, device):
    return [calculate_perplexity(t, model, tokenizer, device)
            for t in tqdm(texts, desc="[2/11] Perplexity")]


# ── 2b. Burstiness ────────────────────────────────────────────────────────────
def process_burstiness(texts):
    results = []
    for text in tqdm(texts, desc="[3/11] Burstiness"):
        if is_valid_text(text):
            lengths = np.array([len(s.split()) for s in sent_tokenize(text) if s.split()])
            results.append(float(np.std(lengths)) if len(lengths) >= 2 else 0.0)
        else:
            results.append(0.0)
    return results


# ── 2c. TTR ───────────────────────────────────────────────────────────────────
def process_ttr(texts):
    out = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, desc="[4/11] TTR")):
        if is_valid_text(text):
            words = re.findall(r"\w+", text.lower())
            if words:
                out[i] = len(set(words)) / len(words)
    return out


# ── 2d. Stylometric (complex-ratio, avg-word-len, avg-sent-len) ───────────────
def process_stylometric(texts):
    cx = np.zeros(len(texts), dtype=np.float32)
    awl = np.zeros(len(texts), dtype=np.float32)
    asl = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, desc="[5/11] Stylometric")):
        if is_valid_text(text):
            words = re.findall(r"\w+", text.lower())
            sents = sent_tokenize(text)
            if words and sents:
                awl[i] = float(np.mean([len(w) for w in words]))
                asl[i] = float(len(words) / len(sents))
                cx[i]  = float(len([w for w in words if len(w) > 6]) / len(words))
    return cx, awl, asl


# ── 2e. UID (Uniform Information Density) ────────────────────────────────────
def calculate_uid(text, model, tokenizer, device, max_len=1024, stride=512):
    if not is_valid_text(text):
        return 0.0
    orig_max = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30)
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    tokenizer.model_max_length = orig_max
    ids = inputs["input_ids"]
    seq_len = ids.size(1)
    if seq_len < 2:
        return 0.0
    surprisals = []
    for begin in range(0, seq_len, stride):
        end = min(begin + max_len, seq_len)
        chunk = ids[:, begin:end].to(device)
        if chunk.shape[1] < 2:
            continue
        with torch.no_grad(), torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(chunk, labels=chunk)
            logits = out.logits
        sl = logits[:, :-1, :].contiguous()
        tl = chunk[:, 1:].contiguous()
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        surprisals.append(loss_fn(sl.view(-1, sl.size(-1)), tl.view(-1)))
        if end >= seq_len:
            break
    if not surprisals:
        return 0.0
    return float(torch.std(torch.cat(surprisals)).item())


def process_uid(texts, model, tokenizer, device):
    return [calculate_uid(t, model, tokenizer, device)
            for t in tqdm(texts, desc="[6/11] UID")]


# ── 2f. Empath ────────────────────────────────────────────────────────────────
def process_empath(texts, lexicon, categories=EMPATH_CATS):
    feats = []
    for i in tqdm(range(0, len(texts), 32), desc="[7/11] Empath"):
        for text in texts[i:i+32]:
            if is_valid_text(text):
                res = lexicon.analyze(text, categories=categories, normalize=True) or {}
            else:
                res = {}
            feats.append({c: res.get(c, 0.0) for c in categories})
    return pd.DataFrame(feats)


# ── 2g. spaCy Syntax Depth ────────────────────────────────────────────────────
def _walk(node, d):
    if node.n_lefts + node.n_rights == 0:
        return d
    return max(_walk(c, d + 1) for c in node.children)


def process_syntax_depth(texts, nlp):
    depths = []
    for text in tqdm(texts, desc="[8/11] Syntax Depth"):
        if is_valid_text(text):
            doc = nlp(text)
            ds  = [_walk(s.root, 1) for s in doc.sents]
            depths.append(float(np.mean(ds)) if ds else 0.0)
        else:
            depths.append(0.0)
    return depths


# ── 2h. Semantic Consistency (MiniLM) ─────────────────────────────────────────
def process_semantic(texts, sem_model, device):
    means = np.zeros(len(texts), dtype=np.float32)
    stds  = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(tqdm(texts, desc="[9/11] Semantic")):
        if not is_valid_text(text):
            continue
        sents = [s for s in sent_tokenize(text) if len(s.split()) > 1]
        if len(sents) < 2:
            continue
        with torch.no_grad():
            emb = sem_model.encode(sents, convert_to_tensor=True, device=device,
                                   batch_size=64, show_progress_bar=False)
        sims = torch.tensor([util.cos_sim(emb[j], emb[j+1]).item()
                              for j in range(len(emb)-1)])
        means[i] = float(torch.mean(sims).item())
        stds[i]  = float(torch.std(sims).item()) if len(sims) > 1 else 0.0
    return means, stds


# ── 2i. BERT Embeddings (768-dim) ─────────────────────────────────────────────
_bert_tokenizer = None
_bert_model     = None


def _load_bert():
    global _bert_tokenizer, _bert_model
    if _bert_model is None:
        print(f"  [BERT] Loading {BERT_MODEL_NAME} …")
        _bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        _bert_model     = AutoModel.from_pretrained(BERT_MODEL_NAME)
        for p in _bert_model.parameters():
            p.requires_grad = False
        _bert_model = _bert_model.to(DEVICE).half()   # fp16 for speed + VRAM
        _bert_model = torch.compile(_bert_model)       # torch.compile speedup
        _bert_model.eval()
        print(f"  [BERT] Model on {DEVICE} (fp16 + compiled)")


def _get_token_chunks(text, max_len=BERT_MAX_LENGTH, overlap=BERT_OVERLAP):
    """
    Tokenise `text` without truncation, split token-ID list into overlapping
    windows, then re-tokenise each window with padding so every returned
    dict has consistent shape (1, max_len).
    Returns list of BatchEncoding dicts, each with 2-D tensors.
    """
    # 1. Encode the full text to raw token IDs (no special tokens yet)
    token_ids = _bert_tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
    )  # plain Python list of ints

    inner_len = max_len - 2        # space after [CLS] and [SEP]
    step      = inner_len - overlap

    chunks = []
    for start in range(0, max(1, len(token_ids)), step):
        window = token_ids[start: start + inner_len]
        if not window:
            break
        # Re-encode as a short list: add special tokens + pad to max_len
        enc = _bert_tokenizer(
            _bert_tokenizer.decode(window, skip_special_tokens=True),
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",   # always returns (1, max_len) tensors
        )
        chunks.append(enc)
        if start + inner_len >= len(token_ids):
            break

    return chunks if chunks else [_bert_tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )]


def get_bert_embeddings_batched(texts):
    """Process all texts in large GPU batches → returns (N, 768) float32 ndarray."""
    _load_bert()
    N = len(texts)
    embeddings = np.zeros((N, 768), dtype=np.float32)

    # Pre-build flat list of chunks + owner index
    all_input_ids      = []   # each: (1, max_len) int tensor
    all_attention_mask = []   # each: (1, max_len) int tensor
    all_token_type_ids = []   # each: (1, max_len) int tensor
    chunk_owner        = []   # int — which text index owns this chunk

    for idx, text in enumerate(texts):
        if is_valid_text(text):
            for enc in _get_token_chunks(text):
                all_input_ids.append(enc["input_ids"])            # shape (1, L)
                all_attention_mask.append(enc["attention_mask"])  # shape (1, L)
                tt = enc.get(
                    "token_type_ids",
                    torch.zeros_like(enc["input_ids"]),
                )
                all_token_type_ids.append(tt)                     # shape (1, L)
                chunk_owner.append(idx)

    total_chunks = len(all_input_ids)
    print(f"  [BERT] {N} texts → {total_chunks} chunks  (batch={BERT_BATCH_SIZE})")

    chunk_acc = {i: [] for i in range(N)}

    for batch_start in tqdm(
        range(0, total_chunks, BERT_BATCH_SIZE),
        desc="[11/11] BERT Embeddings",
    ):
        sl = slice(batch_start, batch_start + BERT_BATCH_SIZE)
        # All tensors are (1, max_len) → cat along dim=0 gives (B, max_len)
        b_ids   = torch.cat(all_input_ids[sl],      dim=0).to(DEVICE)
        b_mask  = torch.cat(all_attention_mask[sl],  dim=0).to(DEVICE)
        b_tt    = torch.cat(all_token_type_ids[sl],  dim=0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE, enabled=(DEVICE == "cuda")
        ):
            out = _bert_model(
                input_ids=b_ids,
                attention_mask=b_mask,
                token_type_ids=b_tt,
            )
        # CLS embedding per chunk: (B, 768)
        cls = out.last_hidden_state[:, 0, :].float().cpu().numpy()

        for i, owner_idx in enumerate(chunk_owner[sl]):
            chunk_acc[owner_idx].append(cls[i])

    # Average chunks per text
    for idx in range(N):
        if chunk_acc[idx]:
            embeddings[idx] = np.mean(chunk_acc[idx], axis=0)

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL FEATURE PIPELINE  (11 steps, checkpoint-resume aware)
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(input_csv: str, output_csv: str, paraphraser_name: str):
    """
    Run the full 11-step feature extraction pipeline on a raw paraphrased CSV.

    Input CSV columns : Writer, Original_Text, Paraphrased_Text
    Output CSV columns: Writer, Text, ppl, burstiness, ttr, complex,
                        avg_word_len, avg_sent_len, uid, gain…traveling,
                        syntax_depth, semantic_mean, semantic_std, is_AI,
                        bert_0…bert_767
    """
    ckpt = output_csv.replace(".csv", "_checkpoint.csv")

    print(f"\n{'='*70}")
    print(f"  FEATURE PIPELINE  |  {paraphraser_name}")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_csv}")
    print(f"{'='*70}\n")

    # ── resume detection ──────────────────────────────────────────────────────
    if os.path.exists(ckpt):
        df = pd.read_csv(ckpt)
        if "bert_0" in df.columns:
            print("[RESUME] All steps done → saving final file.")
            df.to_csv(output_csv, index=False)
            os.remove(ckpt)
            return df
        print(f"[RESUME] Checkpoint found ({len(df)} rows). Resuming…")
    else:
        # ── Step 1/11: Load & prepare ─────────────────────────────────────────
        print("[1/11] Loading and preparing dataset…")
        df = pd.read_csv(input_csv)

        # Use paraphrased text as the evaluation text
        df = df.rename(columns={"Paraphrased_Text": "Text"})
        df["Writer"] = df["Writer"].map(WRITER_CLEAN).fillna(df["Writer"])
        df = df[["Writer", "Text"]].dropna(subset=["Text"]).reset_index(drop=True)

        print(f"  Rows: {len(df)}")
        print(f"  Writers: {df['Writer'].value_counts().to_dict()}")

        # ── Step 2/11: Perplexity (GPT-2) ────────────────────────────────────
        print("\n[2/11] Loading GPT-2 for Perplexity & UID…")
        gpt_tok = GPT2Tokenizer.from_pretrained("gpt2")
        gpt_mod = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt_mod = gpt_mod.to(DEVICE)
        if DEVICE == "cuda":
            gpt_mod = gpt_mod.half()
        gpt_mod.eval()

        df["ppl"] = process_perplexity(df["Text"].tolist(), gpt_mod, gpt_tok, DEVICE)
        df.to_csv(ckpt, index=False)

        # ── Step 3/11: Burstiness ─────────────────────────────────────────────
        df["burstiness"] = process_burstiness(df["Text"].tolist())
        df.to_csv(ckpt, index=False)

        # ── Step 4/11: TTR ────────────────────────────────────────────────────
        df["ttr"] = process_ttr(df["Text"].tolist())
        df.to_csv(ckpt, index=False)

        # ── Step 5/11: Stylometric ────────────────────────────────────────────
        cx, awl, asl = process_stylometric(df["Text"].tolist())
        df["complex"]      = cx
        df["avg_word_len"] = awl
        df["avg_sent_len"] = asl
        df.to_csv(ckpt, index=False)

        # ── Step 6/11: UID ────────────────────────────────────────────────────
        df["uid"] = process_uid(df["Text"].tolist(), gpt_mod, gpt_tok, DEVICE)
        df.to_csv(ckpt, index=False)

        # Free GPT-2 from VRAM
        del gpt_mod, gpt_tok
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # ── Step 7/11: Empath ─────────────────────────────────────────────────
        lexicon = Empath()
        empath_df = process_empath(df["Text"].tolist(), lexicon)
        df = pd.concat([df, empath_df], axis=1)
        df.to_csv(ckpt, index=False)

    # ── Step 8/11: Syntax Depth ──────────────────────────────────────────────
    if "syntax_depth" not in df.columns:
        print("\n[8/11] Syntax Depth (spaCy)…")
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        df["syntax_depth"] = process_syntax_depth(df["Text"].tolist(), nlp)
        del nlp
        gc.collect()
        df.to_csv(ckpt, index=False)
    else:
        print("[8/11] Syntax Depth — already done, skipping.")

    # ── Step 9/11: Semantic Consistency ─────────────────────────────────────
    if "semantic_mean" not in df.columns:
        print("\n[9/11] Semantic Consistency (MiniLM)…")
        sem_model = SentenceTransformer("all-MiniLM-L6-v2")
        sem_model = sem_model.to(DEVICE)
        means, stds = process_semantic(df["Text"].tolist(), sem_model, DEVICE)
        df["semantic_mean"] = means
        df["semantic_std"]  = stds
        del sem_model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        df.to_csv(ckpt, index=False)
    else:
        print("[9/11] Semantic Consistency — already done, skipping.")

    # ── Step 10/11: is_AI label ──────────────────────────────────────────────
    if "is_AI" not in df.columns:
        print("\n[10/11] Adding is_AI label…")
        df["is_AI"] = (df["Writer"].str.lower() != "human").astype(int)
        df = df.dropna().reset_index(drop=True)
        df.to_csv(ckpt, index=False)
    else:
        print("[10/11] is_AI — already done, skipping.")

    # ── Step 11/11: BERT Embeddings ──────────────────────────────────────────
    if "bert_0" not in df.columns:
        print("\n[11/11] BERT Embeddings (bert-base-uncased)…")
        bert_emb = get_bert_embeddings_batched(df["Text"].tolist())
        bert_cols = [f"bert_{i}" for i in range(bert_emb.shape[1])]
        df_bert = pd.DataFrame(bert_emb, columns=bert_cols, index=df.index)
        df = pd.concat([df, df_bert], axis=1)
        del bert_emb, df_bert
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    else:
        print("[11/11] BERT Embeddings — already done, skipping.")

    # ── Final save ───────────────────────────────────────────────────────────
    print(f"\n[DONE] Saving → {output_csv}  (shape={df.shape})")
    df.to_csv(output_csv, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. PHASE-1 MODEL TRAINER  (re-trains if .joblib files are missing)
# ─────────────────────────────────────────────────────────────────────────────

def _train_xgboost(X, y, task_type):
    """Train XGBoost with same hyperparams as Phase 1 Main.ipynb."""
    if task_type == "binary":
        human_idx = y[y == 0].index
        ai_idx    = y[y == 1].index
        n_human   = len(human_idx)
        n_ai      = len(ai_idx)
        if n_human < n_ai:
            rng = np.random.RandomState(SEED)
            ai_sample = rng.choice(ai_idx, size=n_human, replace=False)
            idx = np.concatenate([human_idx, ai_sample])
            X, y = X.loc[idx], y.loc[idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr).astype("float32"), columns=X.columns)
    X_te_s = pd.DataFrame(scaler.transform(X_te).astype("float32"),     columns=X.columns)

    objective = "binary:logistic" if task_type == "binary" else "multi:softprob"
    model = xgb.XGBClassifier(
        objective          = objective,
        tree_method        = "hist",
        device             = "cuda" if torch.cuda.is_available() else "cpu",
        n_estimators       = 2000,
        early_stopping_rounds = 100,
        learning_rate      = 0.01,
        max_depth          = 10,
        random_state       = SEED,
        n_jobs             = MAX_WORKERS,
    )
    model.fit(X_tr_s, y_tr, eval_set=[(X_te_s, y_te)], verbose=False)
    acc = accuracy_score(y_te, model.predict(X_te_s))
    print(f"  Accuracy: {acc:.4f}")
    return model, scaler, X.columns


def ensure_phase1_models():
    """
    Returns dict with keys 'Bin_BERT', 'Multi_BERT', each containing
    {model, scaler, features, le (for multi)}.
    Re-trains from processed_articles.csv if .joblib files are missing.
    """
    bin_model_path   = f"{MODEL_DIR}/Bin_BERT_model.joblib"
    bin_scaler_path  = f"{MODEL_DIR}/Bin_BERT_scaler.joblib"
    multi_model_path = f"{MODEL_DIR}/Multi_BERT_model.joblib"
    multi_scaler_path= f"{MODEL_DIR}/Multi_BERT_scaler.joblib"
    le_path          = f"{MODEL_DIR}/Multi_BERT_label_encoder.joblib"
    feat_path        = f"{MODEL_DIR}/Bin_BERT_features.json"

    all_exist = all(os.path.exists(p) for p in [
        bin_model_path, bin_scaler_path,
        multi_model_path, multi_scaler_path, le_path, feat_path
    ])

    if all_exist:
        print("[MODELS] Loading saved Phase-1 models…")
        with open(feat_path) as f:
            features = json.load(f)
        return {
            "Bin_BERT": {
                "model":    joblib.load(bin_model_path),
                "scaler":   joblib.load(bin_scaler_path),
                "features": features,
            },
            "Multi_BERT": {
                "model":    joblib.load(multi_model_path),
                "scaler":   joblib.load(multi_scaler_path),
                "features": features,
                "le":       joblib.load(le_path),
            },
        }

    # ── Re-train from processed_articles.csv ─────────────────────────────────
    print("\n[MODELS] .joblib files not found → re-training Phase-1 models…")
    print("  Loading processed_articles.csv …")
    df = pd.read_csv("data/processed/processed_articles.csv")
    df = df.drop(columns=["Text"], errors="ignore")

    bert_cols = [c for c in df.columns if c.startswith("bert_")]
    X_all = df.drop(columns=["is_AI", "Writer"])

    # Binary
    le_bin = LabelEncoder()
    y_bin  = df["is_AI"]

    # Multi-class
    le_multi = LabelEncoder()
    y_multi  = le_multi.fit_transform(df["Writer"])

    print("\n  Training Bin_BERT …")
    bin_mod, bin_scal, bin_feats = _train_xgboost(X_all, y_bin, "binary")

    print("\n  Training Multi_BERT …")
    multi_mod, multi_scal, _ = _train_xgboost(X_all, pd.Series(y_multi, index=df.index), "multiclass")

    # Save
    joblib.dump(bin_mod,   bin_model_path)
    joblib.dump(bin_scal,  bin_scaler_path)
    joblib.dump(multi_mod, multi_model_path)
    joblib.dump(multi_scal,multi_scaler_path)
    joblib.dump(le_multi,  le_path)
    with open(feat_path, "w") as f:
        json.dump(list(bin_feats), f, indent=2)

    print(f"  Models saved to {MODEL_DIR}/")

    return {
        "Bin_BERT": {
            "model":    bin_mod,
            "scaler":   bin_scal,
            "features": list(bin_feats),
        },
        "Multi_BERT": {
            "model":    multi_mod,
            "scaler":   multi_scal,
            "features": list(bin_feats),
            "le":       le_multi,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE + EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title):
    """Publication-quality confusion matrix heatmap."""
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, fmt, ttl in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Count", "Normalised (Row %)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.4, linecolor="white",
            ax=ax, cbar=True,
            annot_kws={"size": 9},
        )
        ax.set_title(f"{title}\n{ttl}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Class",      fontsize=11)
        ax.tick_params(axis="x", rotation=35, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


def evaluate_on_dataset(processed_csv: str, models: dict, paraphraser_name: str,
                        fig_suffix: str) -> dict:
    """
    Run Bin_BERT and Multi_BERT inference on a processed paraphrased dataset.
    Returns a result dict for T7.1.
    """
    print(f"\n{'─'*60}")
    print(f"  EVALUATION  |  {paraphraser_name}")
    print(f"{'─'*60}")

    df = pd.read_csv(processed_csv)

    features   = models["Bin_BERT"]["features"]
    bin_model  = models["Bin_BERT"]["model"]
    bin_scaler = models["Bin_BERT"]["scaler"]
    mul_model  = models["Multi_BERT"]["model"]
    mul_scaler = models["Multi_BERT"]["scaler"]
    le         = models["Multi_BERT"]["le"]

    # ── Binary evaluation ─────────────────────────────────────────────────────
    y_bin_true = df["is_AI"]
    X_bin = df.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc = pd.DataFrame(
        bin_scaler.transform(X_bin).astype("float32"), columns=features
    )
    y_bin_pred = bin_model.predict(X_bin_sc)

    bin_acc = accuracy_score(y_bin_true, y_bin_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_bin_true, y_bin_pred, average="macro", zero_division=0
    )
    print(f"\n  [Binary]  Acc={bin_acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # ── Multi-class evaluation ────────────────────────────────────────────────
    valid_classes = list(le.classes_)
    df_multi = df[df["Writer"].isin(valid_classes)].copy()
    y_mul_true = le.transform(df_multi["Writer"])

    X_mul = df_multi.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc = pd.DataFrame(
        mul_scaler.transform(X_mul).astype("float32"), columns=features
    )
    y_mul_pred = mul_model.predict(X_mul_sc)

    mul_acc = accuracy_score(y_mul_true, y_mul_pred)
    mul_prec, mul_rec, mul_f1, _ = precision_recall_fscore_support(
        y_mul_true, y_mul_pred, average="macro", zero_division=0
    )
    print(f"  [Multi ]  Acc={mul_acc:.4f}  P={mul_prec:.4f}  R={mul_rec:.4f}  F1={mul_f1:.4f}")

    # class-level report for insight
    print("\n  Per-class report:")
    print(classification_report(
        y_mul_true, y_mul_pred,
        target_names=valid_classes,
        zero_division=0,
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    fig_path = f"{FIGURE_DIR}/F12.1_Paraphrased_{fig_suffix}.png"
    plot_confusion_matrix(
        y_mul_true, y_mul_pred,
        class_names=valid_classes,
        save_path=fig_path,
        title=f"Multi-Class Attribution — {paraphraser_name} Paraphraser",
    )

    return {
        "Paraphraser": paraphraser_name,
        "Binary_Accuracy": round(bin_acc, 4),
        "Binary_Macro_Precision": round(prec, 4),
        "Binary_Macro_Recall":    round(rec,  4),
        "Binary_Macro_F1":        round(f1,   4),
        "Multi_Accuracy":         round(mul_acc,  4),
        "Multi_Macro_Precision":  round(mul_prec, 4),
        "Multi_Macro_Recall":     round(mul_rec,  4),
        "Multi_Macro_F1":         round(mul_f1,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. BASELINE ROW (Gemma-27B from existing processed_paraphrased_articles.csv)
# ─────────────────────────────────────────────────────────────────────────────

def get_baseline_row(models: dict) -> dict | None:
    """Load existing Gemma-27B paraphrased results to populate T7.1 baseline."""
    baseline_csv = "data/processed/processed_paraphrased_articles.csv"
    if not os.path.exists(baseline_csv):
        print(f"  [BASELINE] {baseline_csv} not found — skipping.")
        return None

    print(f"\n  [BASELINE] Evaluating Gemma-27B (existing) …")
    features   = models["Bin_BERT"]["features"]
    bin_model  = models["Bin_BERT"]["model"]
    bin_scaler = models["Bin_BERT"]["scaler"]
    mul_model  = models["Multi_BERT"]["model"]
    mul_scaler = models["Multi_BERT"]["scaler"]
    le         = models["Multi_BERT"]["le"]

    df = pd.read_csv(baseline_csv)
    if "is_AI" not in df.columns:
        df["is_AI"] = (df["Writer"].str.lower() != "human").astype(int)

    y_bin_true = df["is_AI"]
    X_bin = df.reindex(columns=features, fill_value=0).astype("float32")
    X_bin_sc = pd.DataFrame(
        bin_scaler.transform(X_bin).astype("float32"), columns=features
    )
    y_bin_pred = bin_model.predict(X_bin_sc)
    bin_acc = accuracy_score(y_bin_true, y_bin_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_bin_true, y_bin_pred, average="macro", zero_division=0
    )

    valid_classes = list(le.classes_)
    df_multi = df[df["Writer"].isin(valid_classes)].copy()
    if df_multi.empty:
        return None
    y_mul_true = le.transform(df_multi["Writer"])
    X_mul = df_multi.reindex(columns=features, fill_value=0).astype("float32")
    X_mul_sc = pd.DataFrame(
        mul_scaler.transform(X_mul).astype("float32"), columns=features
    )
    y_mul_pred = mul_model.predict(X_mul_sc)
    mul_acc  = accuracy_score(y_mul_true, y_mul_pred)
    mul_prec, mul_rec, mul_f1, _ = precision_recall_fscore_support(
        y_mul_true, y_mul_pred, average="macro", zero_division=0
    )

    return {
        "Paraphraser": "Gemma-27B (baseline)",
        "Binary_Accuracy":        round(bin_acc,  4),
        "Binary_Macro_Precision": round(prec,     4),
        "Binary_Macro_Recall":    round(rec,      4),
        "Binary_Macro_F1":        round(f1,       4),
        "Multi_Accuracy":         round(mul_acc,  4),
        "Multi_Macro_Precision":  round(mul_prec, 4),
        "Multi_Macro_Recall":     round(mul_rec,  4),
        "Multi_Macro_F1":         round(mul_f1,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print("  TASK 1.3 → 1.4 → 1.5 : MULTIPLE PARAPHRASERS ATTACK PIPELINE")
    print("="*70)

    # ── Step 1.3: Feature Extraction ─────────────────────────────────────────
    print("\n>>> STEP 1.3 — Feature Extraction")

    for cfg in MODELS_CONFIG:
        if not os.path.exists(cfg["out"]):
            if not os.path.exists(cfg["raw"]):
                print(f"[SKIP] Raw file {cfg['raw']} not found.")
                continue
            run_feature_pipeline(cfg["raw"], cfg["out"], cfg["name"])
        else:
            print(f"[SKIP] {cfg['out']} already exists.")

    # ── Load / Re-train Phase-1 models ────────────────────────────────────────
    print("\n>>> Ensuring Phase-1 XGBoost models …")
    models = ensure_phase1_models()

    # ── Step 1.4: Inference + Evaluation ─────────────────────────────────────
    print("\n>>> STEP 1.4 — Inference & Evaluation")

    results = []

    # Baseline Gemma-27B
    baseline = get_baseline_row(models)
    if baseline:
        results.append(baseline)

    for cfg in MODELS_CONFIG:
        if os.path.exists(cfg["out"]):
            res = evaluate_on_dataset(
                processed_csv   = cfg["out"],
                models          = models,
                paraphraser_name= cfg["name"],
                fig_suffix      = cfg["fig"],
            )
            results.append(res)
        else:
            print(f"[SKIP] {cfg['out']} missing for evaluation.")

    # ── Step 1.5: Save Table T7.1 ────────────────────────────────────────────
    print("\n>>> STEP 1.5 — Saving Table & Figures")
    t71_path = f"{TABLE_DIR}/T7.1_Expanded_Paraphrasers.csv"
    t71_df   = pd.DataFrame(results)
    t71_df.to_csv(t71_path, index=False)
    print(f"\n  Saved → {t71_path}")
    print("\n" + t71_df.to_string(index=False))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  PIPELINE COMPLETE")
    print("="*70)
    for cfg in MODELS_CONFIG:
        if os.path.exists(cfg["out"]):
            print(f"  Feature CSV : {cfg['out']}")
    print(f"  Table       : {t71_path}")
    for cfg in MODELS_CONFIG:
        fig_path = f"{FIGURE_DIR}/F12.1_Paraphrased_{cfg['fig']}.png"
        if os.path.exists(fig_path):
            print(f"  Figure      : {fig_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
