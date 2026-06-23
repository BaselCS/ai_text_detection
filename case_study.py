import os
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 999
DEVICE = "cpu"

print("Loading data...")
df_full = pd.read_csv('data/processed/processed_articles.csv')
df_para = pd.read_csv('data/processed/processed_paraphrased_articles.csv')

# 1. Load Metadata and Models FIRST
print("Loading models and metadata...")
with open("results/models/Multi_BERT_metadata.json", 'r') as f:
    features_multi = json.load(f)['feature_list']

with open("results/models/Bin_BERT_metadata.json", 'r') as f:
    features_bin = json.load(f)['feature_list']

scaler_multi = joblib.load("results/models/Multi_BERT_scaler.joblib")
scaler_bin = joblib.load("results/models/Bin_BERT_scaler.joblib")

bin_model_path = "results/models/Bin_BERT_model.joblib"
multi_model_path = "results/models/Multi_BERT_model.joblib"

bin_model = joblib.load(bin_model_path)

if os.path.exists(multi_model_path):
    multi_model = joblib.load(multi_model_path)
    le_multi = LabelEncoder()
    le_multi.fit(df_full['Writer'])
else:
    print("Multi_BERT_model not found. Re-training in memory...")
    le_multi = LabelEncoder()
    y_multi_full = le_multi.fit_transform(df_full['Writer'])
    X_multi_all = df_full.drop(columns=['is_AI', 'Writer', 'Text'], errors='ignore')
    
    X_tr_multi, X_te_multi, y_tr_multi, y_te_multi = train_test_split(
        X_multi_all, y_multi_full, test_size=0.20, random_state=SEED, stratify=y_multi_full
    )
    X_train = X_tr_multi[features_multi]
    X_train_sc = pd.DataFrame(scaler_multi.transform(X_train), columns=X_train.columns).astype('float32')
    X_test_sc = pd.DataFrame(scaler_multi.transform(X_te_multi[features_multi]), columns=X_train.columns).astype('float32')
    
    params = {
        'objective': 'multi:softprob', 'tree_method': 'hist', 'device': DEVICE,
        'n_estimators': 2000, 'early_stopping_rounds': 100,
        'learning_rate': 0.01, 'max_depth': 10,
        'random_state': SEED, 'n_jobs': -1
    }
    multi_model = xgb.XGBClassifier(**params)
    multi_model.fit(X_train_sc, y_tr_multi, eval_set=[(X_test_sc, y_te_multi)], verbose=False)

classes = le_multi.classes_


bin_model.set_params(device=DEVICE)
if multi_model is not None:
    multi_model.set_params(device=DEVICE)

# 2. Match Paraphrased Articles to their Original counterparts using word overlap
print("Searching for the Golden Sample...")
target_uid = None
doc_orig = None
doc_para = None
probs_orig_multi = None
probs_para_multi = None
target_writer = None

import re
def to_word_set(t):
    if not isinstance(t, str):
        return set()
    # Normalize and keep only words with 4 or more letters to focus on semantic content
    return set(re.findall(r'\b\w{4,}\b', t.lower()))

# Prioritize Mistral-7B as requested, but allow other non-Gemma sources as fallback
candidate_writers = ['Mistral-7B', 'Llama-8B', 'Qwen-2-72B', 'Yi-Large', 'GPT-4o']

for writer in candidate_writers:
    print(f"Checking candidates for writer: {writer}")
    df_orig_writer = df_full[df_full['Writer'] == writer]
    df_para_writer = df_para[df_para['Writer'] == writer]
    
    if df_orig_writer.empty or df_para_writer.empty:
        print(f"  No articles found for writer {writer}")
        continue
        
    print(f"  Indexing {len(df_orig_writer)} original articles...")
    orig_pools = []
    for _, row in df_orig_writer.iterrows():
        orig_pools.append((to_word_set(row['Text']), row))
        
    print(f"  Matching {len(df_para_writer)} paraphrased articles...")
    for _, doc_para_temp in df_para_writer.iterrows():
        para_text = doc_para_temp['Text']
        para_words = to_word_set(para_text)
        
        best_overlap = 0
        doc_orig_temp = None
        
        for orig_words, row_orig in orig_pools:
            overlap = len(para_words.intersection(orig_words))
            if overlap > best_overlap:
                best_overlap = overlap
                doc_orig_temp = row_orig
                
        # Require a solid threshold of shared unique long words (at least 20 words)
        # to ensure it's the exact same article and not a different one on the same topic
        if doc_orig_temp is None or best_overlap < 20:
            continue
            
        # Safely convert to float32 DataFrame before transforming to avoid sklearn warnings on object arrays
        X_orig_multi_temp = pd.DataFrame([doc_orig_temp[features_multi]]).astype('float32')
        X_para_multi_temp = pd.DataFrame([doc_para_temp[features_multi]]).astype('float32')
        
        X_orig_multi_sc_temp = scaler_multi.transform(X_orig_multi_temp).astype('float32')
        X_para_multi_sc_temp = scaler_multi.transform(X_para_multi_temp).astype('float32')
        
        probs_orig_temp = multi_model.predict_proba(X_orig_multi_sc_temp)[0]
        probs_para_temp = multi_model.predict_proba(X_para_multi_sc_temp)[0]
        
        pred_orig_temp = classes[np.argmax(probs_orig_temp)]
        pred_para_temp = classes[np.argmax(probs_para_temp)]
        
        # Check if original is correctly attributed to the source, and paraphrased shifts to Gemma-2-9B
        if pred_orig_temp == writer and pred_para_temp == 'Gemma-2-9B':
            target_uid = doc_orig_temp['uid']
            doc_orig = doc_orig_temp
            doc_para = doc_para_temp
            probs_orig_multi = probs_orig_temp
            probs_para_multi = probs_para_temp
            target_writer = writer
            break
            
    if target_uid is not None:
        break

if not target_uid:
    print("Could not find a perfect sample that shifts from a non-Gemma model to Gemma-2-9B.")
    exit(1)

print(f"Golden Sample Found! UID: {target_uid} (True Source: {target_writer})")

# 3. Extract clean sentence-boundary-aligned excerpts (2-3 sentences)
def get_clean_excerpt(text, count=3):
    import re
    # Split text into sentences using basic sentence boundary heuristics
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    excerpt = " ".join(sentences[:count])
    if len(sentences) > count:
        excerpt += " ..."
    return excerpt

text_orig = str(doc_orig['Text'])
text_para = str(doc_para['Text'])

excerpt_orig = get_clean_excerpt(text_orig, 3)
excerpt_para = get_clean_excerpt(text_para, 3)

print("="*60)
print(f"Original Excerpt ({target_writer}):")
print(excerpt_orig)
print("-" * 60)
print("Paraphrased Excerpt (Gemma):")
print(excerpt_para)
print("="*60)

# 4. Get Binary Probabilities for the Golden Sample
X_orig_bin = pd.DataFrame([doc_orig[features_bin]]).astype('float32')
X_para_bin = pd.DataFrame([doc_para[features_bin]]).astype('float32')

X_orig_bin_sc = scaler_bin.transform(X_orig_bin).astype('float32')
X_para_bin_sc = scaler_bin.transform(X_para_bin).astype('float32')

probs_orig_bin = bin_model.predict_proba(X_orig_bin_sc)[0][1]
probs_para_bin = bin_model.predict_proba(X_para_bin_sc)[0][1]

pred_orig = classes[np.argmax(probs_orig_multi)]
pred_para = classes[np.argmax(probs_para_multi)]

print(f"Original: Predicted Class = {pred_orig}, P(AI) = {probs_orig_bin:.4f}")
for c, p in zip(classes, probs_orig_multi):
    print(f"  {c}: {p:.4f}")

print(f"Paraphrased: Predicted Class = {pred_para}, P(AI) = {probs_para_bin:.4f}")
for c, p in zip(classes, probs_para_multi):
    print(f"  {c}: {p:.4f}")

# 5. Plot the results with highly premium aesthetics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

# Colors: slate grey for inactive classes, royal blue for target writer, rose crimson for Gemma
color_grey = '#cbd5e1'
color_target = '#2563eb'
color_gemma = '#e11d48'

colors_orig = [color_target if c == target_writer else color_grey for c in classes]
bars1 = ax1.bar(classes, probs_orig_multi, color=colors_orig, edgecolor='black', linewidth=0.5, alpha=0.9)
ax1.set_title(f"Panel A-Before: Original {target_writer} Excerpt\nPredicted: {pred_orig} | P(AI) = {probs_orig_bin:.3f}", fontweight="bold", fontsize=11, pad=12)
ax1.set_ylabel("Attribution Probability", fontweight="bold", fontsize=10)
ax1.tick_params(axis='x', rotation=45, labelsize=9)
ax1.tick_params(axis='y', labelsize=9)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

colors_para = [color_gemma if c == 'Gemma-2-9B' else color_grey for c in classes]
bars2 = ax2.bar(classes, probs_para_multi, color=colors_para, edgecolor='black', linewidth=0.5, alpha=0.9)
ax2.set_title(f"Panel B-After: Gemma-Paraphrased Excerpt\nPredicted: {pred_para} | P(AI) = {probs_para_bin:.3f}", fontweight="bold", fontsize=11, pad=12)
ax2.set_ylabel("Attribution Probability", fontweight="bold", fontsize=10)
ax2.tick_params(axis='x', rotation=45, labelsize=9)
ax2.tick_params(axis='y', labelsize=9)
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# Function to annotate probabilities on top of the bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        if height > 0.01:  # Only annotate bars with >1% probability to keep it clean
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

autolabel(bars1, ax1)
autolabel(bars2, ax2)

# Clean up axes (remove top and right spines)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['bottom'].set_color('#cccccc')

plt.suptitle("Case Study: Model Attribution Overwriting via Paraphrasing Attacks", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/Case_Study_Attribution.png', bbox_inches='tight')
print("Saved figure to results/figures/Case_Study_Attribution.png")