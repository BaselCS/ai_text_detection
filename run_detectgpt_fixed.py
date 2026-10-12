# ==========================================
# EXTERNAL BASELINE EVALUATION (DetectGPT)
# FIXED VERSION v4 - All bugs resolved
# ==========================================
# FIX 1: mask_filler returns list-of-lists for multiple <mask> tokens.
# FIX 2: truncation=True kwarg not supported by this transformers version.
# FIX 3: Token-level truncation (word count is unreliable proxy).
# FIX 4: Loop-guaranteed truncation + CUDA error recovery.
#
# Results saved to: results/tables/T11_DetectGPT_Baseline_Results_FIXED.csv
# (Does NOT overwrite original T11 or T12 files)
# ==========================================

import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import random
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("STARTING EXTERNAL BASELINE EVALUATION (DetectGPT - FIXED)")
print("=" * 60)

SEED = 999
NUM_VARIATIONS = 100
WINDOW_SIZE = 150

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv('data/processed/processed_articles.csv')

_, df_test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['is_AI'])
df_human = df_test[df_test['is_AI'] == 0]
df_ai = df_test[df_test['is_AI'] == 1]
n_samp = min(len(df_human), len(df_ai))

df_sampled = pd.concat([
    df_human.sample(n_samp, random_state=SEED),
    df_ai.sample(n_samp, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Loaded {len(df_sampled)} balanced samples.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Models on {device}...")

tokenizer_gpt = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token
model_gpt = GPT2LMHeadModel.from_pretrained("gpt2-medium").to(device)
model_gpt.eval()

mask_filler = pipeline(
    "fill-mask",
    model="distilroberta-base",
    device=0 if device == "cuda" else -1,
    top_k=1
)


def get_log_likelihood(text):
    try:
        inputs = tokenizer_gpt(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        if inputs.input_ids.size(1) < 2:
            return 0.0
        with torch.no_grad():
            outputs = model_gpt(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()
    except RuntimeError as e:
        if "CUDA" in str(e):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return 0.0


def perturb_text_safe(text, mask_prob=0.15):
    """
    FIXED VERSION: Properly handles mask_filler output format.

    When mask_filler receives text with multiple <mask> tokens, it returns
    a list of lists: [[{prediction_for_mask_1}], [{prediction_for_mask_2}], ...].
    When there's only one <mask>, it returns a flat list: [{prediction}].

    The original code did:
        for r in res_list:
            masked_text = masked_text.replace("<mask>", r['token_str'], 1)
    This fails when r is a list (multiple masks case), causing TypeError
    silently caught by `except: return text`.
    """
    words = text.split()
    if len(words) < 10:
        return text

    num_masks = max(1, int(len(words) * mask_prob))
    mask_indices = random.sample(range(len(words)), min(num_masks, 15))

    for idx in mask_indices:
        words[idx] = "<mask>"

    masked_text = " ".join(words)

    try:
        # FIX 2+3+4: Loop-guaranteed token truncation.
        # Keep removing words until we are strictly under 510 tokens.
        words_masked = masked_text.split()
        for _ in range(10):  # max 10 shrink iterations
            token_count = len(mask_filler.tokenizer.encode(masked_text, add_special_tokens=True))
            if token_count <= 510:
                break
            # Shrink proportionally with safety margin
            target_words = max(10, int(len(words_masked) * 490 / token_count))
            words_masked = words_masked[:target_words]
            masked_text = " ".join(words_masked)

        # If still too long after loop, skip this perturbation
        final_count = len(mask_filler.tokenizer.encode(masked_text, add_special_tokens=True))
        if final_count > 512:
            return text

        # Check if any <mask> tokens remain after truncation
        if "<mask>" not in masked_text:
            return text

        results = mask_filler(masked_text)

        # FIX 1: Normalize the output format
        # Single mask  -> results = [{'token_str': ...}, ...]  (flat list of dicts)
        # Multiple masks -> results = [[{'token_str': ...}], [{'token_str': ...}], ...]
        if results and isinstance(results[0], list):
            for sub_list in results:
                if sub_list and isinstance(sub_list, list):
                    token = sub_list[0]['token_str']
                else:
                    token = sub_list['token_str']
                masked_text = masked_text.replace("<mask>", token, 1)
        else:
            if results:
                token = results[0]['token_str']
                masked_text = masked_text.replace("<mask>", token, 1)

        return masked_text.replace("<mask>", "")
    except RuntimeError as e:
        if "CUDA" in str(e):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return text
    except Exception as e:
        return text


def process_full_text_sliding_window(text):
    words = text.split()
    chunks = [" ".join(words[i:i + WINDOW_SIZE]) for i in range(0, len(words), WINDOW_SIZE)]

    chunk_orig_ll = []
    chunk_pert_ll = []

    for chunk in chunks:
        if len(chunk) > 3000:
            chunk = chunk[:3000]

        if len(chunk.split()) < 10:
            continue

        orig_ll = get_log_likelihood(chunk)
        chunk_orig_ll.append(orig_ll)

        p_lls = []
        for _ in range(NUM_VARIATIONS):
            p_text = perturb_text_safe(chunk)
            p_lls.append(get_log_likelihood(p_text))

        chunk_pert_ll.append(np.mean(p_lls) if p_lls else orig_ll)

    if not chunk_orig_ll:
        return None

    avg_orig_ll = np.mean(chunk_orig_ll)
    avg_pert_ll = np.mean(chunk_pert_ll)
    discrepancy = avg_orig_ll - avg_pert_ll

    return avg_orig_ll, avg_pert_ll, discrepancy


print("\nProcessing Samples (Fixed Perturbations)...")
features_list = []
valid_y_true = []

cuda_errors = 0
for idx, text in enumerate(tqdm(df_sampled['Text'].astype(str).tolist(), desc="Evaluating DetectGPT (FIXED)")):
    try:
        result = process_full_text_sliding_window(text)
        if result is not None:
            features_list.append(result)
            valid_y_true.append(df_sampled['is_AI'].iloc[idx])
    except RuntimeError as e:
        if "CUDA" in str(e):
            cuda_errors += 1
            # Recover GPU state so next samples don't cascade-fail
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            if cuda_errors <= 3:
                print(f"  [CUDA RECOVERY] Sample {idx}: GPU state reset. ({e})")
        else:
            print(f"  [ERROR] Sample {idx}: {e}")
        continue
    except Exception as e:
        print(f"  [ERROR] Sample {idx}: {e}")
        continue

y_true = np.array(valid_y_true)
features_array = np.array(features_list)

print(f"\nSuccessfully processed {len(valid_y_true)} out of {len(df_sampled)} samples.")

print("\n" + "=" * 60)
print("COMPUTING METRICS (ZERO-SHOT vs SUPERVISED) - FIXED")
print("=" * 60)

# Zero-Shot: threshold on discrepancy score
detectgpt_scores = features_array[:, 2]
threshold = np.median(detectgpt_scores)
y_pred_zero = (detectgpt_scores >= threshold).astype(int)
acc_zero = accuracy_score(y_true, y_pred_zero)
prec_zero, rec_zero, f1_zero, _ = precision_recall_fscore_support(y_true, y_pred_zero, average='macro', zero_division=0)

# Supervised: Logistic Regression on all 3 features
clf = LogisticRegression(random_state=SEED)
y_pred_sup = cross_val_predict(clf, features_array, y_true, cv=5)
acc_sup = accuracy_score(y_true, y_pred_sup)
prec_sup, rec_sup, f1_sup, _ = precision_recall_fscore_support(y_true, y_pred_sup, average='macro', zero_division=0)

results = [
    {'Model': 'DetectGPT (Zero-Shot Curvature)', 'Accuracy': acc_zero, 'Precision': prec_zero, 'Recall': rec_zero, 'F1-Score': f1_zero},
    {'Model': 'DetectGPT (Logistic Regression Base)', 'Accuracy': acc_sup, 'Precision': prec_sup, 'Recall': rec_sup, 'F1-Score': f1_sup}
]

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

# Save to NEW file - do NOT overwrite original T11 or T12
output_path = 'results/tables/T11_DetectGPT_Baseline_Results_FIXED.csv'
df_results.to_csv(output_path, index=False)
print("=" * 60)
print(f"✓ Results saved successfully to {output_path}")
print("  (Original T11 and T12 files are preserved)")
