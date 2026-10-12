# ==========================================
# DetectGPT QUICK VERIFICATION TEST
# NUM_VARIATIONS=3, 200 samples only
# Purpose: Verify the perturb_text_safe fix works
#          before running the full 12-hour evaluation.
# ==========================================

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
print("DetectGPT QUICK TEST (verify fix works)")
print("NUM_VARIATIONS=3, 200 samples, < 30 min expected")
print("=" * 60)

SEED = 999
NUM_VARIATIONS = 3    # Quick test: normally 100
N_SAMPLES = 200       # Quick test: normally all samples
WINDOW_SIZE = 150

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df = pd.read_csv('data/processed/processed_articles.csv')

_, df_test = train_test_split(df, test_size=0.20, random_state=SEED, stratify=df['is_AI'])
df_human = df_test[df_test['is_AI'] == 0]
df_ai    = df_test[df_test['is_AI'] == 1]

n_samp = min(len(df_human), len(df_ai), N_SAMPLES // 2)

df_sampled = pd.concat([
    df_human.sample(n_samp, random_state=SEED),
    df_ai.sample(n_samp, random_state=SEED)
]).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"Using {len(df_sampled)} balanced samples for quick test.")

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
    inputs = tokenizer_gpt(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    if inputs.input_ids.size(1) < 2:
        return 0.0
    with torch.no_grad():
        outputs = model_gpt(**inputs, labels=inputs.input_ids)
        return -outputs.loss.item()


perturbation_errors = 0
perturbation_success = 0

def perturb_text_safe(text, mask_prob=0.15):
    """
    FIXED VERSION: Properly handles mask_filler output format.

    When mask_filler receives text with multiple <mask> tokens, it returns
    a list of lists: [[{prediction_for_mask_1}], [{prediction_for_mask_2}], ...].
    When there is only one <mask>, it returns a flat list: [{prediction}].

    The original code did:
        for r in res_list:
            masked_text = masked_text.replace("<mask>", r['token_str'], 1)
    This fails when r is a list (multiple masks case), causing TypeError
    silently caught by `except: return text`.
    """
    global perturbation_errors, perturbation_success

    words = text.split()
    if len(words) < 10:
        return text

    num_masks = max(1, int(len(words) * mask_prob))
    mask_indices = random.sample(range(len(words)), min(num_masks, 15))

    for idx in mask_indices:
        words[idx] = "<mask>"

    masked_text = " ".join(words)

    try:
        # FIX 2+3: truncation kwarg not supported AND word-count truncation
        # is unreliable (150 words can be 5000+ tokens for URLs/numbers).
        # Use the actual tokenizer to ensure we stay under 512 token limit.
        token_count = len(mask_filler.tokenizer.encode(masked_text, add_special_tokens=True))
        if token_count > 510:
            words_masked = masked_text.split()
            target_words = max(10, int(len(words_masked) * 500 / token_count))
            masked_text = " ".join(words_masked[:target_words])

        results = mask_filler(masked_text)

        # FIX 1: Normalize the output format
        # Single mask  -> results = [{'token_str': ...}, ...]  (flat list of dicts)
        # Multiple masks -> results = [[{'token_str': ...}], [{'token_str': ...}], ...]
        if results and isinstance(results[0], list):
            # Multiple masks: extract top prediction from each sub-list
            for sub_list in results:
                if sub_list and isinstance(sub_list, list):
                    token = sub_list[0]['token_str']
                else:
                    token = sub_list['token_str']
                masked_text = masked_text.replace("<mask>", token, 1)
        else:
            # Single mask: flat list of predictions (top_k results for 1 mask)
            if results:
                token = results[0]['token_str']
                masked_text = masked_text.replace("<mask>", token, 1)

        # Clean up any remaining <mask> tokens
        perturbation_success += 1
        return masked_text.replace("<mask>", "")

    except Exception as e:
        perturbation_errors += 1
        print(f"  [WARNING] Perturbation failed: {type(e).__name__}: {e}")
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


print("\nProcessing Samples (Quick Test - Fixed Perturbations)...")
features_list = []
valid_y_true = []

for idx, text in enumerate(tqdm(df_sampled['Text'].astype(str).tolist(), desc="Evaluating DetectGPT (QUICK)")):
    try:
        result = process_full_text_sliding_window(text)
        if result is not None:
            features_list.append(result)
            valid_y_true.append(df_sampled['is_AI'].iloc[idx])
    except Exception as e:
        print(f"  [ERROR] Sample {idx} failed: {e}")
        continue

y_true = np.array(valid_y_true)
features_array = np.array(features_list)

print(f"\nProcessed {len(valid_y_true)} / {len(df_sampled)} samples.")
print(f"Perturbation success: {perturbation_success}, errors: {perturbation_errors}")

if perturbation_errors == 0:
    print("✅ ZERO perturbation errors — fix is working correctly!")
else:
    pct = 100.0 * perturbation_errors / max(1, perturbation_success + perturbation_errors)
    print(f"⚠️  {pct:.1f}% perturbations still failing — check warnings above.")

print("\n" + "=" * 60)
print("COMPUTING METRICS (ZERO-SHOT vs SUPERVISED) — QUICK TEST")
print("=" * 60)

detectgpt_scores = features_array[:, 2]
print(f"\nDiscrepancy score stats:")
print(f"  min={detectgpt_scores.min():.4f}, max={detectgpt_scores.max():.4f}")
print(f"  mean={detectgpt_scores.mean():.4f}, std={detectgpt_scores.std():.4f}")

threshold = np.median(detectgpt_scores)
y_pred_zero = (detectgpt_scores >= threshold).astype(int)
acc_zero = accuracy_score(y_true, y_pred_zero)
prec_zero, rec_zero, f1_zero, _ = precision_recall_fscore_support(
    y_true, y_pred_zero, average='macro', zero_division=0)

clf = LogisticRegression(random_state=SEED)
y_pred_sup = cross_val_predict(clf, features_array, y_true, cv=5)
acc_sup = accuracy_score(y_true, y_pred_sup)
prec_sup, rec_sup, f1_sup, _ = precision_recall_fscore_support(
    y_true, y_pred_sup, average='macro', zero_division=0)

results = [
    {
        'Model': 'DetectGPT (Zero-Shot Curvature) [QUICK]',
        'Accuracy': round(acc_zero, 4),
        'Precision': round(prec_zero, 4),
        'Recall': round(rec_zero, 4),
        'F1-Score': round(f1_zero, 4)
    },
    {
        'Model': 'DetectGPT (Logistic Regression Base) [QUICK]',
        'Accuracy': round(acc_sup, 4),
        'Precision': round(prec_sup, 4),
        'Recall': round(rec_sup, 4),
        'F1-Score': round(f1_sup, 4)
    }
]

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))

output_path = 'results/tables/T11_DetectGPT_QuickTest.csv'
df_results.to_csv(output_path, index=False)

print("\n" + "=" * 60)
if acc_zero > 0.55:
    print(f"✅ FIX CONFIRMED: Zero-Shot Accuracy = {acc_zero:.4f} (was 0.50 before fix)")
    print("   Ready to run full evaluation with run_detectgpt_fixed.py")
else:
    print(f"⚠️  Zero-Shot Accuracy = {acc_zero:.4f} — still near random chance!")
    print("   Check perturbation logs above for remaining issues.")
print("=" * 60)
print(f"✓ Quick test results saved to {output_path}")
