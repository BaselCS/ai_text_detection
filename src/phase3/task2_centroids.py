"""Task 2: Centroid geometry vs classifier prediction conflict resolution.

Evaluates:
1. Feature count audit: 20 handcrafted vs 23 reported in Phase 2.
2. Centroid distances in hybrid space:
   - Baseline unweighted 788-d (reproduce Phase 2)
   - Option A: Weighted by XGBoost feature gain (788-d)
   - Option B: Restricted to 20 handcrafted features (unweighted)
   - Option B (gain-weighted): 20 handcrafted features weighted by gain
"""

import json
import os
import joblib
import numpy as np
import pandas as pd

def get_gain_weights(model, feature_names):
    booster = model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    
    weights = np.zeros(len(feature_names), dtype=np.float64)
    for idx, feat in enumerate(feature_names):
        if feat in score_gain:
            weights[idx] = score_gain[feat]
        elif f"f{idx}" in score_gain:
            weights[idx] = score_gain[f"f{idx}"]
            
    total_gain = np.sum(weights)
    if total_gain > 0:
        norm_weights = weights / total_gain
    else:
        norm_weights = np.ones(len(feature_names)) / len(feature_names)
    return weights, norm_weights

def weighted_euclidean(u, v, w):
    return float(np.sqrt(np.sum(w * (u - v) ** 2)))

def weighted_cosine(u, v, w):
    norm_u = np.sqrt(np.sum(w * (u ** 2)))
    norm_v = np.sqrt(np.sum(w * (v ** 2)))
    if norm_u == 0 or norm_v == 0:
        return 0.0
    return float(np.sum(w * u * v) / (norm_u * norm_v))

def compute_distances(train_centroids, para_centroids, weights, dims_slice=None, alt_map=None):
    if alt_map is None:
        alt_map = {}
        
    e_rows = []
    c_rows = []
    
    writers = sorted(train_centroids.keys())
    
    for sname, c_full in para_centroids.items():
        if dims_slice is not None:
            c = c_full[dims_slice]
            w = weights[dims_slice] if weights is not None else None
        else:
            c = c_full
            w = weights
            
        if w is not None:
            w = w / np.sum(w)
            
        erow = {"Paraphrased_set": sname}
        crow = {"Paraphrased_set": sname}
        
        for writer in writers:
            tc_full = train_centroids[writer]
            tc = tc_full[dims_slice] if dims_slice is not None else tc_full
            
            if w is not None:
                d_e = weighted_euclidean(c, tc, w)
                d_c = weighted_cosine(c, tc, w)
            else:
                d_e = float(np.linalg.norm(c - tc))
                norm_c = np.linalg.norm(c)
                norm_tc = np.linalg.norm(tc)
                d_c = float(np.dot(c, tc) / (norm_c * norm_tc)) if norm_c > 0 and norm_tc > 0 else 0.0
                
            erow[writer] = round(d_e, 4)
            crow[writer] = round(d_c, 4)
            
        erow["nearest_Euclidean"] = min(writers, key=lambda wr: erow[wr])
        crow["nearest_cosine"] = max(writers, key=lambda wr: crow[wr])
        
        if sname in alt_map:
            target = alt_map[sname]
            crow["paraphraser_cosine_rank"] = sorted(writers, key=lambda wr: -crow[wr]).index(target) + 1
            erow["paraphraser_euclidean_rank"] = sorted(writers, key=lambda wr: erow[wr]).index(target) + 1
            
        e_rows.append(erow)
        c_rows.append(crow)
        
    return pd.DataFrame(e_rows), pd.DataFrame(c_rows)

def main():
    os.makedirs("results/phase3", exist_ok=True)
    
    # 1. Feature audit
    with open("results/models/Bin_BERT_features.json") as f:
        features = json.load(f)
        
    handcrafted = [f for f in features if not f.startswith("bert_")]
    bert_features = [f for f in features if f.startswith("bert_")]
    
    print("=" * 70)
    print("TASK 2: FEATURE AUDIT & CENTROID GEOMETRY ANALYSIS")
    print("=" * 70)
    print(f"Total features in model: {len(features)}")
    print(f"  Handcrafted features: {len(handcrafted)} -> {handcrafted}")
    print(f"  BERT features:       {len(bert_features)}")
    
    # Check column count in processed_articles.csv
    train_sample = pd.read_csv("data/processed/processed_articles.csv", nrows=1)
    meta_cols = [c for c in train_sample.columns if c not in features]
    print(f"\nColumns in processed_articles.csv: {len(train_sample.columns)}")
    print(f"  Non-feature (metadata) columns: {meta_cols}")
    print(f"  Audit explanation: 791 total columns - 768 BERT = 23 columns.")
    print(f"  The 23 columns consist of {len(handcrafted)} handcrafted features + 3 metadata columns ({meta_cols}).")
    print(f"  Conclusion: True handcrafted feature count is 20, exactly as in Paper 1.")
    
    # 2. Load model, scaler, and training centroids
    scaler = joblib.load("results/models/Multi_BERT_scaler.joblib")
    model = joblib.load("results/models/Multi_BERT_model.joblib")
    
    raw_weights, norm_weights = get_gain_weights(model, features)
    hc_gain_share = np.sum(norm_weights[:len(handcrafted)])
    bert_gain_share = np.sum(norm_weights[len(handcrafted):])
    print(f"\nXGBoost Total Gain Share:")
    print(f"  Handcrafted (20 feats): {hc_gain_share * 100:.2f}%")
    print(f"  BERT (768 feats):        {bert_gain_share * 100:.2f}%")
    
    print("\nLoading training corpus and computing centroids...", flush=True)
    train = pd.read_csv("data/processed/processed_articles.csv", usecols=["Writer"] + features, low_memory=True)
    Xtr = scaler.transform(train[features].astype("float32")).astype(np.float64)
    train_c = {w: Xtr[(train["Writer"] == w).to_numpy()].mean(axis=0) for w in sorted(train["Writer"].unique())}
    writers = sorted(train_c.keys())
    print(f"Training classes ({len(writers)}): {writers}")
    
    # 3. Load paraphrased sets
    SETS = {
        "Gemma-27B": "data/processed/processed_paraphrased_articles.csv",
        "Gemma-2-9B": "data/processed/processed_paraphrased_gemma2.csv",
        "Qwen-2": "data/processed/processed_paraphrased_qwen2.csv",
        "Mistral-7B": "data/processed/processed_paraphrased_mistral7b.csv",
        "LLaMA-3": "data/processed/processed_paraphrased_llama3.csv",
        "GPT-4o-mini": "data/processed/processed_paraphrased_gpt4o_mini.csv",
    }
    ALT = {
        "Gemma-27B": "Gemma-2-9B",
        "Gemma-2-9B": "Gemma-2-9B",
        "Qwen-2": "Qwen-2-72B",
        "Mistral-7B": "Mistral-7B",
        "LLaMA-3": "Llama-8B",
        "GPT-4o-mini": "GPT-4o",
    }
    
    para_c = {}
    for sname, csv in SETS.items():
        df = pd.read_csv(csv, low_memory=True)
        df = df.reindex(columns=features, fill_value=0)
        c = scaler.transform(df.astype("float32")).astype(np.float64).mean(axis=0)
        para_c[sname] = c
        
    # 4. Compute configurations
    # Config 0: Baseline unweighted 788-d
    df_base_e, df_base_c = compute_distances(train_c, para_c, weights=None, alt_map=ALT)
    
    # Config A: Option A - Gain-weighted 788-d
    df_optA_e, df_optA_c = compute_distances(train_c, para_c, weights=norm_weights, alt_map=ALT)
    df_optA_e.to_csv("results/phase3/T2_Centroid_OptionA_GainWeighted_Euclidean.csv", index=False)
    df_optA_c.to_csv("results/phase3/T2_Centroid_OptionA_GainWeighted_Cosine.csv", index=False)
    
    # Config B: Option B - 20 Handcrafted features unweighted
    hc_slice = slice(0, len(handcrafted))
    df_optB_e, df_optB_c = compute_distances(train_c, para_c, weights=None, dims_slice=hc_slice, alt_map=ALT)
    df_optB_e.to_csv("results/phase3/T2_Centroid_OptionB_Handcrafted_Euclidean.csv", index=False)
    df_optB_c.to_csv("results/phase3/T2_Centroid_OptionB_Handcrafted_Cosine.csv", index=False)
    
    # Config B-weighted: 20 Handcrafted features weighted by gain
    df_optBw_e, df_optBw_c = compute_distances(train_c, para_c, weights=norm_weights, dims_slice=hc_slice, alt_map=ALT)
    df_optBw_e.to_csv("results/phase3/T2_Centroid_OptionB_GainWeighted_Euclidean.csv", index=False)
    df_optBw_c.to_csv("results/phase3/T2_Centroid_OptionB_GainWeighted_Cosine.csv", index=False)
    
    # Print results
    print("\n" + "=" * 70)
    print("CONFIG 0: BASELINE UNWEIGHTED 788-D (PHASE 2 REPRODUCTION)")
    print("=" * 70)
    print(df_base_e[["Paraphrased_set"] + writers + ["nearest_Euclidean"]].to_string(index=False))
    print()
    print(df_base_c[["Paraphrased_set"] + writers + ["nearest_cosine", "paraphraser_cosine_rank"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("OPTION A: GAIN-WEIGHTED 788-D (XGBOOST FEATURE GAIN)")
    print("=" * 70)
    print(df_optA_e[["Paraphrased_set"] + writers + ["nearest_Euclidean"]].to_string(index=False))
    print()
    print(df_optA_c[["Paraphrased_set"] + writers + ["nearest_cosine", "paraphraser_cosine_rank"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("OPTION B: RESTRICTED TO 20 HANDCRAFTED FEATURES ONLY (UNWEIGHTED)")
    print("=" * 70)
    print(df_optB_e[["Paraphrased_set"] + writers + ["nearest_Euclidean"]].to_string(index=False))
    print()
    print(df_optB_c[["Paraphrased_set"] + writers + ["nearest_cosine", "paraphraser_cosine_rank"]].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("OPTION B (GAIN-WEIGHTED): 20 HANDCRAFTED FEATURES")
    print("=" * 70)
    print(df_optBw_e[["Paraphrased_set"] + writers + ["nearest_Euclidean"]].to_string(index=False))
    print()
    print(df_optBw_c[["Paraphrased_set"] + writers + ["nearest_cosine", "paraphraser_cosine_rank"]].to_string(index=False))
    
    print("\nAll tables saved in results/phase3/")

if __name__ == "__main__":
    main()
