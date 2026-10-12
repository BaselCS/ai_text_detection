import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

warnings.filterwarnings('ignore')

SEED = 999
MAX_WORKERS = -1 

DEVICE = "cuda"

print("Loading Data...")
df = pd.read_csv('data/processed/processed_articles.csv')
df = df.drop(columns=['Text'], errors='ignore')

bert_cols = [c for c in df.columns if c.startswith('bert_')]
X_all = df.drop(columns=['is_AI', 'Writer'])
X_no_bert = X_all.drop(columns=bert_cols)

# Encoders
le_multi = LabelEncoder()
y_multi = le_multi.fit_transform(df['Writer'])
multi_classes = list(le_multi.classes_)

print("Running Multi-Class Statistical Significance Evaluation (5 Seeds Sequentially)...")

seeds_list = [999 ,42, 123, 2024, 8080]

detailed_data_multi = []
stats_results_multi = []


per_class_f1_nobert = {cls: [] for cls in multi_classes}
per_class_f1_bert = {cls: [] for cls in multi_classes}


X_tr_m, X_te_m, y_tr_m, y_te_m = train_test_split(
    X_all, y_multi, test_size=0.20, random_state=SEED, stratify=y_multi
)

# Standardization
sc_m = StandardScaler()
X_tr_sc_m = pd.DataFrame(sc_m.fit_transform(X_tr_m).astype('float32'), columns=X_all.columns)
X_te_sc_m = pd.DataFrame(sc_m.transform(X_te_m).astype('float32'), columns=X_all.columns)

# Create No-BERT feature sets
X_tr_nobert_m = X_tr_sc_m.drop(columns=bert_cols)
X_te_nobert_m = X_te_sc_m.drop(columns=bert_cols)

for s in seeds_list:
    print(f" -> Started Training Multi-Class Seed {s}...")
    
    # Model 1: Features Only (No BERT)
    m_no_multi = xgb.XGBClassifier(
        objective='multi:softprob', 
        tree_method='hist', 
        device=DEVICE, 
        n_estimators=2000, 
        early_stopping_rounds=100, 
        learning_rate=0.01, 
        max_depth=10, 
        random_state=s, 
        n_jobs=MAX_WORKERS,
        subsample=0.8,
        colsample_bytree=0.8
    )
    m_no_multi.fit(X_tr_nobert_m, y_tr_m, eval_set=[(X_te_nobert_m, y_te_m)], verbose=False)
    
    joblib.dump(m_no_multi, f'results/models/Multi_NoBERT_seed_{s}_model.joblib')
    if s == 999:
        joblib.dump(m_no_multi, 'results/models/Multi_NoBERT_model.joblib')
        
    p_no_multi = m_no_multi.predict(X_te_nobert_m)
    
    acc_no_multi = accuracy_score(y_te_m, p_no_multi)
    f1_no_multi = precision_recall_fscore_support(y_te_m, p_no_multi, average='macro', zero_division=0)[2]
    f1_no_multi_per_class = precision_recall_fscore_support(y_te_m, p_no_multi, average=None, zero_division=0)[2]
    
    # Model 2: Features + BERT
    m_b_multi = xgb.XGBClassifier(
        objective='multi:softprob', 
        tree_method='hist', 
        device=DEVICE, 
        n_estimators=2000, 
        early_stopping_rounds=100, 
        learning_rate=0.01, 
        max_depth=10, 
        random_state=s, 
        n_jobs=MAX_WORKERS,
        subsample=0.8,
        colsample_bytree=0.8
    )
    m_b_multi.fit(X_tr_sc_m, y_tr_m, eval_set=[(X_te_sc_m, y_te_m)], verbose=False)
    
    joblib.dump(m_b_multi, f'results/models/Multi_BERT_seed_{s}_model.joblib')
    if s == 999:
        joblib.dump(m_b_multi, 'results/models/Multi_BERT_model.joblib')
        
    p_b_multi = m_b_multi.predict(X_te_sc_m)
    
    acc_b_multi = accuracy_score(y_te_m, p_b_multi)
    f1_b_multi = precision_recall_fscore_support(y_te_m, p_b_multi, average='macro', zero_division=0)[2]
    f1_b_multi_per_class = precision_recall_fscore_support(y_te_m, p_b_multi, average=None, zero_division=0)[2]

    print(f" <- Finished Training Multi-Class Seed {s}!")

    stats_results_multi.append({
        'Seed': s, 'NoBERT_Acc': acc_no_multi, 'NoBERT_F1': f1_no_multi, 
        'BERT_Acc': acc_b_multi, 'BERT_F1': f1_b_multi
    })
    
    for i, cls in enumerate(multi_classes):
        per_class_f1_nobert[cls].append(f1_no_multi_per_class[i])
        per_class_f1_bert[cls].append(f1_b_multi_per_class[i])
    
    detailed_data_multi.append({'Experiment': f"Seed {s}", 'Model': 'Features Only', 'Accuracy': f"{acc_no_multi:.4f}", 'F1-Score': f"{f1_no_multi:.4f}"})
    detailed_data_multi.append({'Experiment': f"Seed {s}", 'Model': 'Features + BERT', 'Accuracy': f"{acc_b_multi:.4f}", 'F1-Score': f"{f1_b_multi:.4f}"})

# Calculate Final Statistics
detailed_data_multi.append({'Experiment': '-'*15, 'Model': '-'*15, 'Accuracy': '-'*15, 'F1-Score': '-'*15})

stats_df_multi = pd.DataFrame(stats_results_multi)
mean_df_multi = stats_df_multi.mean()
std_df_multi = stats_df_multi.std()

detailed_data_multi.append({
    'Experiment': 'FINAL SUMMARY', 'Model': 'Features Only',
    'Accuracy': f"{mean_df_multi['NoBERT_Acc']:.4f} ± {std_df_multi['NoBERT_Acc']:.4f}",
    'F1-Score': f"{mean_df_multi['NoBERT_F1']:.4f} ± {std_df_multi['NoBERT_F1']:.4f}"
})
detailed_data_multi.append({
    'Experiment': 'FINAL SUMMARY', 'Model': 'Features + BERT',
    'Accuracy': f"{mean_df_multi['BERT_Acc']:.4f} ± {std_df_multi['BERT_Acc']:.4f}",
    'F1-Score': f"{mean_df_multi['BERT_F1']:.4f} ± {std_df_multi['BERT_F1']:.4f}"
})

output_path = 'results/tables/T11_Statistical_Significance_Multi_Detailed.csv'
pd.DataFrame(detailed_data_multi).to_csv(output_path, index=False)

# Build and Save Per-Class Summary
per_class_summary = []
for cls in multi_classes:
    nb_mean, nb_std = np.mean(per_class_f1_nobert[cls]), np.std(per_class_f1_nobert[cls], ddof=1)
    b_mean, b_std = np.mean(per_class_f1_bert[cls]), np.std(per_class_f1_bert[cls], ddof=1)
    per_class_summary.append({
        'Class': cls,
        'Features Only (Mean ± SD)': f"{nb_mean:.4f} ± {nb_std:.4f}",
        'Features + BERT (Mean ± SD)': f"{b_mean:.4f} ± {b_std:.4f}"
    })

per_class_df = pd.DataFrame(per_class_summary)
per_class_path = 'results/tables/T11_Per_Class_5Seeds_Summary.csv'
per_class_df.to_csv(per_class_path, index=False)

print("="*60)
print(f"✓ Overall Multi-Class Statistical Significance Test saved to: \n  {output_path}")
print(f"✓ Per-Class Mean ± SD (across 5 seeds) saved to: \n  {per_class_path}")
print("="*60)
print(per_class_df.to_string(index=False))