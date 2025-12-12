import sys
import os
import pandas as pd
import numpy as np

# Adjust path BEFORE local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments.config as cfg
from data.loader import load_data
from models.svm_models import SVMClassifier
from features.preprocessors import get_preprocessor
from utils.metrics import evaluate_model
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def main():
    print("Resuming experiments (Features)...")
    
    # Load partial results
    if os.path.exists("experiment_results_partial.csv"):
        results_df = pd.read_csv("experiment_results_partial.csv")
        results = results_df.to_dict('records')
        print(f"Loaded {len(results)} previous results.")
    else:
        results = []
        print("No partial results found. Starting fresh (Features only).")

    X, y, _, _ = load_data()
    base_model = SVMClassifier(kernel='linear_svc', C=1.0)

    print("\n=== 3. Feature Extraction + Linear SVM ===")
    
    # Reload config to ensure we get the fix
    import importlib
    importlib.reload(cfg)

    for feat_name, feat_params in cfg.FEATURE_CONFIGS:
        if feat_name == 'raw': continue
        
        print(f"Running Feature: {feat_name} {feat_params}")
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            try:
                preproc = get_preprocessor(feat_name, **feat_params)
                X_train_trans = preproc.fit_transform(X_train, y_train)
                X_val_trans = preproc.transform(X_val)
                
                base_model.fit(X_train_trans, y_train)
                preds = base_model.predict(X_val_trans)
                
                met = evaluate_model(y_val, preds, f"Feat_{feat_name}")
                fold_scores.append(met)
            except Exception as e:
                print(f"FAILED {feat_name}: {e}")
                
        if fold_scores:
            avg = pd.DataFrame(fold_scores).mean(numeric_only=True)
            res = avg.to_dict()
            res['Type'] = 'Feature+LinearSVM'
            res['Config'] = f"{feat_name} {feat_params}"
            results.append(res)
            print(f"Result: {res['Accuracy']:.4f}")

    # Save final
    df_res = pd.DataFrame(results)
    df_res.to_csv("experiment_results.csv", index=False)
    print("\nSaved final results to experiment_results.csv")
    print(df_res[['Type', 'Config', 'Accuracy']])

if __name__ == "__main__":
    main()
