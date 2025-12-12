import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_data
from models.svm_models import SVMClassifier
from models.neural_net import PyTorchMLP
from features.preprocessors import get_preprocessor
from utils.metrics import evaluate_model
from experiments.heuristics import estimate_gamma
import experiments.config as cfg

def run_cv(X, y, model, cv=3):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize inside CV
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        met = evaluate_model(y_val, preds, "CV_Fold")
        scores.append(met)
        
    # Average
    avg_scores = pd.DataFrame(scores).mean(numeric_only=True)
    return avg_scores

def main():
    print("Loading Data...")
    X, y, X_test, y_test = load_data()
    
    results = []

    print("\n=== 1. SVM Benchmarks ===")
    for conf in cfg.SVM_CONFIGS:
        print(f"Running SVM: {conf}")
        model = SVMClassifier(**conf)
        scores = run_cv(X, y, model, cv=3)
        res = scores.to_dict()
        res['Type'] = 'SVM'
        res['Config'] = str(conf)
        results.append(res)
        print(f"Result: {res['Accuracy']:.4f}")
    
    # Checkpoint
    pd.DataFrame(results).to_csv("experiment_results_partial.csv", index=False)

    print("\n=== 2. MLP Benchmarks ===")
    for conf in cfg.MLP_CONFIGS:
        print(f"Running MLP: {conf}")
        model = PyTorchMLP(**conf, verbose=False)
        scores = run_cv(X, y, model, cv=3)
        res = scores.to_dict()
        res['Type'] = 'MLP'
        res['Config'] = str(conf)
        results.append(res)
        print(f"Result: {res['Accuracy']:.4f}")
        
    # Checkpoint
    pd.DataFrame(results).to_csv("experiment_results_partial.csv", index=False)

    print("\n=== 3. Feature Extraction + Linear SVM ===")
    # Teacher suggested: "making PCA, kPCA... work as preprocessors"
    # We compare linear SVM performance on these features.
    
    base_model = SVMClassifier(kernel='linear_svc', C=1.0)
    
    for feat_name, feat_params in cfg.FEATURE_CONFIGS:
        if feat_name == 'raw': continue
        
        print(f"Running Feature: {feat_name} {feat_params}")
        
        # We need a custom pipeline runner for this part to do Feature -> CV -> Model
        # Actually I can wrap it in a Pipeline class, but I'm doing manual loop for better control
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Standardize FIRST
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Feature Extraction
            preproc = get_preprocessor(feat_name, **feat_params)
            X_train_trans = preproc.fit_transform(X_train, y_train)
            X_val_trans = preproc.transform(X_val)
            
            # Model
            base_model.fit(X_train_trans, y_train)
            preds = base_model.predict(X_val_trans)
            
            met = evaluate_model(y_val, preds, f"Feat_{feat_name}")
            fold_scores.append(met)
            
        avg = pd.DataFrame(fold_scores).mean(numeric_only=True)
        res = avg.to_dict()
        res['Type'] = 'Feature+LinearSVM'
        res['Config'] = f"{feat_name} {feat_params}"
        results.append(res)
        print(f"Result: {res['Accuracy']:.4f}")

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("experiment_results.csv", index=False)
    print("\nSaved results to experiment_results.csv")
    print(df_res[['Type', 'Config', 'Accuracy']])

if __name__ == "__main__":
    main()
