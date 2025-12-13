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
from sklearn.base import clone

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
        
        # Test Evaluation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        # Handle heuristic gamma for full training if needed
        # (The wrapper handles it in fit, so we just need to ensure new instance or reset? 
        # The wrapper resets model_ in fit, so it's fine. It will re-calculate gamma on full X)
        
        model.fit(X_scaled, y)
        test_preds = model.predict(X_test_scaled)
        test_met = evaluate_model(y_test, test_preds, "Test_SVM")
        res['Test_Accuracy'] = test_met['Accuracy']
        res['Test_F1'] = test_met['F1']
        
        results.append(res)
        print(f"CV Result: {res['Accuracy']:.4f} | Test Result: {res['Test_Accuracy']:.4f}")
    
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
        
        # Test Evaluation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_scaled, y)
        test_preds = model.predict(X_test_scaled)
        test_met = evaluate_model(y_test, test_preds, "Test_MLP")
        res['Test_Accuracy'] = test_met['Accuracy']
        res['Test_F1'] = test_met['F1']

        results.append(res)
        print(f"CV Result: {res['Accuracy']:.4f} | Test Result: {res['Test_Accuracy']:.4f}")
        
    # Checkpoint
    pd.DataFrame(results).to_csv("experiment_results_partial.csv", index=False)

    print("\n=== 3. Feature Extraction + Linear SVM ===")
    # Teacher suggested: "making PCA, kPCA... work as preprocessors"
    # We compare linear SVM performance on these features.
    
    for feat_name, feat_params, clf_params in cfg.FEATURE_CONFIGS:
        if feat_name == 'raw': continue
        
        print(f"Running Feature: {feat_name} {feat_params} + SVM {clf_params}")
        
        # Instantiate model with specific C
        base_model = SVMClassifier(kernel='linear_svc', C=clf_params.get('C', 1.0))
        
        # 1. CV Evaluation
        # We must perform CV manually to include the preprocessor transform in the loop
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Standardize
            scaler_cv = StandardScaler()
            X_train_cv = scaler_cv.fit_transform(X_train_cv)
            X_val_cv = scaler_cv.transform(X_val_cv)
            
            # Handle special name mapping if needed
            real_name_cv = 'kfda' if 'kfda' in feat_name else feat_name
            
            try:
                # Preprocess
                preproc_cv = get_preprocessor(real_name_cv, **feat_params)
                X_train_trans = preproc_cv.fit_transform(X_train_cv, y_train_cv)
                X_val_trans = preproc_cv.transform(X_val_cv)
                
                # Train Model
                # Clone model to reset it
                model_cv = clone(base_model)
                model_cv.fit(X_train_trans, y_train_cv)
                
                preds_cv = model_cv.predict(X_val_trans)
                met_cv = evaluate_model(y_val_cv, preds_cv, "CV_Fold", verbose=False)
                cv_scores.append(met_cv) # evaluate_model returns dict
            except Exception as e:
                print(f"CV Error: {e}")
        
        # Aggregate Scores
        if cv_scores:
            avg_scores = pd.DataFrame(cv_scores).mean(numeric_only=True)
            res = avg_scores.to_dict()
        else:
            res = {'Accuracy': 0.0, 'F1': 0.0}

        res['Type'] = 'Feature+SVM' 
        res['Config'] = f"{feat_name} {feat_params} + {clf_params}"
        
        # 2. Test Evaluation
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            # Handle special case name 'kfda_opt' maps to 'kfda' class
            real_name = 'kfda' if 'kfda' in feat_name else feat_name
            
            # Feature Extraction
            preproc = get_preprocessor(real_name, **feat_params)
            X_trans = preproc.fit_transform(X_scaled, y)
            X_test_trans = preproc.transform(X_test_scaled)
            
            # Model
            base_model.fit(X_trans, y)
            test_preds = base_model.predict(X_test_trans)
            test_met = evaluate_model(y_test, test_preds, f"Test_Feat_{feat_name}")
            
            res['Test_Accuracy'] = test_met['Accuracy']
            res['Test_F1'] = test_met['F1']
        except Exception as e:
            print(f"Error in {feat_name}: {e}")
            res['Test_Accuracy'] = 0.0

        results.append(res)
        print(f"CV Result: {res['Accuracy']:.4f} | Test Result: {res['Test_Accuracy']:.4f}")

    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv("experiment_results.csv", index=False)
    print("\nSaved results to experiment_results.csv")
    print(df_res[['Type', 'Config', 'Accuracy', 'Test_Accuracy']])

if __name__ == "__main__":
    main()
