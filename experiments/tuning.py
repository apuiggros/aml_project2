import sys
import os
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import load_data
from models.svm_models import SVMClassifier
from models.neural_net import PyTorchMLP
from features.preprocessors import get_preprocessor

# Load Data Once
print("Loading Data...")
X, y, X_test, y_test = load_data()

# Standardize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
X_test_norm = scaler.transform(X_test)

def tune_svm():
    print("\n--- Tuning SVM (RBF) ---")
    
    space = [
        Real(1e-2, 1e2, name='C', prior='log-uniform'),
        Real(1e-4, 1e1, name='gamma', prior='log-uniform')
    ]
    
    @use_named_args(space)
    def objective(**params):
        model = SVMClassifier(kernel='rbf', **params)
        # 3-fold CV
        score = cross_val_score(model, X_norm, y, cv=3, n_jobs=-1, scoring='accuracy')
        return -np.mean(score) # Minimize negative accuracy

    res = gp_minimize(objective, space, n_calls=20, random_state=42)
    print(f"Best Accuracy: {-res.fun:.4f}")
    print(f"Best Params: C={res.x[0]:.4f}, gamma={res.x[1]:.4f}")
    return res.x

def tune_kfda_svm():
    print("\n--- Tuning kFDA + Linear SVM ---")
    
    # Tuning kFDA gamma and SVM C
    space = [
        Real(1e-3, 1e1, name='kfda_gamma', prior='log-uniform'),
        Real(1e-2, 1e2, name='svm_C', prior='log-uniform'),
        Integer(2, 5, name='n_components') # Max is 5 (classes-1)
    ]
    
    @use_named_args(space)
    def objective(kfda_gamma, svm_C, n_components):
        # We must build the pipeline manually inside since we have custom steps
        # CV loop
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in skf.split(X_norm, y):
            X_train, X_val = X_norm[train_idx], X_norm[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Feature Extraction
                # Note: kFDA is expensive to fit. 
                kfda = get_preprocessor('kfda', n_components=int(n_components), kernel='rbf', gamma=kfda_gamma)
                X_train_trans = kfda.fit_transform(X_train, y_train)
                X_val_trans = kfda.transform(X_val)
                
                # SVM
                clf = SVMClassifier(kernel='linear_svc', C=svm_C)
                clf.fit(X_train_trans, y_train)
                preds = clf.predict(X_val_trans)
                scores.append(accuracy_score(y_val, preds))
            except Exception as e:
                return 1.0 # Penalize failure

        return -np.mean(scores)

    res = gp_minimize(objective, space, n_calls=20, random_state=42)
    print(f"Best Accuracy: {-res.fun:.4f}")
    print(f"Best Params: kfda_gamma={res.x[0]:.4f}, svm_C={res.x[1]:.4f}, n_components={res.x[2]}")
    return res.x

def tune_mlp():
    print("\n--- Tuning MLP ---")
    
    space = [
        Real(1e-4, 1e-2, name='lr', prior='log-uniform'),
        Real(0.0, 0.5, name='dropout'),
        Integer(50, 200, name='hidden_size')
    ]
    
    @use_named_args(space)
    def objective(lr, dropout, hidden_size):
        # Fixed depth for simplicity, tuning width
        model = PyTorchMLP(hidden_layers=[int(hidden_size), int(hidden_size)//2], 
                           dropout=dropout, lr=lr, epochs=30, verbose=False) # Reduced epochs for speed
        
        # We can't use cross_val_score easily with our wrapper if it doesn't support cloning perfectly or n_jobs
        # Let's use manual CV
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X_norm, y):
            X_train, X_val = X_norm[train_idx], X_norm[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores.append(accuracy_score(y_val, preds))
            
        return -np.mean(scores)

    res = gp_minimize(objective, space, n_calls=15, random_state=42)
    print(f"Best Accuracy: {-res.fun:.4f}")
    print(f"Best Params: lr={res.x[0]:.4f}, dropout={res.x[1]:.4f}, hidden={res.x[2]}")
    return res.x

if __name__ == "__main__":
    print("Starting Bayesian Optimization...")
    best_svm = tune_svm()
    best_kfda = tune_kfda_svm()
    # best_mlp = tune_mlp() # Optional, might take longer
    
    print("\n=== Final Verification on Test Set with Best Parameters ===")
    
    # 1. SVM
    print(f"Training Best SVM (C={best_svm[0]:.2f}, gamma={best_svm[1]:.4f})...")
    svm = SVMClassifier(kernel='rbf', C=best_svm[0], gamma=best_svm[1])
    svm.fit(X_norm, y)
    acc = accuracy_score(y_test, svm.predict(X_test_norm))
    print(f"Best SVM Test Accuracy: {acc:.4f}")
    
    # 2. kFDA
    print(f"Training Best kFDA (gamma={best_kfda[0]:.4f}, C={best_kfda[1]:.2f}, n={best_kfda[2]})...")
    kfda = get_preprocessor('kfda', n_components=int(best_kfda[2]), kernel='rbf', gamma=best_kfda[0])
    X_trans = kfda.fit_transform(X_norm, y)
    X_test_trans = kfda.transform(X_test_norm)
    
    clf = SVMClassifier(kernel='linear_svc', C=best_kfda[1])
    clf.fit(X_trans, y)
    acc = accuracy_score(y_test, clf.predict(X_test_trans))
    print(f"Best kFDA Test Accuracy: {acc:.4f}")

    # 3. KFDA + SVM
    print(f"Training Best KFDA + SVM (gamma={best_kfda[0]:.4f}, C={best_kfda[1]:.2f}, n={best_kfda[2]})...")
    kfda = get_preprocessor('kfda', n_components=int(best_kfda[2]), kernel='rbf', gamma=best_kfda[0])
    X_trans = kfda.fit_transform(X_norm, y)
    X_test_trans = kfda.transform(X_test_norm)
    
    clf = SVMClassifier(kernel='linear_svc', C=best_kfda[1])
    clf.fit(X_trans, y)
    acc = accuracy_score(y_test, clf.predict(X_test_trans))
    print(f"Best KFDA + SVM Test Accuracy: {acc:.4f}")
