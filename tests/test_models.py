import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_data
from models.svm_models import SVMClassifier
from models.neural_net import PyTorchMLP
from sklearn.metrics import accuracy_score

def test_models():
    print("Loading data for testing...")
    X, y, _, _ = load_data()
    X_sub = X[:500] 
    y_sub = y[:500]

    print("\n--- Testing SVM (RBF) ---")
    svm = SVMClassifier(kernel='rbf', C=1.0)
    svm.fit(X_sub, y_sub)
    preds = svm.predict(X_sub)
    acc = accuracy_score(y_sub, preds)
    print(f"SVM Train Acc: {acc:.4f}")

    print("\n--- Testing LinearSVC ---")
    lsvm = SVMClassifier(kernel='linear_svc', C=1.0)
    lsvm.fit(X_sub, y_sub)
    preds = lsvm.predict(X_sub)
    acc = accuracy_score(y_sub, preds)
    print(f"LinearSVC Train Acc: {acc:.4f}")

    print("\n--- Testing PyTorch MLP ---")
    # Small epochs for speed
    mlp = PyTorchMLP(hidden_layers=[50], epochs=10, batch_size=32, verbose=True)
    mlp.fit(X_sub, y_sub)
    preds = mlp.predict(X_sub)
    acc = accuracy_score(y_sub, preds)
    print(f"MLP Train Acc: {acc:.4f}")

if __name__ == "__main__":
    test_models()
