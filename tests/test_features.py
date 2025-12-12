import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loader import load_data
from features.preprocessors import get_preprocessor

def test_features():
    print("Loading data for testing...")
    # Load just a subset to be fast
    X, y, _, _ = load_data()
    X_sub = X[:500] # 500 samples
    y_sub = y[:500]

    n_classes = len(np.unique(y_sub))
    target_dim = n_classes - 1 # 5

    methods = [
        ('pca', {'n_components': 10}),
        ('kpca', {'n_components': 10, 'kernel': 'rbf'}),
        ('fda', {'n_components': target_dim}),
        ('kfda', {'n_components': target_dim, 'kernel': 'rbf'})
    ]

    for name, params in methods:
        print(f"\nTesting {name} with params {params}...")
        try:
            model = get_preprocessor(name, **params)
            model.fit(X_sub, y_sub)
            X_trans = model.transform(X_sub)
            print(f"Success! Output shape: {X_trans.shape}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_features()
