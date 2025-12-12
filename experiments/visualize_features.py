import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_data
from features.preprocessors import get_preprocessor

def plot_projections():
    print("Loading data...")
    X, y, _, _ = load_data()
    # Subsample for clearer plot
    idx = np.random.choice(len(X), 1000, replace=False)
    X_sub = X[idx]
    y_sub = y[idx]

    methods = [
        ('PCA', 'pca', {}),
        ('Kernel PCA (RBF)', 'kpca', {'kernel': 'rbf', 'gamma': 0.01}), # Tweaked gamma for vis
        ('FDA', 'fda', {}),
        ('Kernel FDA (RBF)', 'kfda', {'kernel': 'rbf', 'gamma': 0.01})
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (title, name, params) in enumerate(methods):
        print(f"Projecting with {title}...")
        try:
            # Force 2 components for visualization
            params['n_components'] = 2
            
            model = get_preprocessor(name, **params)
            
            # FDA/kFDA require y
            if name in ['fda', 'kfda']:
                X_trans = model.fit_transform(X_sub, y_sub)
            else:
                X_trans = model.fit_transform(X_sub)
                
            sns.scatterplot(x=X_trans[:,0], y=X_trans[:,1], hue=y_sub, palette='tab10', 
                            ax=axes[i], legend='full', s=20)
            axes[i].set_title(title)
        except Exception as e:
            print(f"Failed {title}: {e}")
            axes[i].text(0.5, 0.5, "Failed", ha='center')

    plt.tight_layout()
    save_path = "feature_projections.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    plot_projections()
