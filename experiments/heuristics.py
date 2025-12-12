import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def estimate_gamma(X, quantile=0.5):
    """
    Estimates gamma for RBF kernel using the quantile of pairwise distances.
    gamma = 1 / (2 * sigma^2)
    where sigma is the quantile distance.
    """
    # Sample a subset if X is too large to avoid O(N^2)
    if X.shape[0] > 1000:
        idx = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        
    dists = euclidean_distances(X_sample, X_sample)
    # Take upper triangle
    valid_dists = dists[np.triu_indices(dists.shape[0], k=1)]
    
    if len(valid_dists) == 0:
        return 1.0 / X.shape[1] # fallback
        
    sigma = np.quantile(valid_dists, quantile)
    if sigma == 0:
        sigma = np.mean(valid_dists)
        
    # gamma = 1 / (2 * sigma^2)
    gamma = 1.0 / (2 * sigma**2)
    return gamma
