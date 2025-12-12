"""
Kernel Fisher Discriminant Analysis (KFDA) Implementation.

Reference:
Mika, S., Ratsch, G., Weston, J., Scholkopf, B., & Muller, K. R. (1999). 
Fisher discriminant analysis with kernels. 
Neural networks for signal processing IX, 41-48.
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class KernelFDA(BaseEstimator, TransformerMixin):
    """
    Kernel Fisher Discriminant Analysis (KFDA).
    Projects data into a lower dimensional space maximizing the between-class 
    scatter while minimizing the within-class scatter, using the kernel trick.
    """
    def __init__(self, n_components=None, kernel="rbf", gamma=None, degree=3, coef0=1, alpha=1e-3):
        """
        Args:
            n_components (int): Number of components to keep.
            kernel (str): Kernel type ('linear', 'rbf', 'poly', etc.).
            gamma (float): Kernel coefficient for rbf/poly.
            degree (int): Degree for poly kernel.
            coef0 (float): Independent term for poly/sigmoid.
            alpha (float): Regularization parameter for within-class scatter matrix.
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.X_fit_ = X
        classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(classes)

        if self.n_components is None:
            self.n_components = n_classes - 1

        # 1. Compute Kernel Matrix K
        # K has shape (n_samples, n_samples)
        
        # Resolve gamma if 'scale' or 'auto'
        gamma = self.gamma
        if isinstance(gamma, str):
            if gamma == 'scale':
                X_var = X.var()
                gamma = 1.0 / (n_features * X_var) if X_var > 0 else 1.0
            elif gamma == 'auto':
                gamma = 1.0 / n_features
        
        # Filter params for pairwise_kernels
        params = {"gamma": gamma, "degree": self.degree, "coef0": self.coef0}
        # Remove None values
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        # Specific filtering for common kernels to avoid TypeErrors
        if self.kernel == 'rbf':
            params.pop('degree', None)
            params.pop('coef0', None)
        elif self.kernel == 'linear':
            params.pop('gamma', None)
            params.pop('degree', None)
            params.pop('coef0', None)
        elif self.kernel == 'poly':
            pass # keeps all
        
        K = pairwise_kernels(X, X, metric=self.kernel, **params)
        
        # Save params for transform
        self._kernel_params = params

        # 2. Compute M (Between-class scatter) and N (Within-class scatter) in feature space
        # We work with the expansion coefficients alpha.
        # N = K K^T - \sum_c n_c K_c K_c^T (Wait, simplified computation below)
        
        # More efficient approach using centered kernel matrices per class
        # M = \sum n_c (mu_c - mu)(mu_c - mu)^T
        # N = \sum K_c (I - 1/n_c 11^T) K_c^T 
        # But we need these in terms of the dual coefficients alpha.
        
        # Following Mika et al.:
        # Maximize J(alpha) = (alpha^T M alpha) / (alpha^T N alpha)
        # alpha are the coefficients for the projection vector w = \sum alpha_i phi(x_i)
        
        # M = \sum_{j=1}^C n_j (M_j - M_star)(M_j - M_star)^T
        # where M_j is mean of class j in feature space (column vector of K-means), M_star is global mean
        # Actually in terms of K:
        # (M)_kl = \sum_{j=1}^C n_j (mu_j^k - mu_*^k)(mu_j^l - mu_*^l) 
        # where mu_j is the mean of the j-th class in terms of kernel values.
        
        # Let's use the matrix formulation:
        # N = K K  - \sum_c n_c (K_c_mean)(K_c_mean)^T 
        # is sometimes computationally unstable.
        
        # Alternative construction:
        # N = sum_c K_c (I - 1_{n_c}) K_c^T is correct but usually we define:
        # N = K @ K (if centered?) No.
        
        # Let's build N explicitly:
        # N = \sum_{c} K_c (I - 1/n_c J) K_c^T
        # where K_c is the submatrix of K with columns belonging to class c (n_samples x n_samples_of_c) NO, 
        # K_c should be columns corresponding to class c? 
        # w = sum alpha_i phi(x_i). 
        # We need N such that alpha^T N alpha = w^T S_W w.
        
        N = np.zeros((n_samples, n_samples))
        M = np.zeros((n_samples, n_samples)) # Between class
        
        # Global mean of kernel rows
        mk_total = np.mean(K, axis=1, keepdims=True) # (n_samples, 1)
        
        for c in classes:
            indices = np.where(y == c)[0]
            n_c = len(indices)
            if n_c == 0: continue
            
            # Kernel columns for this class
            K_c = K[:, indices] # (n_samples, n_c)
            
            # Mean of this class in Kernel space (coefficient-wise)
            mk_c = np.mean(K_c, axis=1, keepdims=True) # (n_samples, 1)
            
            # Update M: n_c * (mk_c - mk_total)(mk_c - mk_total)^T
            diff = mk_c - mk_total
            M += n_c * (diff @ diff.T)
            
            # Update N: K_c (I - 1/n_c 11^T) K_c^T
            # Center K_c
            Identity = np.eye(n_c)
            One = np.ones((n_c, n_c)) / n_c
            Centered_K_c = K_c @ (Identity - One)
            # N += Centered_K_c @ Centered_K_c.T # This is K_c (I-1/n)(I-1/n)^T K_c^T? YES
            # Wait, (I-1/n) is idempotent.
            
            N += K_c @ (Identity - One) @ K_c.T

        # Regularization to ensure invertibility
        N += self.alpha * np.eye(n_samples)

        # Solve generalized eigenvalue problem: M alpha = lambda N alpha
        # We use eigh for symmetric matrices
        # But M and N are symmetric?
        # M is sum of rank-1 updates (symmetric). N is sum of (A A^T) (symmetric).
        # So we can use eigh(M, N) or eigh(N^-1 M) if N invertible.
        # eigh(a, b) solves a x = w b x
        
        # We want to maximize lambda, so take top eigenvectors.
        vals, vecs = eigh(M, N, subset_by_index=(n_samples - self.n_components, n_samples - 1))
        
        # Sort descending
        idx = np.argsort(vals)[::-1]
        self.alphas_ = vecs[:, idx]
        self.eigenvalues_ = vals[idx]
        
        return self

    def transform(self, X):
        check_is_fitted(self, ['X_fit_', 'alphas_', '_kernel_params'])
        X = check_array(X)
        
        # Compute Kernel between X and X_fit_
        K_new = pairwise_kernels(X, self.X_fit_, metric=self.kernel, **self._kernel_params)
        
        # Project
        return K_new @ self.alphas_

