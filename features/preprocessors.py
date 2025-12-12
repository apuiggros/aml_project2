from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from .kfda import KernelFDA

def get_preprocessor(name, n_components=None, **kwargs):
    """
    Factory function to get preprocessors.
    Args:
        name (str): 'pca', 'kpca', 'fda', 'kfda'
        n_components (int): Target dimensions
        **kwargs: Additional args for specific preprocessors
    """
    if name == 'pca':
        return PCA(n_components=n_components, **kwargs)
    
    elif name == 'kpca':
        # KernelPCA defaults: kernel='linear', need to specify 'rbf' etc in kwargs if desired
        return KernelPCA(n_components=n_components, **kwargs)
    
    elif name == 'fda':
        # LDA in sklearn requires n_components to be <= min(n_classes-1, n_features)
        # We create it, fit will constrain it naturally or we can handle it.
        return LinearDiscriminantAnalysis(n_components=n_components, **kwargs)
    
    elif name == 'kfda':
        return KernelFDA(n_components=n_components, **kwargs)
        
    else:
        raise ValueError(f"Unknown preprocessor: {name}")
