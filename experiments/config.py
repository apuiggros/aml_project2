# Experiment configurations

# 1. Comparison: SVM Linear vs Kernel
SVM_CONFIGS = [
    {'kernel': 'linear_svc', 'C': 1.0},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'heuristic'}, # Will use heuristic
    {'kernel': 'poly', 'degree': 2, 'C': 1.0},
    # Placeholder for Optimized
    {'kernel': 'rbf', 'C': 15.0674, 'gamma': 0.0048},
]

# 2. MLP Configs
MLP_CONFIGS = [
    {'hidden_layers': [50], 'dropout': 0.0, 'lr': 0.001, 'epochs': 50},
    {'hidden_layers': [100, 50], 'dropout': 0.2, 'lr': 0.001, 'epochs': 50},
    {'hidden_layers': [200, 100, 50], 'dropout': 0.5, 'lr': 0.001, 'epochs': 100} # Deeper
]

# 3. Preprocessors
# 3. Preprocessors (Name, FeatParams, ClassifierParams)
FEATURE_CONFIGS = [
    ('raw', {}, {'C': 1.0}),
    ('pca', {'n_components': 15}, {'C': 1.0}),
    ('kpca', {'n_components': 15, 'kernel': 'rbf', 'gamma': None}, {'C': 1.0}), 
    ('fda', {'n_components': 5}, {'C': 1.0}),
    ('kfda', {'n_components': 5, 'kernel': 'rbf', 'gamma': 'scale'}, {'C': 1.0}),
    ('kfda_opt', {'n_components': 5, 'kernel': 'rbf', 'gamma': 0.0686}, {'C': 100}) 
]
