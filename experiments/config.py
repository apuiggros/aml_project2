# Experiment configurations

# 1. Comparison: SVM Linear vs Kernel
SVM_CONFIGS = [
    {'kernel': 'linear_svc', 'C': 1.0},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'heuristic'}, # Will use heuristic
    {'kernel': 'poly', 'degree': 2, 'C': 1.0},
]

# 2. MLP Configs
MLP_CONFIGS = [
    {'hidden_layers': [50], 'dropout': 0.0, 'lr': 0.001, 'epochs': 50},
    {'hidden_layers': [100, 50], 'dropout': 0.2, 'lr': 0.001, 'epochs': 50},
    {'hidden_layers': [200, 100, 50], 'dropout': 0.5, 'lr': 0.001, 'epochs': 100} # Deeper
]

# 3. Preprocessors
FEATURE_CONFIGS = [
    ('raw', {}),
    ('pca', {'n_components': 15}),
    ('kpca', {'n_components': 15, 'kernel': 'rbf', 'gamma': None}), # 'scale' not supported in this sklearn version
    ('fda', {'n_components': 5}), # Max components = classes - 1
    ('kfda', {'n_components': 5, 'kernel': 'rbf', 'gamma': 'scale'}) # Handled in KFDA class
]
