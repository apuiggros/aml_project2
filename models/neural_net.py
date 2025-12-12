import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder

class MLPDataModule(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.0, activation='relu'):
        super(MLPDataModule, self).__init__()
        layers = []
        in_dim = input_dim
        
        act_fn = nn.ReLU() if activation == 'relu' else nn.Tanh()
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PyTorchMLP(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible PyTorch MLP.
    """
    def __init__(self, hidden_layers=[100], activation='relu', dropout=0.0, 
                 lr=0.001, weight_decay=1e-4, epochs=100, batch_size=32,
                 device='cpu', verbose=False):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Encoder labels to 0..k-1
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y_enc, dtype=torch.long).to(self.device)
        
        input_dim = X.shape[1]
        self.model_ = MLPDataModule(input_dim, self.hidden_layers, self.n_classes_, 
                                    self.dropout, self.activation).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if self.verbose and (epoch % 10 == 0):
                print(f"Epoch {epoch}: Loss {epoch_loss / len(loader):.4f}")
                
        return self

    def predict(self, X):
        check_is_fitted(self, ['model_', 'classes_'])
        X = check_array(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)
            _, preds = torch.max(outputs, 1)
            
        return self.le_.inverse_transform(preds.cpu().numpy())
