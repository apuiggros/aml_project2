import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_data

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_class_distribution(y):
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(y, return_counts=True)
    class_names = ['Red Soil', 'Cotton Crop', 'Grey Soil', 'Damp Grey Soil', 'Vegetation Stubble', 'Mixture'] # Based on class mapping 1,2,3,4,5,7
    # Note: data loader keeps original labels [1,2,3,4,5,7]
    
    # Map labels to names if possible, or just used raw
    # Original labels: 1, 2, 3, 4, 5, 7
    mapping = {1: 'Red Soil', 2: 'Cotton Crop', 3: 'Grey Soil', 4: 'Damp Grey Soil', 5: 'Veg Stubble', 7: 'Mixture'}
    labels = [mapping.get(u, str(u)) for u in unique]
    
    sns.barplot(x=labels, y=counts, palette="viridis")
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=300)
    print("Saved class_distribution.png")

def plot_correlation_matrix(X):
    # Features 0-35
    df = pd.DataFrame(X)
    corr = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", vmax=1.0, vmin=-1.0)
    plt.title("Feature Correlation Matrix (36 Features)")
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=300)
    print("Saved correlation_matrix.png")

def plot_spectral_profiles(X, y):
    # 4 bands, 9 pixels. 
    # Features are ordered: 
    # Pixel 1: Band 1, 2, 3, 4 ? Or Band 1 (Px1..9), Band 2...
    # UCI Statlog desc: "36 attributes = 4 spectral bands x 9 pixels".
    # Usually ordered by pixel then band or band then pixel.
    # Statlog: "The 36 attributes are ... 4 spectral values for each of the 9 pixels in the 3x3 neighbourhood"
    # Order: P1_B1, P1_B2, P1_B3, P1_B4, P2_B1... ? 
    # Actually checking docs or correlation structure reveals it usually.
    # Let's assume standard interleaved or block. 
    # A common way to visualize "signature" is to average the 9 pixels for each band, 
    # OR just look at the central pixel (which is what we classify).
    # Central pixel is typically pixel 5.
    
    # If 36 cols: 
    # If interleaved: P1B1, P1B2... 
    # Let's average all pixels for each band to get a "Mean Spectral Profile" per class.
    
    # We have 4 bands. 
    # Let's assume features 0,1,2,3 are Pixel 1 Bands 1-4.
    # Features 4,5,6,7 are Pixel 2...
    # So Band 1 indices: 0, 4, 8... 
    # Band 2 indices: 1, 5, 9...
    
    # Construct Band means for each sample
    bands = [[], [], [], []]
    for i in range(9):
        # Pixel i (0-8)
        # Features 4*i + 0 -> Band 1
        # Features 4*i + 1 -> Band 2
        # ...
        bands[0].extend(range(4*i, 4*i+1)) # Band 1 cols
        bands[1].extend(range(4*i+1, 4*i+2)) # Band 2 cols
        bands[2].extend(range(4*i+2, 4*i+3)) # Band 3 cols
        bands[3].extend(range(4*i+3, 4*i+4)) # Band 4 cols
        
    X_bands = np.zeros((X.shape[0], 4))
    for b in range(4):
        # Average the 9 pixels for this band
        cols = [4*i + b for i in range(9)]
        X_bands[:, b] = np.mean(X[:, cols], axis=1)
        
    # Now plot mean profile per class
    df_bands = pd.DataFrame(X_bands, columns=['Band 1', 'Band 2', 'Band 3', 'Band 4'])
    df_bands['Class'] = y
    
    melted = df_bands.melt(id_vars='Class', var_name='Band', value_name='Intensity')
    
    plt.figure(figsize=(10, 6))
    mapping = {1: 'Red Soil', 2: 'Cotton', 3: 'Grey Soil', 4: 'Damp Grey', 5: 'Stubble', 7: 'Mixture'}
    melted['Class Name'] = melted['Class'].map(mapping)
    
    sns.lineplot(data=melted, x='Band', y='Intensity', hue='Class Name', marker='o')
    plt.title("Mean Spectral Profile per Class (Averaged over 3x3 Neighborhood)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("spectral_profiles.png", dpi=300)
    print("Saved spectral_profiles.png")


if __name__ == "__main__":
    X, y, Xt, yt = load_data()
    # Use full training data
    
    plot_class_distribution(y)
    plot_correlation_matrix(X)
    plot_spectral_profiles(X, y)
