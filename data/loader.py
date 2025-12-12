import os
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"

def download_file(url, filepath):
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping download.")
        return
    print(f"Downloading {url} to {filepath}...")
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)
    print("Done.")

def load_data():
    """
    Downloads (if necessary) and loads the Statlog (Landsat Satellite) dataset.
    Returns:
        X_train, y_train, X_test, y_test (numpy arrays)
    """
    train_path = os.path.join(DATA_DIR, "sat.trn")
    test_path = os.path.join(DATA_DIR, "sat.tst")

    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL, test_path)

    # The dataset is space-separated
    # Columns 0-35 are features, Column 36 is the label
    
    print("Loading data...")
    train_df = pd.read_csv(train_path, sep='\\s+', header=None)
    test_df = pd.read_csv(test_path, sep='\\s+', header=None)

    X_train = train_df.iloc[:, :-1].values
    y_train = train_df.iloc[:, -1].values
    
    X_test = test_df.iloc[:, :-1].values
    y_test = test_df.iloc[:, -1].values

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Classes: {np.unique(y_train)}")

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    load_data()
