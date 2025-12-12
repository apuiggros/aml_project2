# Multi-Class Land-Cover Classification Project

This project implements a multi-class classification system for satellite imagery using the **Statlog (Landsat Satellite)** dataset. It explores the performance of kernel methods (SVM, kPCA, kFDA) versus neural networks (MLPs), with a specific focus on understanding how feature space mappings affect classification decisions.

## 1. Theoretical Background

### 1.1 The Dataset
The dataset consists of multi-spectral values from 3x3 pixel neighborhoods in Landsat satellite images.
- **Input**: 36 features (4 spectral bands $\times$ 9 pixels).
- **Target**: Land cover class (6 classes: soil, crops, vegetation, etc.).
- **Goal**: Predict the central pixel's class based on the neighborhood's spectral signature.

### 1.2 Dimensionality Reduction & Feature Extraction
We employ both unsupervised and supervised methods to transform the 36-dimensional input space.

#### Principal Component Analysis (PCA)
- **Type**: Unsupervised Linear.
- **Concept**: Eigendecomposition of the covariance matrix to find orthogonal directions of maximum variance.
- **Role**: Reduces noise and redundancy but may discard discriminative information if variance $\neq$ class separation.

#### Kernel PCA (kPCA)
- **Type**: Unsupervised Non-Linear.
- **Concept**: Performs PCA in a high-dimensional Feature Space $\mathcal{F}$ implicitly defined by a kernel function $k(x, y) = \langle \phi(x), \phi(y) \rangle$.
- **Role**: Unfolds non-linear manifolds (e.g., if data lies on a curved surface), potentially making classes more separable for linear classifiers.

#### Fisher Discriminant Analysis (FDA/LDA)
- **Type**: Supervised Linear.
- **Concept**: maximize the Fisher criterion $J(w) = \frac{w^T S_B w}{w^T S_W w}$, maximizing between-class scatter ($S_B$) while minimizing within-class scatter ($S_W$).
- **Role**: Finds the best linear axes for class separation.

#### Kernel FDA (kFDA)
- **Type**: Supervised Non-Linear.
- **Concept**: Solves the generalized eigenvalue problem $M\alpha = \lambda N\alpha$ in feature space, where $M$ and $N$ are kernelized scatter matrices.
- **Implementation**: We implemented this from scratch in `features/kfda.py`, calculating the kernel matrices carefully to avoid explicit high-dim mappings.
- **Role**: Finds the most discriminative non-linear directions. This often yields the highest performance for complex boundaries.

### 1.3 Classification Models

#### Support Vector Machines (SVM)
- **Linear**: Max-margin hyperplane in input space.
- **Kernel (RBF/Poly)**: Max-margin hyperplane in feature space. The RBF kernel implies an infinite-dimensional space, effectively handling highly non-linear boundaries.

#### Multi-Layer Perceptrons (MLP)
- **Architecture**: A stack of Linear layers interleaved with non-linear activations (ReLU).
- **Learning**: Unlike SVMs which use a fixed kernel, MLPs *learn* the feature transformation via backpropagation.
- **Regularization**: 
  - **Dropout**: Randomly zeros neurons to prevent co-adaptation.
  - **Weight Decay**: L2 penalty on weights to control model complexity.

---

## 2. Project Structure

The project is organized into modular components:

```
AML_PROJECT_2/
├── data/
│   ├── loader.py            # Downloads and loads the Landsat dataset
│   └── sat.trn / sat.tst    # Data files (downloaded automatically)
├── features/
│   ├── kfda.py              # Custom Kernel FDA implementation
│   └── preprocessors.py     # Factory for PCA, kPCA, FDA, kFDA
├── models/
│   ├── svm_models.py        # Wrapper for Linear/Kernel SVMs
│   └── neural_net.py        # PyTorch implementation of MLP (sklearn-compatible)
├── experiments/
│   ├── config.py            # Hyperparameters and search space search
│   ├── heuristics.py        # Gamma estimation heuristic
│   ├── run_experiments.py   # Main experimentation script
│   └── visualize_features.py# Script to generate 2D projections
├── utils/
│   └── metrics.py           # Evaluation metrics (Accuracy, F1, etc.)
├── experiment_results.csv   # Output of the experiments
└── requirements.txt         # Python dependencies
```

## 3. How to Run

### 3.1 Setup
Ensure you have Python installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

### 3.2 Running Experiments
To run the full suite of comparisons (SVM variations, MLPs, and Feature Extraction benchmarks), execute:
```bash
python experiments/run_experiments.py
```
This script will:
1. Download data (if missing).
2. Train and evaluate all SVM configurations defined in `config.py`.
3. Train and evaluate various MLP architectures.
4. Run the feature extraction pipeline (kFDA, etc.) + Linear SVM.
5. Save results to `experiment_results.csv`.

*Note: If the process is interrupted, `experiments/recover_features.py` can be used to resume the feature extraction phase.*

### 3.3 Visualizing Features
To generate 2D scatter plots showing how different methods project the classes:
```bash
python experiments/visualize_features.py
```
This saves `feature_projections.png`.

## 4. Interpreting Results

The `experiment_results.csv` contains:
- **Type**: Model family (SVM, MLP, Feature+LinearSVM).
- **Config**: The specific hyperparameters used.
- **Accuracy**: The primary metric (CV average).

### Key Findings (Sample)
- **Linear Baseline**: ~83%. The problem is non-linear.
- **Kernel SVM (RBF)**: ~88%. Strong performance due to non-linearity.
- **MLPs**: ~90%. Flexible architecture allows comprehensive learning.
- **kFDA + Linear SVM**: ~90%+. **Winner**. kFDA effectively utilizes class labels to find a projection where classes are linearly separable, often beating "black-box" neural networks on this dataset size.
