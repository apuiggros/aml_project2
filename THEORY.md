# Theory & Background

This document explains the machine learning concepts and theories applied in the Landsat Classification Project.

## 1. The Dataset: Statlog (Landsat Satellite)
The dataset consists of multi-spectral values of pixels in 3x3 neighbourhoods in a satellite image, and the classification associated with the central pixel in each neighbourhood. The aim is to predict the classification, given the multi-spectral values.
- **Features**: 36 numerical features (4 spectral bands x 9 pixels in neighborhood).
- **Classes**: 6 land cover classes (e.g., red soil, cotton crop, grey soil, damp grey soil, soil with vegetation stubble, mixture).

## 2. Preprocessing & Dimensionality Reduction
We explore both unsupervised and supervised dimensionality reduction to map the 36-dimensional input space into a lower-dimensional manifold or a more discriminative space.

### 2.1 Principal Component Analysis (PCA)
- **Concept**: Unsupervised linear transformation that projects data onto the directions of maximum variance.
- **Goal**: De-correlate features and reduce noise/redundancy.
- **Math**: Eigendecomposition of the covariance matrix $C = \frac{1}{n} X^T X$. The top $k$ eigenvectors form the projection matrix.

### 2.2 Kernel PCA (kPCA)
- **Concept**: Non-linear extension of PCA using the **Kernel Trick**.
- **Mechanism**: The data is implicitly mapped to a high-dimensional feature space $\phi(x)$ via a kernel function $k(x, y) = \langle \phi(x), \phi(y) \rangle$. Standard PCA is performed in this feature space.
- **Effect**: Can unfold non-linear manifolds (e.g., spectral clusters) that standard PCA would squash.

### 2.3 Fisher Discriminant Analysis (FDA/LDA)
- **Concept**: Supervised linear transformation.
- **Goal**: Maximize the separation between class means relative to the within-class variance.
- **Objective**: Maximize $J(w) = \frac{w^T S_B w}{w^T S_W w}$, where $S_B$ is the between-class scatter matrix and $S_W$ is the within-class scatter matrix.

### 2.4 Kernel FDA (kFDA)
- **Concept**: Non-linear extension of FDA using the kernel trick.
- **Mechanism**: The projection vectors $\mathbf{\alpha}$ are found by solving the generalized eigenvalue problem in the feature space:
  $$M \mathbf{\alpha} = \lambda N \mathbf{\alpha}$$
  where $M$ is the between-class scatter matrix and $N$ is the within-class scatter matrix (regularized) defined in terms of the kernel matrix $K$.
  - $M = \sum_{c} n_c (\mathbf{m}_c - \mathbf{m})(\mathbf{m}_c - \mathbf{m})^T$ where $\mathbf{m}_c$ is the class mean in feature space (represented by column means of $K$).
  - $N = \sum_{c} K_c (I - \mathbf{1}_{n_c}) K_c^T + \xi I$, where $K_c$ are the kernel columns for class $c$.
- **Benefit**: Can find complex, non-linear discriminative directions that separate classes effectively in the input space.

## 3. Classification Models

### 3.1 Support Vector Machines (SVM)
- **Linear SVM**: Finds the optimal hyperplane that maximizes the margin between classes.
- **Kernel SVM**: Uses kernels (RBF, Polynomial) to learn non-linear decision boundaries. The decision function becomes $f(x) = \sum \alpha_i y_i k(x_i, x) + b$.

### 3.2 Multi-Layer Perceptron (MLP)
- **Architecture**: Feed-forward neural network with hidden layers.
- **vs. Kernels**: While kernels rely on fixed feature mappings (infinite dimensional for RBF), MLPs *learn* the feature representation through their hidden layers.
- **Regularization**: 
    - **Weight Decay** ($L_2$ regularization): Penalizes large weights to prevent overfitting.
    - **Dropout**: Randomly zeroes out neurons during training to encourage robust feature learning.

## 4. Experimental Strategy
We use Stratified K-Fold Cross-Validation to ensure reliable performance estimates, given the moderate dataset size. We analyze not just accuracy, but also precision, recall, and confusion matrices to understand misclassifications.
