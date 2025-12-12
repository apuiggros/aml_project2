from sklearn.svm import SVC, LinearSVC
from sklearn.base import BaseEstimator, ClassifierMixin
from experiments.heuristics import estimate_gamma

class SVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper for SVMs to easily switch between Linear and Kernel implementations
    and multi-class strategies.
    """
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', degree=3, coef0=0.0, 
                 decision_function_shape='ovo', max_iter=-1):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.decision_function_shape = decision_function_shape
        self.max_iter = max_iter
        self.model_ = None

    def fit(self, X, y):
        # Handle heuristic gamma
        gamma_value = self.gamma
        if self.gamma == 'heuristic':
            gamma_value = estimate_gamma(X)
            # print(f"Heuristic Gamma: {gamma_value:.4f}")

        if self.kernel == 'linear_svc':
            # LinearSVC is generally faster for linear case and implements OVR by default (multi_class='ovr')
            # It handles multi-class differently than SVC(kernel='linear')
            self.model_ = LinearSVC(C=self.C, max_iter=max(1000, self.max_iter), 
                                    dual="auto")
        else:
            kernel_name = self.kernel
            if self.kernel == 'linear_svc_primal': # Fallback naming if needed
                 kernel_name = 'linear'

            self.model_ = SVC(kernel=kernel_name, C=self.C, gamma=gamma_value, 
                              degree=self.degree, coef0=self.coef0,
                              decision_function_shape=self.decision_function_shape,
                              max_iter=self.max_iter)
        
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        # SVC needs probability=True for predict_proba, which slows it down.
        # We might not need probas for all experiments, but if we do, we need to set it.
        # Let's enforce probability=True for SVC if we want probas?
        # For now, let's assume we rely on decision function or prediction.
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        # Fallback or raise
        raise NotImplementedError("Probability prediction not enabled (requires probability=True in SVC)")
