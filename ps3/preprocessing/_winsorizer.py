import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_bound_ = np.percentile(X, self.lower_quantile * 100)
        self.upper_bound_ = np.percentile(X, self.upper_quantile * 100)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X_clipped = np.clip(X, self.lower_bound_, self.upper_bound_)
        return X_clipped
