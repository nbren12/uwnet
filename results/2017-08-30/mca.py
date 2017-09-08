"""Code for maximum covariance analysis regression
"""
from scipy.linalg import svd, lstsq
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator


class MCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4):
        self.n_components = n_components

    def fit(self, X, Y):

        m, n = X.shape

        self.cov_ = X.T.dot(Y)/(m-1)  # covariance matrix
        U, S, Vt = svd(self.cov_, full_matrices=False)

        self._u = U
        self._v = Vt.T

        # x_scores = self.transform(X)
        # b = lstsq(x_scores, Y)[0]
        # self.coef_ = self.x_components_ @ b

        return self

    def transform(self, X, y=None):
        x_scores = X.dot(self.x_components_)

        if y is not None:
            y_scores_ = y.dot(self.y_components_)
            return x_scores, y_scores_

        return x_scores

    @property
    def x_components_(self):
        return self._u[:, :self.n_components]

    @property
    def y_components_(self):
        return self._v[:, :self.n_components]
