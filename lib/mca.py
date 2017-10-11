"""Code for maximum covariance analysis regression
"""
import numpy as np
from scipy.linalg import svd, lstsq
from sklearn.base import RegressorMixin, TransformerMixin, BaseEstimator


class MCA(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=4, demean=True,
                 y_transformer=None):
        self.n_components = n_components
        self.demean = demean

        if y_transformer is None:
            self.y_transformer = lambda x: x
        else:
            self.y_transformer = y_transformer

    def fit(self, X, y=None):

        # This is basically a hack to work around the difficult
        # with using scaled output for MCA but not for the linear regression
        y = self.y_transformer.fit_transform(y)

        X = np.asarray(X)
        Y = np.asarray(y)

        m, n = X.shape
        self.x_mean_ = X.mean(axis=0)
        self.y_mean_ = Y.mean(axis=0)

        if self.demean:
            X = X - self.x_mean_
            Y = Y - self.y_mean_

        self.cov_ = X.T.dot(Y)/(m-1)  # covariance matrix
        U, S, Vt = svd(self.cov_, full_matrices=False)

        self._u = U
        self._v = Vt.T

        # x_scores = self.transform(X)
        # b = lstsq(x_scores, Y)[0]
        # self.coef_ = self.x_components_ @ b

        return self

    def transform(self, X, y=None):

        X = np.asarray(X)
        # Y = np.asarray(y)
        x_scores = (X-self.x_mean_).dot(self.x_components_)
        # if y is not None:
        #     y_scores = (Y-self.y_mean_).dot(self.y_components_)
        #     return x_scores, y_scores
        # else:
        return x_scores


    def inverse_transform(self, X):
        return X.dot(self.x_components_.T)

    @property
    def x_components_(self):
        return self._u[:, :self.n_components]

    @property
    def y_components_(self):
        return self._v[:, :self.n_components]
