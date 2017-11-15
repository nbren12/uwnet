"""Module for evaluating models
"""


def mean_squared_error(x, y, axis=-1):
    return ((x - y)**2).mean(axis=axis)


def weighted_r2_score(y_test, y_pred, weight):
    """Compute weighted R2 score

    Parameters
    ----------
    y_test : (n, m)
        numpy array of truth
    y_pred : (n, k)
        numpy array of prediction
    weight : (k,)
        weights of each feature

    """

    mse = mean_squared_error(y_test, y_pred, axis=0)
    variance = mean_squared_error(y_test, y_test.mean(axis=0), axis=0)

    return 1 - (weight * mse).sum() / (weight * variance).sum()
