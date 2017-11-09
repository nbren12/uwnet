"""Module for evaluating models
"""

def mse(truth, pred, dims=('x', 'y', 'time')):
    """Mean squared error
    """
    sse = ((pred - truth)**2).mean(dims)
    return sse
