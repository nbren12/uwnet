def r2_score(truth, pred, mean_dims, dims=None, w=1.0):
    """ R2 score for xarray objects
    """
    if dims is None:
        dims = mean_dims

    mu = truth.mean(mean_dims)
    sum_squares_error = ((truth - pred)**2 * w).mean(dims)
    sum_squares = ((truth - mu)**2 * w).mean(dims)

    return 1 - sum_squares_error / sum_squares
