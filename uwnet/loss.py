import torch
from toolz import curry
from .timestepper import Batch, predict_multiple_steps


def get_other_dims(x, dim):
    return set(range(x.dim())) - {dim}


def mean_over_dims(x, dims):
    """Take a mean over a list of dimensions keeping the as singletons"""
    for dim in dims:
        x = x.mean(dim=dim, keepdim=True)
    return x


def mean_other_dims(x, dim):
    """Take a mean over all dimensions but the one specified"""
    other_dims = get_other_dims(x, dim)
    return mean_over_dims(x, other_dims)


def weighted_r2_score(truth, prediction, weights, dim=-1):
    """Compute the weighted R2 score"""
    mean = mean_other_dims(truth, dim)
    squares = weighted_mean_squared_error(truth, mean, weights, dim)
    residuals = weighted_mean_squared_error(truth, prediction, weights, dim)
    return 1 - residuals/squares


def weighted_mean_squared_error(truth, prediction, weights, dim=-1):
    """Compute the weighted mean squared error

    Parameters
    ----------
    truth: torch.tensor
    prediction: torch.tensor
    weights: torch.tensor
        one dimensional tensor of weights. Must be the same length as the truth
        and prediction arrays along the dimension `dim`
    dim:
        the dimension the data should be weighted along
    """
    error = truth - prediction
    error2 = error * error
    error2 = mean_other_dims(error2, dim)
    return torch.mean(error2 * weights)


def mse(x, y, layer_mass):
    x = x.float()
    y = y.float()
    layer_mass = layer_mass.float()
    w = layer_mass / layer_mass.mean()

    if x.dim() == 3:
        x = x.unsqueeze(1)

    if x.size(1) > 1:
        if layer_mass.size(0) != x.size(1):
            raise ValueError
        return torch.mean(torch.pow(x - y, 2) * w)
    else:
        return torch.mean(torch.pow(x - y, 2))


@curry
def MVLoss(keys, layer_mass, scale, x, y):
    """MSE loss

    Parameters
    ----------
    keys
        list of keys to compute loss with
    x : truth
    y : prediction
    """

    losses = {
        key:
        mse(x[key], y[key], layer_mass) / torch.tensor(scale[key]**2).float()
        for key in keys
    }
    return sum(losses.values())


def select_keys_time(x, keys, t):
    return {key: x[key][t] for key in keys}


def compute_loss(criterion, prognostics, y):
    return sum(criterion(prognostics[key], y[key]) for key in prognostics)


def compute_multiple_step_loss(criterion, model, batch, prognostics, *args,
                               **kwargs):
    """Compute the loss across multiple time steps with an Euler stepper

    Yields
    ------
    t: int
       the time step of the prediction
    prediction: dict
       the predicted state

    """
    batch = Batch(batch, prognostics)
    prediction_generator = predict_multiple_steps(model, batch, *args,
                                                  **kwargs)
    return sum(
        compute_loss(criterion, prediction, batch.get_prognostics_at_time(t))
        for t, prediction in prediction_generator)
