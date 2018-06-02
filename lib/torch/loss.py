import torch
from toolz import curry


@curry
def weighted_loss(weight, x, y):
    # return torch.mean(torch.pow(x - y, 2).mul(weight.float()))
    return torch.mean(torch.abs(x - y).mul(weight.float()))


@curry
def dynamic_loss(truth, pred, weights=None):
    x = truth['prognostic']
    y = pred['prognostic']

    total_loss = 0
    # time series loss
    for key in y:
        w = weights.get(key, 1.0)
        total_loss += weighted_loss(w, x[key], y[key]) / len(y)

    return total_loss
