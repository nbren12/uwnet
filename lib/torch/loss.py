import torch
from toolz import curry


@curry
def weighted_loss(weight, x, y):
    # return torch.mean(torch.pow(x - y, 2).mul(weight.float()))
    return torch.mean(torch.abs(x - y).mul(weight.float()))


@curry
def dynamic_loss(truth, pred,
                 weights=None,
                 precip_in_loss=False,
                 radiation_in_loss=False):
    x = truth['prognostic']
    y = pred['prognostic']

    total_loss = 0
    # time series loss
    for key in y:
        w = weights.get(key, 1.0)
        total_loss += weighted_loss(w, x[key], y[key]) / len(y)

    if precip_in_loss:
        # column budget losses this compares he predicted precipitation for
        # each field to reality
        prec = truth['forcing']['Prec']
        predicted = pred['diagnostic']['Prec']
        observed = (prec[1:] + prec[:-1]) / 2
        total_loss += torch.mean(torch.pow(observed - predicted, 2)) / 5

    if radiation_in_loss:
        qrad = truth['forcing']['QRAD']
        predicted = pred['diagnostic']['QRAD'][0]
        observed = qrad[0]
        total_loss += torch.mean(torch.pow(observed - predicted, 2))

    return total_loss
