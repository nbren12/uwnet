import torch
from toolz import curry


def mse(x, y, layer_mass):
    x = x.float()
    y = y.float()
    layer_mass = layer_mass.float()
    w = layer_mass / layer_mass.mean()

    if x.dim() == 2:
        x = x[..., None]

    if x.size(-1) > 1:
        if layer_mass.size(-1) != x.size(-1):
            raise ValueError

        return torch.mean(torch.pow(x - y, 2) * w)
    else:
        return torch.mean(torch.pow(x - y, 2))


@curry
def MVLoss(layer_mass, scale, x, y):
    """MSE loss

    Parameters
    ----------
    x : truth
    y : prediction
    """

    losses = {
        key:
        mse(x[key], y[key], layer_mass) / torch.tensor(scale[key]**2).float()
        for key in scale
    }
    return sum(losses.values())
