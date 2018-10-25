import torch
from toolz import curry
from .utils import select_time


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


def compute_multiple_step_loss(criterion, model, batch, initial_time,
                               prediction_length, time_step, prognostics=('QT', 'SLI')):
    loss = 0.0
    for t in range(initial_time, initial_time + prediction_length):
        # Load the initial condition
        initial_condition = {}
        is_first_step = initial_time == t
        for key in batch:
            is_prognostic = key in prognostics
            if is_prognostic:
                if is_first_step:
                    initial_condition[key] = batch[key][t]
                else:
                    initial_condition[key] = one_step_prediction[key]
            else:
                initial_condition[key] = batch[key][t]

        one_step_truth = {key: batch[key][t+1] for key in batch}

        # make a one step prediction
        apparent_sources = model(initial_condition)
        one_step_prediction = {}
        for key in prognostics:
            total_source_term = (apparent_sources[key] / 86400 +
                                    initial_condition['F' + key])
            one_step_prediction[
                key] = initial_condition[key] + time_step * total_source_term

            loss += criterion(one_step_prediction[key],
                              one_step_truth[key])
    return loss
