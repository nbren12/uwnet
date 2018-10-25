import torch
from toolz import curry, merge
import attr
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


def select_keys_time(x, keys, t):
    return {key: x[key][t] for key in keys}


@attr.s
class Batch(object):
    """A object for getting appropriate fields from a batch of data"""
    data = attr.ib()
    prognostics = attr.ib()

    @property
    def forcings(self):
        return set(self.data.keys()) - set(self.prognostics)

    def get_forcings_at_time(self, t):
        return select_keys_time(self.data, self.forcings, t)

    def get_known_forcings_at_time(self, t):
        out = {}
        for key in self.prognostics:
            original_key = 'F' + key
            if original_key in self.data:
                out[key] = self.data[original_key][t]
        return out

    def get_prognostics_at_time(self, t):
        return select_keys_time(self.data, self.prognostics, t)


def predict_one_step(prognostics, apparent_source, forcing, time_step):
    prediction = {}
    for key in prognostics:
        total_source = (apparent_source[key] / 86400 + forcing[key])
        prediction[key] = prognostics[key] + time_step * total_source
    return prediction


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


def predict_multiple_steps(model, batch: Batch, initial_time, prediction_length,
                           time_step):
    """Yield Euler step predictions with a neural network"""
    prognostics = batch.get_prognostics_at_time(initial_time)
    for t in range(initial_time, initial_time + prediction_length):
        forcings = batch.get_forcings_at_time(t)
        known_forcing = batch.get_known_forcings_at_time(t)
        inputs = merge(forcings, prognostics)
        apparent_sources = model(inputs)
        prognostics = predict_one_step(prognostics, apparent_sources,
                                       known_forcing, time_step)
        yield t + 1, prognostics
