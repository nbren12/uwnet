"""

Step Types:

- instability
- multi

"""
import torch
from toolz import curry
from torch.nn.functional import mse_loss

from uwnet.utils import mean_other_dims
from .timestepper import predict_multiple_steps


def r2_score(truth, prediction):
    return weighted_r2_score(truth, prediction, 1.0, dim=None)


def weighted_r2_score(truth, prediction, weights, dim=-1):
    """Compute the weighted R2 score"""
    mean = mean_other_dims(truth, dim)
    squares = weighted_mean_squared_error(truth, mean, weights, dim)
    residuals = weighted_mean_squared_error(truth, prediction, weights, dim)
    return 1 - residuals / squares


@curry
def mse_with_integral(truth, prediction, weights, dim=-3):
    ctruth = (truth * weights).sum(dim) / 1000
    cpred = (prediction * weights).sum(dim) / 1000

    l1 = mse_loss(ctruth, cpred)
    l2 = weighted_mean_squared_error(truth, prediction, weights, dim)

    return l1 + l2


@curry
def weighted_mean_squared_error(truth, prediction, weights=1.0, dim=-1):
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


def select_keys_time(x, keys, t):
    return {key: x[key][t] for key in keys}


def compute_loss(criterion, prognostics, y):
    return sum(criterion(prognostics[key], y[key]) for key in prognostics)


def compute_multiple_step_loss(criterion, model, batch, *args, **kwargs):
    """Compute the loss across multiple time steps with an Euler stepper

    Yields
    ------
    t: int
       the time step of the prediction
    prediction: dict
       the predicted state

    """
    prediction_generator = predict_multiple_steps(model, batch, *args,
                                                  **kwargs)
    return sum(
        compute_loss(criterion, prediction, batch.get_prognostics_at_time(t))
        for t, prediction, _ in prediction_generator)


def equilibrium_penalty(criterion, model, batch, dt, n=20):
    from random import randint
    i = randint(0, batch.num_time - 1)

    state0 = batch.get_prognostics_at_time(i)
    mean = batch.get_prognostics().apply(lambda x: x.mean(dim=1))
    g = batch.get_known_forcings()
    mean_forcing = g.apply(lambda x: x.mean(dim=1))
    state = state0

    for t in range(n):
        inputs = batch.get_model_inputs(i, state)
        src = model(inputs)
        state = state + dt * src + dt * mean_forcing * 86400

    return compute_loss(criterion, mean, state)


def total_loss(criterion, model, z, batch, time_step=.125):
    """Compute the loss across multiple time steps with an Euler stepper
    """
    dt = time_step
    pred, x1 = get_input_output(model, dt, batch)

    l1 = compute_loss(criterion, x1, pred)
    l2 = equilibrium_penalty(criterion, model, batch, dt)

    loss = l1 + l2
    loss_info = {
            'Q1/Q2': l1.item(),
            'equilibrium': l2.item(),
            'total': loss.item(),
        }

    return loss, loss_info, (pred, x1)


def instability_penalty_step(self, engine, batch):
    self.optimizer.zero_grad()
    loss, info, (y_pred, y) = total_loss(self.criterion, self.model, self.z, batch)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    engine.state.loss_info = info
    return y_pred, y


def multiple_step_loss_step(self, engine, batch):
    self.optimizer.zero_grad()

    nt = batch.num_time - 1
    loss = compute_multiple_step_loss(
        self.criterion, self.model, batch, 0, nt, self.time_step)
    info = {'multiple': loss.item()}

    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    engine.state.loss_info = info
    return info


@curry
def get_input_output(model, dt, batch):
    src = model(batch.data)

    g = batch.get_known_forcings()
    progs = batch.get_prognostics()

    forcing = g.apply(lambda x: (x[:, 1:] + x[:, :-1]) / 2)
    x0 = progs.apply(lambda x: x[:, :-1])
    x1 = progs.apply(lambda x: x[:, 1:])
    src = src.apply(lambda x: x[:, :-1])

    # pred = x0 + dt * src + dt * 86400 * forcing
    # return pred, x1

    # pred = x0 + dt * src + dt * 86400 * forcing
    return src, (x1-x0)/dt - 86400 * forcing


def get_step(step_type, _config):
    if step_type == 'instability':
        return instability_penalty_step
    elif step_type == 'multi':
        return multiple_step_loss_step
    else:
        raise NotImplementedError(
            f"Training step type '{step_type}' is not implemented")
