"""

Step Types:

- instability
- multi

"""
import torch
from toolz import curry
from torch.nn.functional import mse_loss

from uwnet.utils import mean_other_dims
from .timestepper import predict_multiple_steps, TimeStepper, Batch
from . import tensordict


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


def instability_penalty_step(self, engine, data, alpha=1.0):
    # TODO pass the list of prognostics to step
    batch = Batch(data.float(), prognostics=['QT', 'SLI'])
    self.optimizer.zero_grad()

    dt = .125
    pred, x1 = get_input_output(self.model, dt, batch)

    l1 = compute_loss(self.criterion, x1, pred)
    l2 = equilibrium_penalty(self.criterion, self.model, batch, dt)

    loss = l1 + alpha * l2

    info = {
        'Q1/Q2': l1.item(),
        'equilibrium': l2.item(),
        'total': loss.item(),
    }

    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    engine.state.loss_info = info
    return pred, x1


def multiple_step_loss_step(self, engine, batch):
    self.optimizer.zero_grad()
    stepper = TimeStepper(self.model, time_step=self.time_step)

    prediction = stepper(batch)

    # TODO: maybe TimeStepper should return a batch object
    prediction = tensordict.lag(prediction, -1, batch.time_dim)
    truth = batch.data_for_lag(1)

    loss = ((truth - prediction)**2)
    combined_var_loss = sum(loss.values())
    loss = combined_var_loss[combined_var_loss < 100.0].mean()

    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    info = {"loss": loss.item()}

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
    # residual forcing
    residual = (x1-x0)/dt - 86400 * forcing
    return src, residual


def get_step(name, self, kwargs):
    if name == 'instability':
        fun = instability_penalty_step
    elif name == 'multi':
        fun = multiple_step_loss_step
    else:
        raise NotImplementedError(
            f"Training step type '{name}' is not implemented")

    return curry(fun, self, **kwargs)
