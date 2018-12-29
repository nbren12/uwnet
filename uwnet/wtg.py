import torch
from toolz import assoc, curry
import attr

from .jacobian import jacobian_backward, jacobian_norm, jacobian, max_signed_eigvals


def step_with_model(rhs, state, dt=.125, n=100):
    """Perform a number of time steps with a model"""
    for t in range(n):
        qt_dot = rhs(state)
        new_qt = state['QT'] + dt * qt_dot['QT']
        state = assoc(state, 'QT', new_qt)
        yield state


def centered_difference(x, dim=-3):
    """Compute centered difference with first order difference at edges"""
    x = x.transpose(dim, 0)
    dx = torch.cat(
        [
            torch.unsqueeze(x[1] - x[0], 0), x[2:] - x[:-2],
            torch.unsqueeze(x[-1] - x[-2], 0)
        ],
        dim=0)
    return dx.transpose(dim, 0)


def compute_stability_ratio(qt, sli, z):
    dsli = centered_difference(sli) / centered_difference(z)
    dqt = centered_difference(qt) / centered_difference(z)
    lam = dqt / dsli
    return lam


@curry
def model_wtg_interactive(model, z, input):
    lam = compute_stability_ratio(input['QT'], input['SLI'], z)
    return model_wtg(model, lam, input)


@curry
def model_wtg(model, lam, input):
    """The WTG model is

    q_t = Q2 - Q1 lambda

    where lambda = q_z/s_z, or some approximation thereof.

    """
    y = model(input)
    qt_dot = y['QT'] - y['SLI'] * lam
    return {'QT': qt_dot}


def wtg_penalty(model, z, batch, dt=.125, n=20, max_loss=100.0):
    """Compute the WTG penality

    Parameters
    ----------
    dt : float
        time step to use
    n : int
        Number of iterations
    max_loss : float
        Stop iteration if the loss exceeds this amount. (default: 100.0)

    """
    from random import randint
    i = randint(0, batch.num_time - 1)
    j = randint(0, batch.size - 1)

    input = batch.get_model_inputs(i)

    # get random batch member
    input = input.apply(lambda x: x[j])
    
    nz, nx, ny = input['QT'].shape
    x = randint(0, nx-1)
    y = randint(0, ny-1)
    
    
    qt_mean = batch.get_time_mean('QT')[j]
    sli_mean = batch.get_time_mean('SLI')[j]
    z = z.view(-1, 1, 1)
    lam = compute_stability_ratio(qt_mean, sli_mean, z)
    lam[:5]= lam[5]
    rhs = model_wtg(model, lam)

    input['QT'].requires_grad = True
    y = rhs(input)
    dqt = y['QT']
    qt = input['QT']
    A = jacobian(dqt, qt).squeeze()
    return max_signed_eigvals(A, niter=10)

#     last_loss = 0.0
#     for state in step_with_model(rhs, input, dt=dt, n=n):
#         loss = criterion(qt_mean, state['QT'])
#         if loss.item() > 100.0:
#             return loss
#         last_loss = loss
    return loss
