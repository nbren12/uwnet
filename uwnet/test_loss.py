from .loss import compute_multiple_step_loss
import torch
import pytest


def test_compute_multiple_step_loss():
    def criterion(x, y):
        return torch.abs(x - y).mean()

    def model(x):
        return {'x': 0.0}

    n = 10
    prognostics = ['x']
    batch = {'x': torch.zeros(n).float(), 'Fx': torch.zeros(n).float()}
    loss = compute_multiple_step_loss(criterion, model, batch, prognostics, 0,
                                      n - 1, 1.0)
    assert loss.item() == pytest.approx(0.0)

    def model(x):
        return {'x': 86400}

    batch = {'x': torch.arange(n).float(), 'Fx': torch.zeros(n).float()}
    loss = compute_multiple_step_loss(criterion, model, batch, prognostics, 0,
                                      n - 1, 1.0)
    assert loss.item() == pytest.approx(0.0)
