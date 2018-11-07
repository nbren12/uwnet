from .loss import compute_multiple_step_loss, weighted_mean_squared_error
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


@pytest.mark.parametrize('x,w,dim,expected', [
    (torch.rand(10), torch.rand(10), -1, 0.0),
    (torch.rand(10, 5), torch.rand(5), -1, 0.0),
    (torch.rand(5, 10), torch.rand(5), 0, 0.0),
    (torch.rand(4, 5, 6), torch.rand(5), 1, 0.0),
])
def test_weighted_mean_squared_error(x, w, dim, expected):
    loss = weighted_mean_squared_error(x, x, w, dim)
    assert loss.dim() == 0
    assert loss.item() == pytest.approx(expected)


def test_weighted_mean_squared_error_value():
    """Test the MSE function for a non-zero value"""
    a = 2
    x = torch.rand(5)
    y = x + a
    w = torch.rand(5)
    expected = a**2 * w.mean()
    loss = weighted_mean_squared_error(x, y, w)
    assert loss.item() == pytest.approx(expected.item())
