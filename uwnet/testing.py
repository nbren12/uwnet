import numpy as np


def assert_tensors_allclose(*args):
    args = [arg.detach().numpy() for arg in args]
    return np.testing.assert_allclose(*args)
