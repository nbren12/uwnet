from .tensordict import TensorDict
from .timestepper import Batch, TimeStepper
from .testing import assert_tensors_allclose
import torch


def _get_batch():
    n = 10
    array = torch.arange(n).view(1, -1)
    name = 'a'

    data = {name: array}
    return Batch(data, prognostics=[name]), name, array


def test_Batch_get_prognostics_at_time():
    batch, name, array = _get_batch()
    t = 1
    out = batch.get_prognostics_at_time(t)

    assert out[name].item() == array[0, t].item()


def test_TimeStepper():
    n_batch = 1
    n = 10
    array = torch.arange(n).view(1, -1).double()
    name = 'a'
    forcing_name = 'F' + name

    data = Batch(TensorDict({name: array, forcing_name: array * 0}),
                 prognostics=[name])
    one = TensorDict({name: torch.ones(n_batch, dtype=array.dtype)})

    def _mock_source(x):
        """Step by one each time"""
        return one

    stepper = TimeStepper(_mock_source, time_step=86400)
    output = stepper(data)

    assert output[name].shape == (1, n)
    assert_tensors_allclose(output['a'][:, :-1], array[:, 1:])
