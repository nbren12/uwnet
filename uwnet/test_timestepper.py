from .timestepper import Batch
import torch


def test_Batch_get_prognostics_at_time():
    n = 10
    array = torch.arange(n).view(1, -1)
    name = 'a'

    data = {name: array}
    batch = Batch(data, prognostics=[name])

    t = 1
    out = batch.get_prognostics_at_time(t)

    assert out[name].item() == array[0, t].item()
