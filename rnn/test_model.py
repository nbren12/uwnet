import numpy as np
import torch
from .model import SimpleLSTM, MLP
from .utils import select_time, get_batch_size, stack_dicts


def _mock_batch(n, nt, nz, init=torch.rand):
    return {
        'LHF': init(n, nt),
        'SHF': init(n, nt),
        'SOLIN': init(n, nt),
        'qt': init(n, nt, nz),
        'sl': init(n, nt, nz),
        'FQT': init(n, nt, nz),
        'FSL': init(n, nt, nz),
        # 'p': init(nz),
    }

def test_model():
    lstm = SimpleLSTM({}, {})
    n = 10
    nt = 100
    nz = 34

    batch = {
        'LHF': torch.zeros(n),
        'SHF': torch.zeros(n),
        'SOLIN': torch.zeros(n),
        'qt': torch.zeros(n, nz),
        'sl': torch.zeros(n, nz),
        'FQT': torch.zeros(n, nz),
        'FSL': torch.zeros(n, nz),
    }

    hid = lstm.init_hidden(n)
    out, hid = lstm(batch, hid)
    assert set(['sl', 'qt']) == set(out.keys())
    assert out['sl'].size() == (n, nz)
    assert out['qt'].size() == (n, nz)

def test_tbtt():

    lstm = SimpleLSTM({}, {})
    n = 10
    nt = 100
    nz = 34

    batch = {
        'LHF': torch.zeros(n, nt),
        'SHF': torch.zeros(n, nt),
        'SOLIN': torch.zeros(n, nt),
        'qt': torch.zeros(n, nt, nz),
        'sl': torch.zeros(n, nt, nz),
        'FQT': torch.zeros(n, nt, nz),
        'FSL': torch.zeros(n, nt, nz),
    }


    n = get_batch_size(batch)
    hid = lstm.init_hidden(n)
    output = []
    for t in range(nt):
        pred, hid = lstm(select_time(batch, t), hid)
        output.append(pred)


def test_init_hidden():
    lstm = SimpleLSTM(None, None)
    hid = lstm.init_hidden(11)

    assert hid[0].size() == (11, 256)
    assert hid[1].size() == (11, 256)


def test_select_time():
    batch = _mock_batch(100, 10, 11)
    ibatch = select_time(batch, 0)

    assert ibatch['sl'].size() == (100, 11)
    assert ibatch['SHF'].size() == (100, )


def test_get_batch_size():
    batch = _mock_batch(100, 10, 11)
    assert get_batch_size(batch) == 100


def test_stack_dicts():
    batches = _mock_batch(3, 4, 5)

    seq = [select_time(batches, i) for i in range(4)]
    out = stack_dicts(seq)

    assert out['sl'].size() == batches['sl'].size()
    print(out.keys())
    for key in out:
        print(key)
        np.testing.assert_allclose(out[key].numpy(), batches[key].numpy())


def test_mlp_forward():
    batch = _mock_batch(3, 4, 34)

    mlp = MLP({}, {})

    prog = select_time(batch, 0)
    qt_b = batch.pop('qt')
    batch.pop('sl')

    pred = mlp(prog, batch)

    assert pred['sl'].size() == qt_b.size()
