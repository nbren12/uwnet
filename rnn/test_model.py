import torch
from .model import SimpleLSTM
from .train import select_time, get_batch_size


def _mock_batch(n, nt, nz):
    return {
        'LHF': torch.zeros(n, nt),
        'SHF': torch.zeros(n, nt),
        'SOLIN': torch.zeros(n, nt),
        'qt': torch.zeros(n, nt, nz),
        'sl': torch.zeros(n, nt, nz),
        'FQT': torch.zeros(n, nt, nz),
        'FSL': torch.zeros(n, nt, nz),
        'p': torch.zeros(nz),
    }

def test_model():
    scaler = lambda x: x 

    lstm = SimpleLSTM(scaler)
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
    scaler = lambda x: x 

    lstm = SimpleLSTM(scaler)
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
    scaler = lambda x: x
    lstm = SimpleLSTM(scaler)
    hid = lstm.init_hidden(11)

    assert hid[0].size() == (11, 256)
    assert hid[1].size() == (11, 256)


def test_select_time():
    batch = _mock_batch(100, 10, 11)
    ibatch = select_time(batch, 0)

    assert ibatch['sl'].size() == (100, 11)
    assert ibatch['SHF'].size() == (100, )
    assert ibatch['p'].size() == (11,)


def test_get_batch_size():
    batch = _mock_batch(100, 10, 11)
    assert get_batch_size(batch) == 100
