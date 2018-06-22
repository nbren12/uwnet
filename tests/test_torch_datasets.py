import xarray as xr
import numpy as np
from lib.torch.datasets import WindowedData, XRTimeSeries

def test_windowed_dataset():
    arr = np.arange(4).reshape((4, 1, 1, 1))
    data = WindowedData(arr, chunk_size=3)


    assert data.reshaped.shape == (4, 1, 1)
    assert len(data) == 2

    assert data[0][:, 0].tolist() == [0, 1, 2]
    assert data[1][:, 0].tolist() == [1, 2, 3]


