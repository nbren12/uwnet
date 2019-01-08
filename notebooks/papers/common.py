import pandas as pd
import xarray as xr


def data_array_dict_to_dataset(d, dim='keys'):
    idx = pd.Index(d.keys(), name=dim)
    return xr.concat(d.values(), dim=idx)


textwidth = 6.5
