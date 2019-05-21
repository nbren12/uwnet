from src.data import open_data
import common
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def autocorr(x, dim, base_state_dims=(), avg_dims=()):

    base_state_dims = (dim,) + tuple(base_state_dims)
    avg_dims = base_state_dims + tuple(avg_dims)
    
    n = len(x[dim])
    lags = np.arange(-n//2, n//2)

    time = x[dim]
    # assume constant spacing
    dt = float(time[1]-time[0])
    
    x = x - x.mean(base_state_dims)
    
    denom = (x*x).sum(avg_dims)
    
    corrs = []
    for lag in lags:
        shift = {dim: lag}
        xs = x.shift(**shift)
        corr = (xs * x).dropna(dim).sum(avg_dims)/denom
        corrs.append(corr)
        
    lags = xr.DataArray(lags * dt, name='lag', attrs={'units': 'day'}, dims=['lag'])
        
    return xr.concat(corrs, dim=lags)


def autocorr_time(*args):
    acf = autocorr(*args)
    return acf.where(acf.lag > 0).fillna(0.0).integrate('lag')


def compute_acf(x):
    return autocorr(x, 'time', avg_dims=['y', 'z'], base_state_dims=['x'])


def get_data():
    ds = open_data('training')
    variables = ['QT', 'QP', 'SLI', 'U', 'V']
    return ds[variables].apply(compute_acf)


def plot(data):
    data.to_dataframe().plot()
    plt.xlim(left=0)

    

data = get_data()
plot(data)
plt.savefig("qp_acf.pdf")