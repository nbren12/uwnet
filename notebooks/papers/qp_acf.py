from src.data import open_data
import common
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def autocorr(x, dim, base_state_dims=(), avg_dims=(), n=None):

    base_state_dims = (dim,) + tuple(base_state_dims)
    avg_dims = base_state_dims + tuple(avg_dims)
    
    if n is None:
        n = len(x[dim])
    lags = np.arange(1, n//2)

    time = x[dim]
    # assume constant spacing
    dt = float(time[1]-time[0])
    
    x = x - x.mean(base_state_dims)
    
    denom = (x*x).sum(avg_dims)
    
    corrs = []
    for lag in lags:
        print("Lag", lag)
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
    return autocorr(x, 'time', avg_dims=['y', 'z'], base_state_dims=['x'], n=20)


def get_data():
    ds = open_data('training')
    variables = ['QT', 'QP', 'SLI', 'U', 'V']
    return ds[variables].apply(compute_acf)

def get_data_cached():
    path = "acf.nc"
    try:
        return xr.open_dataset(path)
    except FileNotFoundError:
        data = get_data()
        data.to_netcdf(path)
        return data


def plot(data):
    df = data.to_dataframe()

    row = {key: [1.0] for key in df.columns}
    row_df = pd.DataFrame(row, index=[0.0])
    df = row_df.append(df)

    width = common.textwidth/2
    figsize = (width, width/1.4)
    df.plot(marker='o', figsize=figsize)
    plt.xlim(left=0)
    plt.xlabel('Lag [day]')
    plt.ylabel('Auto-correlation')
    plt.ylim([0, 1.15])
    plt.legend(frameon=False, loc="upper right", ncol=2)
    plt.xlabel('Lag [day]')
    plt.tight_layout()

    

data = get_data_cached()
plot(data)
plt.savefig("qp_acf.pdf")
