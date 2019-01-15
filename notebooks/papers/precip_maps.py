import xarray as xr

from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap


def get_data():
    net_precip = runs['debias'].data_2d.NPNN
    ds = open_data('training')
    evap = lhf_to_evap(ds.LHF)
    net_precip_truth = ds.Prec - evap

    time = net_precip.time[0] + 2

    return xr.Dataset({
        'NG-Aqua': net_precip_truth.interp(time=time),
        'NN': net_precip.interp(time=time),
    }).compute()


def plot(data):
    data.to_array(dim='Run').plot(col='Run', vmax=100, vmin=-10)


plot(get_data())
