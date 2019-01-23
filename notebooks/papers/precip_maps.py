import xarray as xr

from src.data import open_data, runs
from uwnet.thermo import lhf_to_evap
from matplotlib import colors
import numpy as np



# Example of making your own norm.  Also see matplotlib.colors.
# From Joe Kington: This one gives two different linear ramps:
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


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
    norm = MidpointNormalize(midpoint=0, vmin=-20, vmax=50)
    data = data.assign_coords(x=data.x/1e6, y=data.y/1e6)
    grid = data.to_array(dim='Run').plot(col='Run', norm=norm, add_labels=False)

    grid.set_xlabels("x (1000 km)")
    grid.set_ylabels("y (1000 km)")

    grid.axes[0,0].set_title("a) NG-Aqua", loc='left')
    grid.axes[0,1].set_title("b) NN", loc='left')


plot(get_data())
