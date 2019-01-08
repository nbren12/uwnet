import matplotlib.pyplot as plt
import xarray as xr

from src.data import training_data
from uwnet.thermo import compute_apparent_source
from common import textwidth


def compute_damping_coefficient(x, src, dims=['x', 'time']):
    denom = (x * x).sum(dims)
    return -(x * src).sum(dims) / denom.where(denom > 1e-6)


def plot_pres_vs_y(data, pres, ax, fig, title=''):
    minimum, maximium = float(data.min()), float(data.max())
    levels = [data.min(), 0, .001, .1, 1, 2, 4, 8, 10]
    levels = [level for level in levels if minimum <= level <= maximium]
    if len(levels) < 5:
        levels = None

    im = ax.contourf(data.y, pres, data, levels=levels)
    fig.colorbar(im, ax=ax)

    ax.invert_yaxis()
    ax.set_ylabel('p (mbar)')
    ax.set_xlabel('y (m)')
    ax.set_title(title)


def plot(data):
    q_damping, t_damping, pres = data
    fig, (ax, axT) = plt.subplots(
        1,
        2,
        sharex=True,
        figsize=(textwidth, textwidth / 3),
        constrained_layout=True)
    plot_pres_vs_y(
        q_damping, pres, ax, fig, title='QT damping (1/day)')
    plot_pres_vs_y(
        t_damping, pres, axT, fig, title='SLI damping (1/day)')


def get_data():

    # get data
    ds = xr.open_dataset(training_data, chunks={'time': 20}).isel(step=0)

    sources = dict(
        q1=compute_apparent_source(ds.SLI, 86400 * ds.FSLI),
        q2=compute_apparent_source(ds.QT, 86400 * ds.FQT))

    ds = ds.assign(**sources)
    pres = ds.p[0]

    ds = ds.isel(time=slice(0, 100))

    q_damping = compute_damping_coefficient(ds.QT, ds.q2)
    t_damping = compute_damping_coefficient(ds.SLI, ds.q1)
    return (q_damping, t_damping, pres)


plot(get_data())
