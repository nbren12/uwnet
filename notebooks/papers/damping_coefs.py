import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


from src.data import training_data, open_data
from uwnet.thermo import compute_apparent_source
from common import textwidth


def compute_damping_coefficient(x, src, dims=['x', 'time']):
    denom = (x * x).sum(dims)
    return -(x * src).sum(dims) / denom.where(denom > 1e-6)


def plot_pres_vs_y(data, pres, ax, fig, title=''):
    minimum, maximium = float(data.min()), float(data.max())
    levels = np.arange(10)
    levels = [level for level in levels if minimum <= level <= maximium]
    if len(levels) < 5:
        levels = None

    im = ax.contourf(data.y/1e6, pres, data, levels=levels, extend='both')
    fig.colorbar(im, ax=ax, orientation='horizontal')

    ax.invert_yaxis()
    ax.set_ylabel('p (mbar)')
    ax.set_xlabel('y (1000 km)')
    ax.set_title(title)


def plot(data):
    q_damping, t_damping, pres, df = data
    fig, (ax, axT) = plt.subplots(
        1,
        2,
        figsize=(textwidth, textwidth / 3),
        constrained_layout=True)
    plot_pres_vs_y(
        q_damping, pres, ax, fig, title='QT damping (1/day)')

    qt_dyn_label = r'$\left( \frac{\partial q_T}{\partial t} \right)_{GCM}$'
    keys = ['QT', 'FQT']
    df['FQT'] *= 86400
    labels = [r'$q_T$ [g/kg]', qt_dyn_label + ' [g/kg/d]']
    plot_twin(df, keys, labels,  axT)

    ax.set_title('a) Damping (1/day)')
    axT.set_title(r'b) $q_T$ and ' + qt_dyn_label)


def plot_twin(df, keys, labels, a):

    c1, c2 = ['b', 'y']

    x = df.index
    y1 = df[keys[0]]
    y2 = df[keys[1]]

    lab1, lab2 = labels

    l1, = a.plot(x, y1, c=c1)
    a_twin = a.twinx()
    l2, = a_twin.plot(x, y2, c=c2, alpha=.9)

    a_twin.set_ylabel(lab2, color=c2)
    a.set_ylabel(lab1, color=c1)


def get_data():

    # get data
    ds = xr.open_dataset(training_data, chunks={'time': 20})
    sources = dict(
        q1=compute_apparent_source(ds.SLI, 86400 * ds.FSLI),
        q2=compute_apparent_source(ds.QT, 86400 * ds.FQT))

    ds = ds.assign(**sources)
    pres = ds.p[0]

    ds = ds.isel(time=slice(0, 100))

    q_damping = compute_damping_coefficient(ds.QT, ds.q2).compute()
    t_damping = compute_damping_coefficient(ds.SLI, ds.q1).compute()
    return (q_damping, t_damping, pres, get_time_series_data())


def get_time_series_data():
    ds = open_data('training')
    index = {'x': 0, 'y': 32, 'z': 20}
    location = ds.isel(**index)
    return location.to_dataframe()


plot(get_data())
plt.savefig("damping.pdf")