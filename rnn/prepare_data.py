import xarray as xr
from lib import thermo
from lib.torch.datasets import XRTimeSeries


def load_data(paths):
    data = {}
    for info in paths:
        for field in info['fields']:
            ds = xr.open_dataset(info['path'], chunks={'time': 10})
            data[field] = ds[field]

    # compute layer mass from stat file
    rho = data.pop('RHO')[0]
    rhodz = thermo.layer_mass(rho)

    data['layer_mass'] = rhodz

    TABS = data.pop('TABS')
    QV = data.pop('QV')
    QN = data.pop('QN', 0.0)
    QP = data.pop('QP', 0.0)

    sl = thermo.liquid_water_temperature(TABS, QN, QP)
    qt = QV + QN

    data['sl'] = sl
    data['qt'] = qt

    objects = [
        val.to_dataset(name=key).assign(x=sl.x, y=sl.y)
        for key, val in data.items()
    ]
    return xr.merge(objects, join='inner').sortby('time')


def get_dataset(paths, post=None, **kwargs):
    # paths = yaml.load(open("config.yaml"))['paths']
    ds = load_data(paths)
    if post is not None:
        ds = post(ds)
    ds = ds.load()
    return XRTimeSeries(ds, [['time'], ['x', 'y'], ['z']])
