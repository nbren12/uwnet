import config
import glob
import torch
from toolz import *
import pandas as pd
import re
import numpy as np
import zarr
import xarray as xr


def get_paths(case):
    return f"{config.sam}/OUT_3D/{case}_*.nc"


def get_debugging(case):
    files = list(glob.glob(f"{config.sam}/OUT_3D/{case}_*.pt"))
    # sort by step
    def key(x):
        return int(re.search(r"(\d+).pt$", x).group(1))
    return sorted(files, key=key)

    
def open_case(*args):
    return xr.open_mfdataset(get_paths(*args), concat_dim='time')


def get_debug_step(i, case="NG1_2018-07-18_5.dbg"):
    path = config.sam  + f"OUT_3D/{case}_{i}.pt"
    return torch.load(path)


def get_pw(args):
    kw, out = args
    
    kw = budget.add_units(kw)
    out = budget.add_units(out)
    
    
    fqt = budget.path(kw['FQT'], kw['layer_mass'])/budget.density_water
    evap = out['LHF']/budget.Lv / budget.density_water
    prec = out['Prec']
    
    
    vals = pw.mean().to("mm"), fqt.mean().to("mm/day"), evap.mean().to("mm/day"), prec.mean().to("mm/day")
    keys= 'PW SAM Evap Prec'.split()
    
    return dict(zip(keys, vals))


def argmax(x):
    """Return indices of maximum"""
    shape = x.shape
    i_flat = x.argmax()
    return np.unravel_index(i_flat, shape)


def h20_budget(case):
    return merge_with(np.stack, [get_pw(get_debug_step(i, case=case)) for i in range(20, 140, 20)])


def plot_h20_budget(case):
    b = h20_budget(case)
    pd.DataFrame(b).plot()
    
    
def get_var(field, dims, root):
    
    def get_dim(dim):
        try:
            return root[dim][:]
        except KeyError:
            return 
    
    coords = {dim : root[dim][:] for dim in dims if dim in root}
    return xr.DataArray(root[field], dims=dims, coords=coords, name=field)

def get_var3d(field, root):
    x = get_var(field, ['day', 'p', 'y', 'x'], root)
    return x


def get_var2d(field, root):
    x = get_var(field, ['day', 'm', 'y', 'x'], root)
    return x.isel(m=0)

def plot_2d_snaps(x, n, **kw):
    x.isel(day=slice(0,None, n)).plot(col='day', col_wrap=5, **kw)
    
def plot_p_vs_t(x, title='', **kw):
    x.plot(y='p', **kw)
    plt.title(title)
    return plt.ylim([1015, 10])


def zarr_to_xr(root):
    
    vars_3d = ['U', 'V', 'W', 'qt', 'sl', 'FQT', 'FSL', 'Q1NN', 'Q2NN']
    vars_2d = ['Prec', 'LHF']
    
    data_vars = {}
    for f in vars_3d:
        data_vars[f] = get_var3d(f, root)
        
    for f in vars_2d:
        data_vars[f] = get_var2d(f, root)
        
    data_vars['layer_mass'] = get_var('layer_mass', ['p'], root)
    
    return xr.Dataset(data_vars)

def open_debug_zarr_as_xr(path):
    ds = zarr.open_group(path)
    return zarr_to_xr(ds)


# water budget stuff
def water_budget(ds):
    """Compute water budget if Q2NN is present"""
    q2_int = (ds.Q2NN * ds.layer_mass).sum('p') * 86400
    q2_ib = q2_int
    
    q_ls = (ds.FQT * ds.layer_mass).sum('p') * 86400
    prec = ds.Prec
    evap = lhf_to_evap(ds.LHF)
    
    return xr.Dataset(dict(Prec=prec, evap=evap, QLS=q_ls, Q2=q2_ib,
                           imbalance=q2_ib-(evap -prec)))


def lhf_to_evap(lhf):
    return lhf* 86400 / 2.51e6


def precipitable_water(qv, layer_mass):
    return (qv * layer_mass).sum('p')


def water_budget_from_obs(obs):
    """Compute the water budget if Q2NN is not present"""

    pw = precipitable_water(obs.qt, obs.layer_mass)
    
    ds =  xr.Dataset(dict(
        evap = lhf_to_evap(obs.LHF),
            fqt_i = precipitable_water(obs.FQT, obs.layer_mass) *86400, #mm/day
            prec = obs.Prec,
            storage = pw.diff('time')/pw.time.diff('time')))
    
    # imbalance
    ds['imbalance'] = ds.storage - (ds.fqt_i + ds.evap - ds.prec)
    return ds


def to_normal_units_nn_columns(ds):
    """Convert output from uwnet.columns to have proper SI units"""
    scales = {'FQT': 1/86400 / 1000, 
              'FSL': 1/86400,
              'Q1NN': 1/86400/1000,
              'qt': 1/1000,
              'qtOBS': 1/1000,
              'Q2NN': 1/86400/1000,
             }
    
    for key, scale in scales.items():
        if key in ds:
            ds = ds.assign(**{key: ds[key]*scale})
    
    return ds


def moisture_budget_plots(h20b):
    import holoviews as hv
    
    def make_overlay(x):
    
        return  hv.Dataset(x.to_array(name='water'))\
              .to.curve()\
              .overlay()
    
    
    hmap = hv.HoloMap({
        'Domain Mean': make_overlay(h20b.sel(time=slice(100, 140)).mean(['x', 'y'])),
        'Location': make_overlay(h20b.isel(x=0,y=32)),
        'Zonal Mean': make_overlay(h20b.isel(time=slice(1,10)).mean(['x', 'time']))

    })
    
    opts ={
        'Curve': dict(norm=dict(framewise=True, axiswise=True),
                      plot=dict(width=400), 
                      ),
        'NdOverlay.II': {'plot': dict(show_legend=False)},
        'NdOverlay.I': {'plot': dict(show_legend=False)},
    }
    
    return hmap.layout().cols(2).opts(opts)