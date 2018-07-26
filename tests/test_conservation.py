import xarray as xr
import torch
from uwnet.interface import step_model
from uwnet.model import MLP


def lhf_to_evap(lhf):
    return lhf * 86400 / 2.51e6


def precipitable_water(qv, layer_mass, dim='p'):
    return (qv * layer_mass).sum(dim)


# water budget stuff
def water_budget(ds, dim='p'):
    """Compute water budget if Q2NN is present"""
    q2_int = (ds.Q2NN * ds.layer_mass).sum(dim) * 86400
    q2_ib = q2_int
    prec = ds.Prec
    evap = lhf_to_evap(ds.LHF)

    return xr.Dataset(
        dict(Prec=prec, evap=evap, Q2=q2_ib, imbalance=q2_ib - (evap - prec)))


def to_normal_units_nn_columns(ds):
    """Convert output from uwnet.columns to have proper SI units"""
    scales = {
        'FQT': 1 / 86400 / 1000,
        'FSL': 1 / 86400,
        'Q1NN': 1 / 86400 / 1000,
        'qt': 1 / 1000,
        'qtOBS': 1 / 1000,
        'Q2NN': 1 / 86400 / 1000,
    }

    for key, scale in scales.items():
        if key in ds:
            ds = ds.assign(**{key: ds[key] * scale})

    return ds


def ds_to_np_dict(ds):
    out = {}
    for key in ds:
        out[key] = ds[key].values

    for key in 'SST SOLIN'.split():
        out[key] = out[key][None]

    return out


def output_to_ds(out, coords):
    out_vars = [
        ('LHF', ['m', 'y', 'x']),
        ('RADTOA', ['m', 'y', 'x']),
        ('RADSFC', ['m', 'y', 'x']),
        ('Prec', ['m', 'y', 'x']),
        ('SHF', ['m', 'y', 'x']),
        ('qt', ['z', 'y', 'x']),
        ('sl', ['z', 'y', 'x']),
        ('Q1NN', ['z', 'y', 'x']),
        ('Q2NN', ['z', 'y', 'x']),
    ]

    data_vars = {name: (dims, out[name]) for name, dims in out_vars}
    return xr.Dataset(data_vars, coords=coords).squeeze('m')


# ds = xr.open_zarr("all.1.zarr").pipe(to_normal_units_nn_columns)
# ds0 = ds.isel(time=0)
# ds0.to_netcdf('ds0.nc')
ds0 = xr.open_dataset('ds0.nc')

model = MLP.from_dict(torch.load("13_actual_constraint/5.pkl")['dict'])
step = model.step

x = ds_to_np_dict(ds0)
x['dt'] = 86400  # d
out = step_model(step, **x)
out_ds = output_to_ds(out, ds0.coords)
out_ds['layer_mass'] = ds0.layer_mass

h20b = water_budget(out_ds, dim='z')

# pw before and after
def domain_pw(qt, w):
    return float(precipitable_water(qt, w, 'z').mean(['x', 'y']))

pw0 = domain_pw(ds0['qt'], ds0.layer_mass)
pw1 = domain_pw(out_ds['qt'], ds0.layer_mass)

print("Change in Domain mean PW", float(pw1-pw0)/x['dt']*86400, "mm/day")
net_evap = (h20b.evap - h20b.Prec).mean(['x', 'y'])
print("P-E domain mean", float(net_evap), " mm/day")

# heat before and after
heat0 = domain_pw(ds0['sl'], ds0.layer_mass) * 1004
heat1 = domain_pw(out_ds['sl'], ds0.layer_mass) * 1004

mean = out_ds.mean(['x', 'y'])
net_heating = float(mean.Prec * 2.51e6/ 86400 + mean.SHF + mean.RADSFC-
                    mean.RADTOA)

print("Change in heat", (heat1-heat0)/x['dt'], "W/m^2")
print("Lv P + SHF", net_heating, "W/m2")

mean = h20b.mean().load()
imbalance = float(mean.imbalance)

if abs(imbalance) > 1e-3:
    raise ValueError(mean)

