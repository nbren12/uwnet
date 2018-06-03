import xarray as xr
from lib.thermo import liquid_water_temperature
from tqdm import tqdm


def compute_tend_single(x):
    t = x.time[0:1]
    return (x.diff('time')/x.time.diff('time')).assign_coords(time=t)

ncs = snakemake.input
out_path = snakemake.output[0]

times = []
fqt = []
fsl = []
for file in tqdm(ncs):
    ds = xr.open_dataset(file)

    qt = ds.QV
    sl = liquid_water_temperature(ds.TABS, 0.0, 0.0)

    fqt.append(compute_tend_single(qt))
    fsl.append(compute_tend_single(sl))



xr.Dataset({
    'FQT': xr.concat(fqt, dim='time'),
    'FSL': xr.concat(fsl, dim='time')
}).to_netcdf(out_path)

