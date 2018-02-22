from xnoah import swap_coord
import xarray as xr
import glob, os
from tqdm import tqdm

def load_dir(d):
    iop = xr.open_dataset(d + '/iop.nc')
    cam = xr.open_dataset(d + '/cam.nc')

    cam = cam.assign(x=cam.lon*0 + iop.x,
                     y=cam.lat*0 + iop.y)

    return swap_coord(cam, {'lon': 'x', 'lat': 'y'})


def get_dirnames(files):
    return set(os.path.dirname(f) for f in files)

dirs = get_dirnames(snakemake.input)
ds = xr.concat([load_dir(d) for d in tqdm(dirs)], dim='x').sortby('x')


# common calculations
ds = ds.rename({'lev': 'p'})
ds['qt'] = ds.Q * 1000
ds['sl'] = ds['T'] + ds.Z3 * 9.81/1004
ds['prec'] = (ds.PRECC + ds.PRECL) * 86400*1000

ds.to_netcdf(snakemake.output[0])

