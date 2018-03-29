import os
from lib import scam
from lib.cam import load_cam
import xarray as xr

wildcards = snakemake.wildcards
output = snakemake.output
input = snakemake.input

RCE = snakemake.params.get('RCE', False)

i = int(wildcards.i)
j = int(wildcards.j)

loc = xr.open_dataset(input[0], chunks={'lon': 1, 'lat': 1})\
        .isel(lon=i, lat=j)

if RCE:
    print("Running in RCE Mode (homogeneous forcing in time)")
    loc = loc * 0 + loc.mean('tsec')

output_dir = os.path.dirname(output[0])
os.system(f"rm -rf {output_dir}")
os.system(f"mkdir -p {output_dir}")

scam.save_iop_dir(output_dir, loc)

os.system(f"ext/scam/run_docker.sh {output_dir} > /dev/null") 
load_cam(f"{output_dir}/camrun.cam.h0.*.nc")\
    .to_netcdf(output[0])
