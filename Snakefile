import os
import sys
from os.path import join

## VARIABLES
DATA_PATH = "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX"
DATA_URL = "https://atmos.washington.edu/~nbren12/data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX.tar"
NUM_STEPS = config.get('NSTEPS', 640)
TRAINING_DATA = "data/processed/training.nc"
TROPICS_DATA = "data/processed/tropics.nc"


## Temporary output locations
SAM_PROCESSED = "data/tmp/{step}.nc"
SAM_PROCESSED_LOG = "data/tmp/{step}.log"
SAM_PROCESSED_ALL = "data/sam_processed.nc"

## Set environmental variables
# add 'bin' folder to PATH
os.environ['PATH'] = os.path.abspath('bin') + ':' + os.environ['PATH']

print("Number of steps to process:", NUM_STEPS)

## RULES
rule all:
    input: TROPICS_DATA

rule download_data:
    output: DATA_PATH
    shell: "cd data/raw && curl {DATA_URL} | tar xv"

rule concat_sam_processed:
    input: expand(SAM_PROCESSED, step=range(NUM_STEPS))
    output: SAM_PROCESSED_ALL
    shell:
        """
        echo {input} | ncrcat -o {output}
        ncatted -a units,FQT,c,c,'g/kg/s' \
        -a units,FSLI,c,c,'K/s' \
        -a units,FU,c,c,'m/s^2' \
        -a units,FV,c,c,'m/s^2' \
        -a units,x,c,c,'m' \
        -a units,y,c,c,'m' \
        {output}
        """

rule add_constant_and_2d_variables:
    input:
        data = DATA_PATH,
        sam  = SAM_PROCESSED_ALL
    output:
        TRAINING_DATA
    run:
        import xarray as xr
        from uwnet.thermo import layer_mass

        stat_path = join(input.data, 'stat.nc')
        twod_path = join(input.data, 'coarse', '2d', 'all.nc')

        # compute layer_mass
        stat = xr.open_dataset(stat_path)
        rho = stat.RHO.isel(time=0).drop('time')
        w = layer_mass(rho)

        ds = xr.open_dataset(input.sam)

        # get 2D variables
        d2 = (xr.open_dataset(twod_path, chunks={'time': 1})
              .sel(time=ds.time)
              .assign_coords(x=ds.x, y=ds.y))

        # add variables to three-d
        ds['RADTOA'] = d2.LWNT - d2.SWNT
        ds['RADSFC'] = d2.LWNS - d2.SWNS
        ds['layer_mass'] = w
        ds['rho'] = rho

        for key in 'Prec SHF LHF SOLIN SST'.split():
            ds[key] = d2[key]

        # append these variables
        ds.to_netcdf(output[0], engine='h5netcdf')

rule process_with_sam_once:
    input: DATA_PATH
    output: SAM_PROCESSED
    log: SAM_PROCESSED_LOG
    shell:
        """
        {sys.executable} -m src.data.process_ngaqua -n {input} {wildcards.step} {output} > {log} 2> {log}
        ncks -O --mk_rec_dmn time {output} {output}
        """

rule tropical_subset:
    input: TRAINING_DATA
    output: TROPICS_DATA
    shell: "ncks -d y,24,40 {input} {output}"
