import os
import sys
from os.path import join, abspath

## VARIABLES
DATA_PATH = config.get("data_path", "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX")
DATA_URL = "https://atmos.washington.edu/~nbren12/data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX.tar"
NUM_STEPS = config.get('NSTEPS', 640)
TRAINING_DATA = "data/processed/training.nc"
TROPICS_DATA = "data/processed/tropics.nc"
SAM_PATH = config.get("sam_path", "/opt/sam")
DOCKER = config.get("docker", True)

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

        # Compute forcings
        for key in ['QT', 'SLI', 'U', 'V']:
            forcing_key = 'F' + key
            src = ds[key].diff('step') / ds.step.diff('step') / 86400
            src = src.isel(step=0).drop('step')
            ds[forcing_key] = src

        # append these variables
        ds.to_netcdf(output[0], engine='h5netcdf')

rule process_with_sam_once:
    input: DATA_PATH
    output: SAM_PROCESSED
    log: SAM_PROCESSED_LOG
    params: sam=SAM_PATH,
            docker='--docker' if DOCKER else '--no-docker'
    shell:
        """
        {sys.executable} -m src.data.process_ngaqua \
            -n {input}  \
            --sam {params.sam} \
            {params.docker} \
            {wildcards.step} {output} > {log} 2> {log}
        """

rule tropical_subset:
    input: TRAINING_DATA
    output: TROPICS_DATA
    shell: "ncks -d y,24,40 {input} {output}"



rule sam_run_report:
    output: "reports/data/runs/{run}.html"
    params: run="data/runs/{run}", ipynb="reports/data/runs/{run}.ipynb",
            template=abspath("notebooks/templates/SAMNN-report.ipynb")
    shell: """
    papermill -p run_path $PWD/{params.run} -p training_data $PWD/{TRAINING_DATA} \
            --prepare-only {params.template} {params.ipynb}
    jupyter nbconvert --execute {params.ipynb}
    # clean up the notebook
    rm -f {params.ipynb}
    """

rule sam_run:
    output: directory("data/runs/model{model}-epoch{epoch}")
    shell: """
    {sys.executable} src/criticism/run_sam_ic_nn.py -nn models/{wildcards.model}/{wildcards.epoch}.pkl \
        -t 0 -p parameters2.json data/runs/model{wildcards.model}-epoch{wildcards.epoch}
    """
