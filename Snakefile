import os
import sys
from os.path import join, abspath, dirname
from datetime import datetime

import xarray as xr


def get_current_date_string():
    today = datetime.now()
    return today.isoformat().split('T')[0]


## VARIABLES
DATA_PATH = config.get("data_path", "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX")
DATA_URL = "https://atmos.washington.edu/~nbren12/data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX.tar"
NUM_STEPS = config.get('NSTEPS', 640)
TRAINING_DATA = "data/processed/training.nc"
TROPICS_DATA = "data/processed/tropics.nc"
SAM_PATH = config.get("sam_path", "/opt/sam")
DOCKER = config.get("docker", True)
TODAY = get_current_date_string()
RUN_SAM_SCRIPT = config.get("sam_script", "setup/docker/execute_run.sh")

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

rule process_with_sam_once_concat:
    input: "data/tmp/{step}/.done"
    output: SAM_PROCESSED
    run:
        from src.data.sam import SAMRun
        import shutil
        path = dirname(input[0])
        run = SAMRun(path, 'control')

        ds = run.data_3d

        ds = ds.rename({'time': 'step'})
        ds = ds.assign_coords(time=ds.step[0], step=ds.step - ds.step[0])
        ds = ds.expand_dims('time')
        ds.attrs['sam_namelist'] = json.dumps(run.namelist)
        ds.to_netcdf(output[0], unlimited_dims=['time'], engine='h5netcdf')

        # clean up SAM run
        shutil.rmtree(path)

rule process_with_sam_once:
    input: DATA_PATH
    output: touch("data/tmp/{step}/.done")
    log: SAM_PROCESSED_LOG
    shell: """
    rundir=$(dirname {output})
    rm -rf $rundir
    {sys.executable} src/criticism/run_sam_ic_nn.py  \
        -t 0 -p assets/parameters_process.json \
        -n data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX \
        -s {SAM_PATH} \
        $rundir
    # run sam
    {RUN_SAM_SCRIPT} $rundir >> {log} 2>> {log}
    exit 0
    """

rule tropical_subset:
    input: TRAINING_DATA
    output: TROPICS_DATA
    shell: "ncks -d y,24,40 {input} {output}"

rule zonal_time_mean:
    input: TRAINING_DATA
    output: "data/processed/training.mean.nc"
    run:
        ds = xr.open_dataset(input[0])
        mean = ds.isel(step=0).mean(['x', 'time'])
        mean.to_netcdf(output[0])


## SAM Execution ############################################################
TRAINING_MEAN=abspath("data/processed/training.mean.nc")
rule sam_run_report:
    input: "data/runs/{run}/.{id}.done"
    output: "reports/data/runs/{run}/{id}.html"
    params: run="data/runs/{run}", ipynb="reports/data/runs/{run}/{id}.ipynb",
            template=abspath("notebooks/templates/SAMNN-report.ipynb")
    shell: """
    papermill -p run_path $PWD/{params.run} -p training_data $PWD/{TRAINING_DATA} \
             -p caseid {wildcards.id} \
            -p training_data_mean {TRAINING_MEAN} \
            --prepare-only {params.template} {params.ipynb}
    jupyter nbconvert --execute {params.ipynb}
    # clean up the notebook
    rm -f {params.ipynb}
    """

rule sam_run:
    # need to use a temporary file here so that the model output isn't deleted
    output: touch("data/runs/model{model}-epoch{epoch}/.{id}.done")
    log: "data/runs/model{model}-epoch{epoch}/{id}.log"
    params: rundir="data/runs/model{model}-epoch{epoch}/",
            model="models/{model}/{epoch}.pkl"
    shell: """
    rm -rf {params.rundir}
    {sys.executable} src/criticism/run_sam_ic_nn.py -nn {params.model} \
        -t 0 -p assets/parameters2.json {params.rundir}
    # run sam
    {RUN_SAM_SCRIPT} {params.rundir} >> {log} 2>> {log}
    exit 0
    """

rule nudge_run:
    output: directory(f"data/runs/{TODAY}-nudging")
    shell: """
    {sys.executable} src/criticism/run_sam_ic_nn.py \
    -t 0 -p assets/parameters_nudging.json {output}
    """

rule micro_run:
    output: directory(f"data/runs/{TODAY}-microphysics")
    shell: """
    {sys.executable} src/criticism/run_sam_ic_nn.py \
    -t 0 -p assets/parameters_micro.json {output}
    """

## Model Training Rules ########################################

rule train_pca_pre_post:
    output: "models/prepost.pkl"
    # input: TRAINING_DATA
    shell: """
    python -m uwnet.train train_pre_post with data={TRAINING_DATA} prepost.path={output}
    """

rule train_model:
    input: "models/prepost.pkl"
    shell: """
    python -m uwnet.train with data={TRAINING_DATA} prepost.path={input} prepost.kind='saved' \
        batch_size=64 lr=.005 epochs=5 -m uwnet
    """
