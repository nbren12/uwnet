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
NUM_STEPS = config.get('NSTEPS', 10)
TRAINING_DATA = "data/processed/training/{sigma}.nc"
TROPICS_DATA = "data/processed/tropics.nc"
SAM_RESOLUTION = "128x64x34"
SAM_PATH = config.get("sam_path", f"/opt/sam/OBJ/{SAM_RESOLUTION}")
DOCKER = config.get("docker", True)
TODAY = get_current_date_string()
RUN_SAM_SCRIPT = config.get("sam_script", "setup/docker/execute_run.sh")

# wildcard targets
TRAINING_CONFIG = "assets/training_configurations/{model}.json"
TRAINED_MODEL = "models/{model}"
TRAINING_LOG = "models/{model}/log"
TRAINING_DONE = join(TRAINED_MODEL, ".done")
SAM_RUN = "data/runs/{model}/epoch{epoch}/"
SAM_RUN_STATUS = join(SAM_RUN, ".done")
SAM_LOG = join(SAM_RUN, "log")

## Temporary output locations
SAM_PROCESSED_LOG = "data/tmp/{step}.log"

## Set environmental variables
# add 'bin' folder to PATH
os.environ['PATH'] = os.path.abspath('bin') + ':' + os.environ['PATH']

print("Number of steps to process:", NUM_STEPS)

## RULES
rule all:
    input: expand(TRAINING_DATA, sigma=["sigma0.75", "noBlur"])

rule download_data:
    output: DATA_PATH
    shell: "cd data/raw && curl {DATA_URL} | tar xv"

rule preprocess_concat_sam_processed:
    input: expand("data/tmp/{{sigma}}/{step}.nc", step=range(NUM_STEPS))
    output: TRAINING_DATA
    shell:
        """
        echo {input} | ncrcat -o {output}
        """

rule preprocess_process_with_sam_once_blurred:
    input: DATA_PATH,
           sam_parameters="assets/sam_preprocess.json"
    output: "data/tmp/sigma{sigma}/{step}.nc"
    params: ngaqua_root=DATA_PATH, sigma="{sigma}"
    script: "uwnet/data/preprocess.py"


rule preprocess_process_with_sam_once:
    input: DATA_PATH,
            sam_parameters="assets/sam_preprocess.json"
    output: "data/tmp/noBlur/{step}.nc"
    params: ngaqua_root=DATA_PATH, sigma=False
    script: "uwnet/data/preprocess.py"

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
    jupyter nbconvert  --ExecutePreprocessor.timeout=600 \
                       --allow-errors \
                       --execute {params.ipynb}
    # clean up the notebook
    rm -f {params.ipynb}
    """

rule sam_run:
    # need to use a temporary file here so that the model output isn't deleted
    # input: TRAINED_MODEL
    input: TRAINING_DONE
    output: touch(SAM_RUN_STATUS)
    log: SAM_LOG
    params: rundir=SAM_RUN,
            model= join(TRAINED_MODEL, "{epoch}.pkl"),
            ngaqua = DATA_PATH,
            sam_src = config['sam_path'],
            step=0
    shell: """
    rm -rf {params.rundir}
    {sys.executable} -m  src.sam.create_case -nn {params.model} \
        -n {params.ngaqua} \
        -s {params.sam_src} \
        -t {params.step} -p assets/parameters_sam_neural_network.json \
       {params.rundir}
    # run sam
    {RUN_SAM_SCRIPT} {params.rundir} >> {log} 2>> {log}
    exit 0
    """

rule nudge_run:
    output: directory(f"data/runs/{TODAY}-nudging")
    shell: """
    {sys.executable} -m  src.sam.create_case \
    -t 0 -p assets/parameters_nudging.json {output}
    """

rule micro_run:
    output: touch(f"data/runs/{TODAY}-{{kind}}/.{{id}}.done")
    params: rundir=f"data/runs/{TODAY}-{{kind}}"
    log: f"data/runs/{TODAY}-{{kind}}/{{id}}.log"
    shell: """
    rm -rf {params.rundir}
    {sys.executable}   -m src.sam.create_case \
    -t 0 -p assets/parameters_{wildcards.kind}.json {params.rundir}

    {RUN_SAM_SCRIPT} {params.rundir} >> {log} 2>> {log}
    exit 0
    """

# Save the state within the SAM-python interface for one time step
rule save_sam_interface_state:
    output: "assets/state.pt"
    shell:"""
    dir=$(mktemp --directory)
    {sys.executable} -m  src.sam.create_case \
         -p assets/parameters_save_python_state.json \
         $dir
    execute_run.sh $dir
    mv -f $dir/state.pt assets/
    rm -rf $dir
    """

## Model Training Rules ########################################

rule train_pca_pre_post:
    output: "models/prepost.pkl"
    # input: TRAINING_DATA
    shell: """
    python -m uwnet.train train_pre_post with data={TRAINING_DATA} prepost.path={output}
    """

rule train_model:
    input: TRAINING_CONFIG
    output: touch(TRAINING_DONE)
    log: join(TRAINED_MODEL, "log")
    params:
        dir=TRAINED_MODEL
    shell: """ 
    python -m uwnet.train with {input}  output_dir={params.dir} > {log} 2> {log}
    """

rule debias_trained_model:
    input: "models/{model}.pkl"
    output: "models/{model}.debiased.pkl"
    params: data=TRAINING_DATA, prognostics=['QT', 'SLI']
    script: "src/models/debias.py"


rule compute_noise_for_trained_model:
    input: "models/{model}.pkl"
    output: "noise/{model}.pkl"
    run:
        from uwnet.whitenoise import fit
        import torch
        noise = fit(input[0])
        torch.save(noise, output[0])
