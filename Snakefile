import os
import sys
from os.path import join, abspath, dirname
import json
from datetime import datetime
from src import data


## Set environmental variables
# add 'bin' folder to PATH
os.environ['PATH'] = os.path.abspath('bin') + ':' + os.environ['PATH']


def get_current_date_string():
    today = datetime.now()
    return today.isoformat().split('T')[0]

singularity: "docker://nbren12/uwnet:latest"

## VARIABLES
DATA_PATH = config.get("data_path", "data/raw/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX")
DATA_URL = "https://atmos.washington.edu/~nbren12/data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX.tar"
NUM_STEPS = config.get('NSTEPS', 640)
TRAINING_DATA = "data/processed/training/{data_name}.nc"
RESHAPED_DATA = "data/processed/reshaped/{data_name}/{train_or_test}.zarr"
TROPICS_DATA = "data/processed/tropics.nc"
SAM_RESOLUTION = "128x64x34"
sam_src = config.get("sam_path", "/opt/sam")
print("SAM_SRC", sam_src)
DOCKER = config.get("docker", True)
TODAY = get_current_date_string()
num_epoch = config.get("num_epoch", 1)
epochs = list(range(1, num_epoch+1))

# wildcard targets
TRAINING_CONFIG = "assets/training_configurations/{model}.json"

TRAINING_LOG = f"{{type}}/{{model}}/{num_epoch}.log" # change this
TRAINED_MODEL = f"{{type}}/{{model}}/{num_epoch}.log" # change this
NN_METRIC = "nn/{model}/{epoch}.{train_or_test}.json"

SAM_RUN = "data/runs/{sam_params}/{type}/{model}/epoch{epoch}/"
SAM_RUN_SK = f"data/runs/sklearn/{{model}}"
SAM_RUN_STATUS = join(SAM_RUN, ".done")
SAM_LOG = join(SAM_RUN, "log")

VISUALIZE_SAM_DIR = "reports/runs/{sam_params}/{type}/{model}/epoch{epoch}"
MODEL_FILE = "nn/{model}/{epoch}.pkl"
DEBIASED_MODEL = "debiased/{model}/{epoch}.pkl"

types = ["nn"]
models = ["NNLower", "NNAll"]
sam_params = ["samnn"]

#types = ["sklearn"]
#models = ["rf_regressor"]
#sam_params = ["parameters_sam_rf_regressor"]

# SAM_RUNS = expand(SAM_RUN_STATUS, model=models, epoch=["5"], type=types, sam_params=sam_params)
SAM_RUNS = data.run_paths.values()
SAM_REPORTS = []

def add_report(type, model, epoch, sam_params='samnn'):
    SAM_REPORTS.append(VISUALIZE_SAM_DIR.format(sam_params=sam_params, type=type, model=model, epoch=str(epoch)))

add_report('nn', 'NNLowerDecayLR', 20)

# Plots
scripts = ['bias',
 'qp_acf',
 'damping_coefs',
 'hovmoller_mean_pw_prec',
 'pattern_correlation',
 'pdf',
 'precip_maps',
 'predicted_vs_actual_q1',
 'r2_q1_q2',
 'rms_weather_plots',
 'snapshots_pw',
 'spinup_error',
 'bootstrap']

jacobian_figures_relative = ["saliency-unstable.png", "saliency-stable.png"]
jacobian_figures_absolute = [join("notebooks/papers/", fig)
                             for fig in jacobian_figures_relative]

other_figures_absolute = [join("notebooks/papers/", script + ".pdf")
                         for script in scripts]
all_figs = jacobian_figures_absolute + other_figures_absolute

## Temporary output locations
SAM_PROCESSED_LOG = "data/tmp/{step}.log"

rule nn_metrics:
    input: expand(NN_METRIC, model=models, epoch=epochs, train_or_test=["train", "test"])

rule reports:
    input: SAM_REPORTS

rule data:
    input: expand(RESHAPED_DATA, data_name=['noBlur'], train_or_test=["train", "test"])

rule download_data:
    output: directory(DATA_PATH)
    shell: "cd data/raw && curl {DATA_URL} | tar xv"

rule preprocess_concat_sam_processed:
    input: expand("data/tmp/{{data_name}}/{step}.nc", step=range(NUM_STEPS))
    output: TRAINING_DATA
    shell:
        """
        echo {input} | ncrcat -o {output}
        """

rule preprocess_process_with_sam_once_blurred:
    input: DATA_PATH,
           sam_parameters="assets/sam_preprocess.json"
    output: temp("data/tmp/sigma{sigma}/{step}.nc")
    params: ngaqua_root=DATA_PATH, sigma="{sigma}"
    script: "uwnet/data/preprocess.py"


rule preprocess_process_with_sam_once:
    input: DATA_PATH,
            sam_parameters="assets/sam_preprocess.json"
    output: temp("data/tmp/noBlur/{step}.nc")
    params: ngaqua_root=DATA_PATH, sigma=False
    script: "uwnet/data/preprocess.py"


rule preprocess_process_with_sam_no_hyperdiff:
    input: DATA_PATH,
            sam_parameters="assets/sam_preprocess_no_hyperdiff.json"
    output: temp("data/tmp/advectionOnly/{step}.nc")
    params: ngaqua_root=DATA_PATH, sigma=False
    script: "uwnet/data/preprocess.py"

rule reshape_training_data:
    input: TRAINING_DATA
    output: directory(RESHAPED_DATA)
    params:
        variables = ('QT', 'SLI', 'SOLIN', 'SST', 'QRAD', 'FQT', 'FSLI'),
        shuffle = True
    script: "uwnet/data/reshape.py"

rule tropical_subset:
    input: TRAINING_DATA
    output: TROPICS_DATA
    shell: "ncks -d y,24,40 {input} {output}"

rule zonal_time_mean:
    input: data.training_data
    output: data.ngaqua_climate_path
    run:
        import xarray as xr
        ds = xr.open_dataset(input[0])
        mean = ds.mean(['x', 'time'])
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

rule sam_run_nn:
    # need to use a temporary file here so that the model output isn't deleted
    input: "{type}/{model}/{epoch}.pkl"
    output: touch(SAM_RUN_STATUS)
    log: SAM_LOG
    params: rundir=SAM_RUN,
            ngaqua=DATA_PATH,
            sam_src=sam_src,
            sam_params="assets/{sam_params}.json",
            step=0
    shell: """
    rm -rf {params.rundir}
    {sys.executable} -m  src.sam.create_case -nn {input} \
    -n {params.ngaqua} \
    -s {params.sam_src} \
    -t {params.step} \
    -p {params.sam_params} \
    {params.rundir}
    # run sam
    cd {params.rundir}
    sh run.sh >> log 2>> log
    """

rule sam_run_sk:
    # need to use a temporary file here so that the model output isn't deleted
    input: "sklearn_models/{model}.pkl"
    output: touch(join(SAM_RUN_SK,".done"))
    log: join(SAM_RUN_SK, "log")
    params: rundir=SAM_RUN_SK,
            ngaqua=DATA_PATH,
            sam_params="assets/parameters_sam_{model}.json",
            sam_src=sam_src,
            step=0
    shell:  """
    rm -rf {params.rundir}
    {sys.executable} -m  src.sam.create_case -sk {input} \
    -n {params.ngaqua} \
    -s {params.sam_src} \
    -p {params.sam_params} \
    -t {params.step} \
    {params.rundir}
    # run sam
    cd {params.rundir}
    sh run.sh >> log 2>> log
     """


rule nudging_run:
    # need to use a temporary file here so that the model output isn't deleted
    output: touch(f"data/runs/{TODAY}-nudging/.done")
    params: rundir=f"data/runs/{TODAY}-nudging",
            ngaqua=DATA_PATH,
            sam_src=sam_src,
            sam_params="assets/parameters_nudge_nn.json",
            step=0
    shell: """
    rm -rf {params.rundir}
    {sys.executable} -m  src.sam.create_case \
    -n {params.ngaqua} \
    -s {params.sam_src} \
    -t {params.step} \
    -p {params.sam_params} \
    {params.rundir}
    # run sam
    cd {params.rundir}
    sh run.sh
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


def get_training_data(wildcards):
    path = TRAINING_CONFIG.format(**wildcards)
    with open(path) as f:
        model_config = json.load(f)
        return [model_config['train_data'], model_config['test_data']]

rule train_models:
    input: expand("nn/{model}/{epoch}.pkl", epoch=epochs, model=models)

rule train_nn:
    input:
        config=TRAINING_CONFIG,
        data=get_training_data
    resources: mem_mb=26000
    output: expand("nn/{{model}}/{epoch}.pkl", epoch=epochs)
    log: f"nn/{{model}}/log"
    params:
        dir="nn/{model}/"
    shell: """
    python -m uwnet.ml_models.nn.train with {input.config}  epochs={num_epoch} output_dir={params.dir} > {log} 2> {log}
    """

rule train_rf:
    input:
        config=TRAINING_CONFIG

    resources: mem_mb=26000
    output:  "sklearn_models/{model}.pkl"
    log: "sklearn_models/{model}.log"
    shell:  """
    python -m uwnet.ml_models.train_generic_sklearn -op {params.dir} -cf {input.config} \
        > {log} 2> {log}
    """

rule nn_metric:
    input: "data/processed/reshaped/noBlur/{train_or_test}.zarr", "nn/{model}/{epoch}.pkl"
    output: NN_METRIC
    resources: mem_mb=8000
    shell: "python uwnet/criticism/evaluate.py {input} > {output}"

rule debias_nn:
    input: TRAINED_MODEL
    output: DEBIASED_MODEL
    params: data=get_training_data, prognostics=['QT', 'SLI'], model=MODEL_FILE
    script: "src/models/debias.py"


rule compute_noise_for_trained_model:
    input: "models/{model}.pkl"
    output: "noise/{model}.pkl"
    run:
        from uwnet.whitenoise import fit
        import torch
        noise = fit(input[0])
        torch.save(noise, output[0])

## Visualizations ########################################

rule visualize_sam_run:
    input: SAM_RUN_STATUS
    output: directory(VISUALIZE_SAM_DIR)
    params:
        run = SAM_RUN
    shell: "python -m src.visualizations.sam_run {params.run} {output}"

## Plots for paper ########################################

rule upload_figs:
    input: all_figs
    shell: "cp {input} ~/public_html/reports/uwnet/plots2019/"

rule paper_plots:
    input: all_figs


rule vis_jacobian:
    input: SAM_RUNS, data.ngaqua_climate_path
    output: jacobian_figures_absolute
    shell: "python notebooks/papers/jacobian.py {output}"

rule run_vis_script:
    input: script="notebooks/papers/{script}.py", runs=SAM_RUNS
    output: "notebooks/papers/{script}.pdf"
    shell: "cd notebooks/papers/ && python {wildcards.script}.py {wildcards.script}.pdf"
