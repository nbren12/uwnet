import numpy as np
import sys
import xarray as xr
import os

conda: "environment.yml"

# configurations
configfile: "config.yaml"
nseeds = config.get('nseeds', 10)
nepoch = config.get('nepochs', 6)

# setup environment
os.environ['PYTHONPATH'] = os.path.abspath(os.getcwd())
print("PYTHONPATH", os.environ['PYTHONPATH'])
shell.executable("/bin/bash")

# input files
files_3d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/all.nc"
file_2d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc"
file_stat = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"

output_files = [
    "data/output/scam.nc",
    "data/processed/rce/10-8/cam.nc"
    "data/output/model.VaryT-20/3/5/columns.nc"
]

rule all:
    input: expand("data/output/model.{region}/0/3/state.torch", region=range(3))

rule download_data_file:
    output: "data/raw/{f}"
    shell: "rsync --progress -z nbren12@olympus:/home/disk/eos8/nbren12/Data/id/{wildcards.f} {output}"

def modeling_experiments():
    model_fit_params = {}
    model_fit_params['0'] = dict(y=np.r_[1:17,48:64], window_size=10)
    model_fit_params['1'] = dict(y=np.r_[15:28,36:50], window_size=10)
    model_fit_params['2'] = dict(y=np.r_[26:38], window_size=20)
    return model_fit_params



model_files = expand("data/output/model.{k}/{seed}.torch",
                     k=modeling_experiments(), seed=range(nseeds))

model_errors = expand("data/output/model.{k}/{seed}/{epoch}/error.nc",
                      k=modeling_experiments(), seed=range(nseeds),
                      epoch=range(nepoch+1))

model_args = expand("data/output/model.{k}/{seed}/arguments.json",
                      k=modeling_experiments(), seed=range(nseeds),
                      epoch=range(nepoch+1))

rule fit_all_models:
    input: model_files


def fit_model_params(wildcards):
    d =  modeling_experiments()[wildcards.k].copy()
    d['seed'] = int(wildcards.seed)
    d['cuda']  = config.get('cuda', False)
    d['output_dir'] = f"data/output/model.{wildcards.k}/{wildcards.seed}/"
    d['num_epochs'] = nepoch
    return d


rule fit_model:
    input: **config['paths']
    output:
        expand("data/output/model.{{k}}/{{seed}}/{epoch}/state.torch", epoch=range(nepoch+1)),
        "data/output/model.{k}/{seed}/loss.json",
        "data/output/model.{k}/{seed}/arguments.json",
    log: "data/output/model.{k}/{seed}/log.txt"
    params: fit_model_params
    script: "scripts/train_neural_network.py"


rule test_model:
    input: inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc",
           model="{d}/state.torch"
    output: "{d}/error.nc"
    script: "scripts/test_error.py"

rule combine_errors:
    input: model_errors
    output: "data/output/test_error.nc"
    script: "scripts/combine_errors.py"

rule forced_column_slp:
    input: **dict(state="{d}/state.torch", **config['paths'])
    priority: 10
    output: "{d}/columns.nc"
    script: "scripts/forced_column_slp.py"

rule forced_column_slp_substep:
    input: state="{d}/state.torch",
            inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    priority: 10
    params: nsteps=9
    output: "{d}/20minsubstep.nc"
    script: "scripts/forced_column_slp.py"

rule rce_column_slp:
    input: state="{d}/state.torch",
           inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    priority: 10
    output: "{d}/rce.nc"
    params: RCE=True
    script: "scripts/forced_column_slp.py"

rule plot_model:
    input: cols="{d}/columns.nc",
    output: "{d}/plots.html"
    script: "scripts/model_report.py"

include: "rules/cam.smk"

rule compute_tendencies_sam:
    output: "data/interim/tendencies/{physics}/{t}.nc"
    log: "data/interim/tendencies/{physics}/{t}.log"
    shell: "python -m lib.sam -t {wildcards.t} -p {wildcards.physics} ~/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX {output} > {log}"

rule combine_sam_tends:
    input: expand("data/interim/tendencies/dry/{t}.nc", t=range(640))
    output: "data/processed/dry_tends.nc"
    script: "scripts/combine_sam_tends.py"

