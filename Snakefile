import sys
import xarray as xr
import os
from lib.thermo import liquid_water_temperature, total_water, q1, q2
from lib.util import xopena, wrap_xarray_calculation
from lib.scam import create_iopfile
from xnoah import swap_coord

configfile: "config.yaml"
nseeds = config.get('nseeds', 10)
nepoch = config.get('nepochs', 6)

# setup environment
os.environ['PYTHONPATH'] = os.path.abspath(os.getcwd())
shell.executable("/bin/bash")


print(os.environ['PYTHONPATH'])
# subworkflow ngaqua:
#     snakefile: "snakemake/sam.rules"
#     workdir: "data/ngaqua"
#     configfile: "results/2017-09-28/ngaqua/config.yaml"


# rule all:
#     input: ngaqua("3d/Q1.nc")

output_files = [
    "data/output/model.VaryT-20/3.rce.nc",
    "data/output/model.VaryT-20/3.columns.nc",
    "data/output/scam.nc",
    "data/processed/rce/10-8/cam.nc"
]

rule all:
    input: output_files


ngaqua_files =[
    'coarse/',
    'coarse/3d',
    'coarse/3d/TABS.nc',
    'coarse/3d/QRAD.nc',
    'coarse/3d/QP.nc',
    'coarse/3d/QV.nc',
    'coarse/3d/V.nc',
    'coarse/3d/W.nc',
    'coarse/3d/QN.nc',
    'coarse/3d/U.nc',
    'coarse/3d/W.destaggered.nc',
    'coarse/2d/all.nc',
    'stat.nc',
    'README'
]

run_ids = [
    '726a6fd3430d51d5a2af277fb1ace0c464b1dc48', '2/NG_5120x2560x34_4km_10s_QOBS_EQX'
]

manifest = {
    '2/NG_5120x2560x34_4km_10s_QOBS_EQX': [
        'coarse/3d/all.nc',
        'coarse/2d/all.nc',
        'stat.nc',
    ]
}


rule download_data_file:
    output: "data/raw/{f}"
    shell: "rsync --progress -z nbren12@olympus:/home/disk/eos8/nbren12/Data/id/{wildcards.f} {output}"

def _run_output(id):
    for f in manifest.get(id, ngaqua_files):
        yield os.path.join("data/raw", id, f)

rule all_data:
    input: _run_output(run_ids[1])

rule weights:
    input: "data/raw/ngaqua/stat.nc"
    output: "data/processed/ngaqua/w.nc"
    script: "scripts/weights.py"

files_3d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/all.nc"
file_2d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc"
file_stat = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"

rule inputs_and_forcings:
    input: d3=files_3d, d2=file_2d, stat=file_stat
    output: inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    script: "scripts/inputs_and_forcings.py"

rule denoise:
    input: "data/proccesed/forcings.nc"
    output: "data/processed/denoised/forcings.nc"
    run:
        from lib.denoise import denoise
        xr.open_dataset(input[0])\
          .apply(denoise)\
         .to_netcdf(output[0])

rule time_series_data:
    input: inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    output: "data/output/time_series_data.pkl",
    script: "scripts/torch_preprocess.py"



def modeling_experiments():
    """A comprehensive list of the all the modelling experiments to present for the paper.
    """
    model_fit_params = {}
    for n in [5, 64, 128, 256]:
        key = f'VaryNHid-{n}'
        model_fit_params[key] = dict(nhidden=(n,))

    for T in [2, 5, 10, 20, 40]:
        key = f'VaryT-{T}'
        model_fit_params[key] = dict(window_size=T)

    # model_fit_params['1'] = dict(nhidden=(256,))
    # model_fit_params['best'] = dict(nhidden=(256,), num_epochs=2)
    # model_fit_params['lrs'] = dict(nhidden=(256,), num_epochs=4, lr=.001)

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
    input: inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    output:
        expand("data/output/model.{{k}}/{{seed}}/{epoch}/{f}",\
               epoch=range(nepoch+1), f=["model.torch"]),
        "data/output/model.{k}/{seed}/loss.json",
        "data/output/model.{k}/{seed}/arguments.json",
    log: "data/output/model.{k}/{seed}/log.txt"
    params: fit_model_params
    script: "scripts/train_neural_network.py"


rule test_model:
    input: inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc",
           model="{d}/model.torch"
    output: "{d}/error.nc"
    script: "scripts/test_error.py"

rule combine_errors:
    input: model_errors
    output: "data/output/test_error.nc"
    script: "scripts/combine_errors.py"

rule forced_column_slp:
    input: model="data/output/{model}/{id}.torch",
           inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    priority: 10
    output: "data/output/{model}/{id}.columns.nc"
    script: "scripts/forced_column_slp.py"

rule rce_column_slp:
    input: model="data/output/{model}/{id}.torch",
            inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    priority: 10
    output: "data/output/{model}/{id}.rce.nc"
    params: RCE=True
    script: "scripts/forced_column_slp.py"

wildcard_constraints:
    i="\d+",
    j="\d+"


rule make_iop_file:
    input: d3=files_3d, d2=file_2d, stat=file_stat
    output: "data/processed/iop.nc"
    run:
        create_iopfile.main(input.d2, input.d3, input.stat, output[0])

rule prepare_iop_directories:
    input: "data/processed/iop.nc"
    output: nml="data/processed/iop/{i}-{j}/namelist.txt",
            nc="data/processed/iop/{i}-{j}/iop.nc"
    run:
        iop = xr.open_dataset(input[0])
        output_dir = "data/processed/iop"
        create_iopfile.save_all_dirs(iop, output_dir)

rule run_scam_forced:
    input: "data/processed/iop.nc"
    output: "data/processed/iop/{i}-{j}/cam.nc"
    script: "scripts/run_scam.py"

rule run_scam_rce:
    input: "data/processed/iop.nc"
    output: "data/processed/rce/{i}-{j}/cam.nc"
    params: RCE=True
    script: "scripts/run_scam.py"

rule combine_scam:
    input: expand("data/processed/iop/{i}-{j}/cam.nc", i=range(128), j=8)
    output: "data/output/scam.nc"
    script: "scripts/combine_scam.py"


rule plot_model:
    input: x="data/processed/inputs.nc",
           f="data/processed/forcings.nc",
           mod="data/output/model.1/1.torch"
    output: "data/output/plots.html"
    script: "scripts/model_report.py"
