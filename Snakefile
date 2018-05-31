import sys
import xarray as xr
import os
from lib.scam import create_iopfile

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
    input: expand("data/output/model.{region}/0/3/state.torch", region=range(5))

rule download_data_file:
    output: "data/raw/{f}"
    shell: "rsync --progress -z nbren12@olympus:/home/disk/eos8/nbren12/Data/id/{wildcards.f} {output}"

# rule inputs_and_forcings:
#     input: d3=files_3d, d2=file_2d, stat=file_stat
#     output: inputs="data/processed/inputs.nc",
#             forcings="data/processed/forcings.nc"
#     script: "scripts/inputs_and_forcings.py"


rule inputs_and_forcings:
    input: d3="data/processed/sam_tend.nc", d2=file_2d, stat=file_stat
    output: inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    script: "scripts/inputs_and_forcings.py"

rule time_series_data:
    input: inputs="data/processed/inputs.nc",
            forcings="data/processed/forcings.nc"
    output: "data/output/time_series_data.pkl",
    script: "scripts/torch_preprocess.py"

def modeling_experiments():
    model_fit_params = {}
    # global
    model_fit_params['0'] = dict(south=1, north=18)
    model_fit_params['1'] = dict(south=16, north=28)
    model_fit_params['2'] = dict(south=26, north=38)
    model_fit_params['3'] = dict(south=36, north=48)
    model_fit_params['4'] = dict(south=46, north=63)
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
    input: state="{d}/state.torch",
           inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
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
    input: cols="{d}/columns.nc",
    output: "{d}/plots.html"
    script: "scripts/model_report.py"


rule compute_tendencies_sam:
    output: "data/interim/tendencies/{physics}/{t}.nc"
    log: "data/interim/tendencies/{physics}/{t}.log"
    shell: "scripts/sam_init_cond.py -t {wildcards.t} -p {wildcards.physics} ~/Data/2018-05-30-NG_5120x2560x34_4km_10s_QOBS_EQX {output} > {log}"

rule combine_sam_tends:
    input: expand("data/interim/tendencies/{{physics}}/{t}.nc", t=range(640))
    output: "data/processed/tend_{physics}.nc"
    run:
        import sh
        from tqdm import tqdm

        pat = re.compile("(\d+).nc$")
        def key(s):
            return int(pat.search(s).group(1))

        input = sorted(input, key=key)

        deleteme = []
        print("Repacking")
        for f in tqdm(input):
            rec = f + 'rec.nc'
            sh.ncpdq("-O", '-a', 'time,step', f, rec)
            deleteme.append(rec)


        print("Concatenating")
        sh.ncrcat("-O", deleteme, output[0])

        print("cleaning up")
        for f in deleteme:
            os.unlink(f)
