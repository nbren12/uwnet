import sys
import xarray as xr
import os
from lib.thermo import liquid_water_temperature, total_water, q1, q2
from lib.util import xopena, wrap_xarray_calculation
from lib.scam import create_iopfile
from xnoah import swap_coord

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

rule all:
    input: "data/output/columns.nc", "data/output/scam.nc"


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
            #forcings="data/processed/denoised/forcings.nc"
    output: "data/output/time_series_data.pkl",
    script: "scripts/torch_preprocess.py"


rule fit_model:
    input: inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    output: "data/output/model.{k}.torch"
    params: num_epochs=4, num_steps=2000, nsteps=1, nhidden=(128, 128), lr=.01,
            window_size=10, cuda=False, batch_size=200,
            radiation='zero',
            precip_in_loss=False,
            precip_positive=False,
            interactive_vertical_adv=False
    script: "scripts/torch_time_series2.py"


rule forced_column_slp:
    input: model="data/output/model.1.torch",
           inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    output: "data/output/columns.nc"
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

rule run_scam:
    input: "data/processed/iop.nc"
    output: "data/processed/iop/{i}-{j}/cam.nc"
    run:
        from lib import scam
        from lib.cam import load_cam
        i = int(wildcards.i)
        j = int(wildcards.j)
        loc = xr.open_dataset(input[0], chunks={'lon': 1, 'lat':1})\
                .isel(lon=i, lat=j)

        output_dir = os.path.dirname(output[0])
        shell(f"rm -rf {output_dir}")
        shell(f"mkdir -p {output_dir}")

        scam.save_iop_dir(output_dir, loc)

        shell(f"ext/scam/run_docker.sh data/processed/iop/{i}-{j}/ > /dev/null") 
        load_cam(f"data/processed/iop/{i}-{j}/camrun.cam.h0.*.nc")\
            .to_netcdf(output[0])


rule combine_scam:
    input: expand("data/processed/iop/{i}-{j}/cam.nc", i=range(128), j=8)
    output: "data/output/scam.nc"
    script: "scripts/combine_scam.py"


rule plot_model:
    input: x="data/processed/inputs.nc",
           f="data/processed/forcings.nc",
           mod="data/output/model.1.torch"
    output: "data/output/plots.html"
    script: "scripts/model_report.py"

rule fit_models:
    input: expand("data/output/model.{k}.torch", k=range(4))

