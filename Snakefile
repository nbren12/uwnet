import sys
import xarray as xr
import os
from lib.thermo import liquid_water_temperature, total_water, q1, q2
from lib.util import xopena, wrap_xarray_calculation

os.environ['PYTHONPATH'] = os.path.abspath(os.getcwd())


print(os.environ['PYTHONPATH'])
# subworkflow ngaqua:
#     snakefile: "snakemake/sam.rules"
#     workdir: "data/ngaqua"
#     configfile: "results/2017-09-28/ngaqua/config.yaml"


# rule all:
#     input: ngaqua("3d/Q1.nc")

rule all:
    input: "data/raw/ngaqua/stat.nc", "data/ml/ngaqua/multistep_objective.torch"


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
        'coarse/3d/TABS.nc',
        'coarse/3d/QRAD.nc',
        'coarse/3d/QP.nc',
        'coarse/3d/QV.nc',
        'coarse/3d/V.nc',
        'coarse/3d/W.nc',
        'coarse/3d/QN.nc',
        'coarse/3d/U.nc',
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


rule prognostic_variables:
    input: T="data/raw/ngaqua/coarse/3d/TABS.nc",
           qn="data/raw/ngaqua/coarse/3d/QN.nc",
           qp="data/raw/ngaqua/coarse/3d/QP.nc",
           qv="data/raw/ngaqua/coarse/3d/QV.nc"
    output: sl="data/calc/ngaqua/sl.nc",
            qt="data/calc/ngaqua/qt.nc"
    run:
        sl = liquid_water_temperature(input.T, input.qn, input.qp)\
		.to_dataset(name="sl").to_netcdf(output.sl)
        qt = total_water(input.qv, input.qn)\
               .to_dataset(name="qt")\
               .to_netcdf(output.qt)

rule advection_wildcard:
    input: u="data/raw/ngaqua/coarse/3d/U.nc",
            v="data/raw/ngaqua/coarse/3d/V.nc",
            w="data/raw/ngaqua/coarse/3d/W.destaggered.nc",
            f="data/calc/ngaqua/{f}.nc"
    output: "data/calc/adv/ngaqua/{f}.nc"
    script: "scripts/advection.py"

# sl forcing
rule advection_forcing:
    input: adv="data/calc/adv/{f}.nc",
    output: "data/calc/forcing/{f}.nc"
    run:
        f = -xopena(input.adv)*86400
        f.to_dataset(name=f.name)\
         .to_netcdf(output[0])

rule apparent_source:
    input: forcing="data/calc/forcing/{f}.nc",
           data="data/calc/{f}.nc"
    output: "data/calc/apparent/{f}.nc"
    script: "scripts/apparent_source.py"

rule q1:
    input: "data/calc/apparent/ngaqua/sl.nc",
           "data/raw/ngaqua/coarse/3d/QRAD.nc"
    output: "data/calc/ngaqua/q1.nc"
    run:
        d = xr.open_mfdataset(input)
        q1(d).to_netcdf(output[0])

rule q2:
    input: "data/calc/apparent/ngaqua/qt.nc"
    output: "data/calc/ngaqua/q2.nc"
    run:
        d = xr.open_mfdataset(input)
        q2(d).to_netcdf(output[0])

# rule time_series_data:
#     input: forcing= ["data/calc/forcing/ngaqua/sl.nc", "data/calc/forcing/ngaqua/qt.nc"],
#             inputs=["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"],
#             weight= "data/processed/ngaqua/w.nc"
#     output: "data/ml/ngaqua/time_series_data.pkl"
#     script: "scripts/torch_preprocess.py"


rule time_series_data:
    input: expand("data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/{f}.nc",\
                  f="U V W QV QN TABS QP".split(" "))
    output: "data/ml/ngaqua/time_series_data.pkl"
    script: "scripts/torch_preprocess.py"

rule fit_model:
    input: "data/ml/ngaqua/time_series_data.pkl"
    output: "data/ml/ngaqua/model.{k}.torch"
    params: num_epochs=4, num_steps=1000, nsteps=1, nhidden=(256, ), lr=.01,
            window_size=100, cuda=True, batch_size=1000
    script: "scripts/torch_time_series2.py"

rule fit_models:
    input: expand("data/ml/ngaqua/model.{k}.torch", k=range(4))

