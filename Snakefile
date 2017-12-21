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
    input: "data/raw/ngaqua/stat.nc", "data/ml/ngaqua/time_series_fit.torch", "data/ml/ngaqua/multistep_objective.torch"


def ngaqua_files():
    return ['coarse/',
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
            'stat.nc',
            'README']

# download data
rule ngaqua_file:
    output: "data/raw/ngaqua/{f}"
    shell: "scp nbren12@olympus:/home/disk/eos8/nbren12/Data/id/726a6fd3430d51d5a2af277fb1ace0c464b1dc48/{wildcards.f} {output}"

rule data:
    input: expand("data/raw/ngaqua/{f}", f=ngaqua_files())

rule weights:
    input: "data/raw/ngaqua/stat.nc"
    output: "data/processed/ngaqua/w.nc"
    script: "lib/scripts/weights.py"


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
    script: "lib/scripts/advection.py"

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
    script: "lib/scripts/apparent_source.py"

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

rule time_series_data:
    input: forcing= ["data/calc/forcing/ngaqua/sl.nc", "data/calc/forcing/ngaqua/qt.nc"],
            inputs=["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"],
            weight= "data/processed/ngaqua/w.nc"
    output: "data/ml/ngaqua/time_series_data.pkl"
    script: "lib/scripts/torch_preprocess.py"



rule multiple_step_obj:
    input: "data/ml/ngaqua/time_series_data.pkl"
    output: "data/ml/ngaqua/multistep_objective.torch"
    shell:
        """
        {sys.executable} lib/scripts/torch_time_series.py multi \
                   --num_epochs 4 --window_size 10 --num_steps 500 --batch_size 100 --learning-rate .010\
        --weight_decay 0.00 \
        {input} {output}
        """
