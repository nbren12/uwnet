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

rule adams_bashforth_3:
    input: "data/calc/forcing/{f}.nc"
    output: "data/calc/ab3/{f}.nc"
    script: "lib/scripts/adams_bashforth.py"

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

rule tropics:
    input: "data/{f}.nc"
    output: "data/tropics/{f}.nc"
    shell: "ncks -d y,24,39 {input} {output}"

rule linear_regression:
    input: "data/ml/ngaqua/data.pkl"
    output: "data/ml/ngaqua/linear_model.pkl"
    params: model="linear"
    script: "lib/scripts/fit_model.py"

rule mca_regression:
    input: "data/ml/ngaqua/data.pkl"
    output: "data/ml/ngaqua/mca_regression.pkl"
    params: model="mcr"
    script: "lib/scripts/fit_model.py"

rule mca:
    input: "data/ml/ngaqua/data.pkl"
    output: "data/ml/ngaqua/mca.pkl"
    params: n_components=4
    script: "lib/scripts/mca_script.py"

input_data = [
    "data/calc/ngaqua/qt.nc",
    "data/calc/ngaqua/sl.nc",
    "data/calc/ngaqua/q1.nc",
    "data/calc/ngaqua/q2.nc",
    "data/raw/ngaqua/coarse/3d/QRAD.nc",
]

# prepare data for linear model fitting these rule reads in Q1 and Q2, computes
# Q1c, and then outputs the required data in a dict_like object
rule prepvars:
    input: data3d=input_data,
           # removed 2d variables from the analysis, since they don't seem necessary
           # data2d=expand("data/ngaqua/2d/{f}.nc", f=['LHF', 'SHF']),
            weight="data/processed/ngaqua/w.nc"
    output: "data/ml/ngaqua/data.pkl"
    params: input_vars="qt sl".split(' '),
            output_vars="Q1c Q2".split(' ')
    script: "lib/scripts/prepvars.py"


# prepared data for DMD analysis this rule reads in the data, applies the
# forcing, and then saves the output
rule dmd_data:
    input: forcing= ["data/calc/forcing/ngaqua/sl.nc", "data/calc/forcing/ngaqua/qt.nc"],
           inputs=["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"],
           weight= "data/processed/ngaqua/w.nc"
    output: "data/ml/dmd.pkl"
    script: "lib/scripts/dmd.py"

# prepared data for DMD analysis this rule reads in the data, applies the
# forcing, and then saves the output
rule time_series_data:
    input: forcing= ["data/calc/forcing/ngaqua/sl.nc", "data/calc/forcing/ngaqua/qt.nc"],
            inputs=["data/calc/ngaqua/sl.nc", "data/calc/ngaqua/qt.nc"],
            weight= "data/processed/ngaqua/w.nc"
    output: "data/ml/ngaqua/time_series_data.pkl"
    script: "lib/scripts/data_to_numpy.py"

rule time_series_slp:
    input: "data/ml/ngaqua/time_series_data.npz"
    output: "data/ml/ngaqua/time_series_fit.torch"
    params: n=4, weight_decay=0.5
    script: "lib/scripts/torch_cli.py"


rule multiple_step_obj:
    input: "data/ml/ngaqua/time_series_data.pkl"
    output: "data/ml/ngaqua/multistep_objective.torch"
    shell:
        """
        {sys.executable} lib/scripts/torch_time_series.py multi \
                   --num_epochs 1 --window_size 10 --num_steps 1000 --batch_size 500 --learning-rate .010\
        --weight_decay 0.00 \
        {input} {output}
        """
