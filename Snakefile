import sys
import xarray as xr
import os
from lib.thermo import liquid_water_temperature, total_water, q1, q2
from lib.util import xopena, wrap_xarray_calculation
from lib import create_iopfile
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
    input: "data/processed/iop/cam.nc"


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


files_3d =expand("data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/{f}.nc",
                 f="U V W QV QN TABS QP QRAD".split(" ")),
file_2d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc"
file_stat = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"

rule inputs_and_forcings:
    input: d3=files_3d,
           d2=file_2d,
           stat=file_stat,
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
    output: "data/ml/ngaqua/time_series_data.pkl",
    script: "scripts/torch_preprocess.py"


rule fit_model:
    input: "data/ml/ngaqua/time_series_data.pkl"
    output: "data/ml/ngaqua/model.{k}.torch"
    params: num_epochs=4, num_steps=2000, nsteps=1, nhidden=(256,), lr=.01,
            window_size=10, cuda=False, batch_size=200,
            radiation='zero',
            precip_in_loss=False,
            precip_positive=False,
            interactive_vertical_adv=False
    script: "scripts/torch_time_series2.py"


rule forced_column_slp:
    input: model="data/ml/ngaqua/model.1.torch",
           inputs="data/processed/inputs.nc",
           forcings="data/processed/forcings.nc"
    output: "data/ml/ngaqua/columns.nc"
    script: "scripts/forced_column_slp.py"


wildcard_constraints:
    i="\d+",
    j="\d+"


rule prepare_iop_directories:
    input: d3=files_3d, d2=file_2d, stat=file_stat
    output: nml="data/processed/iop/{i}-{j}/namelist.txt",
            nc="data/processed/iop/{i}-{j}/iop.nc"
    run:
        create_iopfile.main(input.d2, input.d3, input.stat,
                            output_dir="data/processed/iop")

rule run_scam:
    input: nml="data/processed/iop/{i}-{j}/namelist.txt",
           nc="data/processed/iop/{i}-{j}/iop.nc"
    output: "data/processed/iop/{i}-{j}/cam.nc"
    run:
        shell("""
        dir=$(dirname {input.nml})
        ext/scam/run_docker.sh $dir > /dev/null
        """)

        import lib.cam
        dirname = os.path.dirname(input.nml)
        pattern = os.path.join(dirname, "camrun.cam.h0.*.nc")
        lib.cam.load_cam(pattern)\
               .to_netcdf(output[0])

rule combine_scam:
    input: expand("data/processed/iop/{i}-{j}/{file}", i=range(128), j=7,\
                  file=['cam.nc', 'iop.nc'])
    output: "data/processed/iop/cam.nc"
    script: "scripts/combine_scam.py"


rule plot_model:
    input: x="data/processed/inputs.nc",
           f="data/processed/forcings.nc",
           mod="data/ml/ngaqua/model.1.torch"
    output: "data/ml/ngaqua/plots.html"
    script: "scripts/model_report.py"

rule fit_models:
    input: expand("data/ml/ngaqua/model.{k}.torch", k=range(4))

