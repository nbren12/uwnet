import xarray as xr
from lib.scam import create_iopfile

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
