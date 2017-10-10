import os

os.environ['PYTHONPATH'] = os.path.abspath(os.getcwd())


print(os.environ['PYTHONPATH'])
subworkflow ngaqua:
    snakefile: "snakemake/sam.rules"
    workdir: "data/ngaqua"
    configfile: "results/2017-09-28/ngaqua/config.yaml"


# rule all:
#     input: ngaqua("3d/Q1.nc")


rule weights:
    input: "data/ngaqua/stat.nc"
    output: "data/processed/ngaqua/w.nc"
    script: "lib/weights.py"
