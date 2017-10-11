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


rule linear_regression:
    input: "data/ml/ngaqua/data.pkl"
    output: "data/ml/ngaqua/linear_model.pkl"
    script: "lib/linear_regression.py"



rule pca:
    input: data3d=expand("data/ngaqua/3d/{f}.nc", f=['QT', 'SL']),
            data2d=expand("data/ngaqua/2d/{f}.nc", f=['LHF', 'SHF']),
            weight="data/processed/ngaqua/w.nc"
    output: "data/ml/ngaqua/pca.pkl"
    script: "lib/pca.py"

rule prepvars:
    input: data3d=expand("data/ngaqua/3d/{f}.nc", f=['QT', 'SL', 'Q1', 'QRAD', 'Q2']),
            data2d=expand("data/ngaqua/2d/{f}.nc", f=['LHF', 'SHF']),
            weight="data/processed/ngaqua/w.nc"
    output: "data/ml/ngaqua/data.pkl"
    params: input_vars="QT SL LHF SHF".split(' '),
            output_vars="Q1c Q2".split(' ')
    script: "lib/prepvars.py"
