# Machine learning approaches to convective parametrization

This project contains the code accompanying an article to be submitted.


## Setup

This project uses anaconda to manage the various libraries and packages used. If you do not already have it installed you will need to install [miniconda](https://conda.io/miniconda.html). Once that is installed the environment for this package can be installed by running
    conda env create environment.yml

Then, you have to activate this environment by running

    source activate nn_atmos_param
or 

    conda activate nn_atmos_param
depending on your platform.
Now, we need to make sure that the `lib` directory of this project is in the pythonpath. There are many ways to do this, but you can use

    python setup.py develop

## Downloading the data

The raw NG-Aqua data are available as a tarball at https://doi.org/10.5281/zenodo.1226370.
Once you download, that extract by running

    tar xzf NG_5120x2560x34_4km_10s_QOBS_EQX.tar.gz

in this project root directory.

## Fitting a Neural Network

This project uses [snakemake](https://snakemake.readthedocs.io/en/stable/) to manage its computational workflow. This tool operates very similarly to GNU make. The parameters used for training are defined in the  `modeling_experiments` function in the file `Snakefile`. To fit one of the predefined neural networks simply by running
    
    snakemake data/output/model.{model_name}/{seed}/{epoch}/state.torch
    
For example,

    snakemake ./data/output/model.VaryT-20/0/5/state.torch

will transform the data into a format suitable for the training using the script `scripts/inputs_and_forcings.py` and then train the neural network using `scripts/train_neural_network.py`.
Behind the scenes, this last script calls the function `train_multistep_objective` in `./lib/torch/training.py`.

## Running SCAM

SCAM is run using Docker. 
