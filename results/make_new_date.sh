#!/bin/bash
# make a new date directory with a typical environment.yml file

dir=$(date +%F)
mkdir $dir

pushd $dir
cat << EOF > environment.yml
name: my-analysis
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.6
  - numpy
  - dask
  - xarray=0.9
  - scikit-learn=0.19
  - matplotlib
  - seaborn
  - ipython
  - jupyter
  - cython
  - pip
  - pip:
      - git+https://github.com/nbren12/gnl@master=packages/gnl
      - git+https://github.com/nbren12/gnl@master=packages/xnoah
      - snakemake
EOF


# link the library folder
ln -s ../../lib .
