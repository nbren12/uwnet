# Machine learning approaches to convective parametrization
[![CircleCI](https://circleci.com/gh/VulcanTechnologies/uwnet.svg?style=svg&circle-token=3c696b66d8dd0b789e012a14e93a6397c2cbe833)](https://circleci.com/gh/VulcanTechnologies/uwnet)[![Build Status](https://travis-ci.org/nbren12/uwnet.svg?branch=master)](https://travis-ci.org/nbren12/uwnet)

## Documentation

The documentation is hosted on github pages: https://nbren12.github.io/uwnet/

## Setup

The software requirements for this project are more complicated than most
python data analysis projects because it uses several unique tools for running
python codes from an atmospheric model written in Fortran. However, the entire
workflow is containerized with [docker][docker], and can be run wherever docker
is installed.

# Quickstart

From the project's root directory the docker environment can be entered by
typing

    make enter

This opens a shell variable in a docker container with all the necessary
software requirements.

To run the whole workflow from start to finish, type
    
    snakemake -j <number of parallel jobs>

This will take a long time! To see all the steps and the corresponding commands
in this workflow, type

    snakemake -n -p

This whole analysis is specified in the Snakefile, which is the first place to
look.


[docker]: https://www.docker.com/

# Evaluating performance

Evaluating ML Paramerizations is somewhat different than normal ML scoring.
Some useful metrics which work for xarray data are available in
`uwnet.metrics`. In particular `uwnet.metrics.r2_score` computes the ubiquitous
R2 score.
