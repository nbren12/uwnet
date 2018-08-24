# SAMUWgh
github-hosted UW Version of SAM, the System for Atmospheric Modeling

This repository was created by Peter Blossey to maintain a version of
SAM for use at the University of Washington with git as the version control
system.

## Tests

We use pFUnit to manage our unit tests. Currently pFUnit is assumed to be installed at the location pointed to by the environmental variable `PFUNIT`.

To compile and run the the tests, first compile the model using `./Build` and then run `make -C tests`.

## Running the model

This only works with docker at the moment. To build the docker image run

    docker build -t nbren12/samuwgh .

Then open a bash shell in this image by running

    docker/bash.sh

Build the model:

    ./Build
    
Run the model:

    ./SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM


## Running the model with python

There is a `dopython` namelist option that calls `SRC/python_caller.f90` at the very end of the time step. This subroutine pushes the relevant variables in SAM's state to a dictionary contained in the module `SRC/python/module.py`. The python code looks into this dictionary, computes an updated value of the things like U and SLI, and puts these values back in the dictionary. Then the fortran code reads the desired values back from this dictionary into the model state.

### Specifying a trained model

The path to the saved model is specified using the environmental variable `UWNET_MODEL`. Currently, the python model is only running a Held-Suarez benchmark, which is hardcoded at the moment.
