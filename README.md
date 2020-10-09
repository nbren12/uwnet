# Machine learning approaches to convective parametrization
[![CircleCI](https://circleci.com/gh/VulcanClimateModeling/uwnet.svg?style=svg&circle-token=3c696b66d8dd0b789e012a14e93a6397c2cbe833)](https://circleci.com/gh/VulcanClimateModeling/uwnet)

## Documentation

The documentation is hosted on github pages: https://nbren12.github.io/uwnet/

## Setup

The software requirements for this project are more complicated than most
python data analysis projects because it uses several unique tools for running
python codes from an atmospheric model written in Fortran. However, the entire
workflow is containerized with [docker][docker], and can be run wherever docker
is installed.

### Obtaining permission to use SAM

The System for Atmospheric Modeling (SAM) is a key part of the pre-processing
pipeline and prognostic evaluation of this machine learning project, but it is
not necessary for offline evaluation or training.

If you want access to SAM, please email the author Marat Khairoutdinov (cc'ing
me) to ask for permission. Then, I can give you access to the slightly modified
version of SAM used for this project.

Once you have arranged this access, the SAM source code can be download to the
path `ext/sam` using

    git submodule --init --recursive


# Quickstart

## Setting up the environment

This project uses two dependency management systems. Docker is needed to run
the SAM model and SAM-related preprocessing steps. you do not need this if
you are only training a model from pre-processed data (the data in zenodo).
Poetry is a simpler pure python solution that should work for most common scenarios.

To use docker, you first need to build the image:

    make build_image

If you get an error `make: nvidia-docker: Command not found`, edit the
Makefile to have `DOCKER = docker` instead of `nvidia-docker`. (Assuming
docker is already installed.) Then, the docker environment can be entered by
typing

    make enter

This opens a shell variable in a docker container with all the necessary
software requirements.

To use poetry, you can install all the needed packages and enter a sandboxed
environment by running

    poetry install
    poetry shell

The instructions below assume you are in one of these environments

## Running the workflow

To run train the models, type
    
    snakemake -j <number of parallel jobs>

This will take a long time! To see all the steps and the corresponding commands
in this workflow, type

    snakemake -n -p

This whole analysis is specified in the Snakefile, which is the first place to
look.

To reproduce the plots for the Journal of Atmospheric science paper, run

    make jas2020


[docker]: https://www.docker.com/

# Evaluating performance

Evaluating ML Paramerizations is somewhat different than normal ML scoring.
Some useful metrics which work for xarray data are available in
`uwnet.metrics`. In particular `uwnet.metrics.r2_score` computes the ubiquitous
R2 score.

# Performing online tests

SAM has been modified to call arbitrary python functions within it's time stepping loop. These python functions accept a dictionary of numpy arrays as inputs, and store output arrays with specific names to this dictionary. Then SAM will pull the output contents of this dictionary back into Fortran and apply any computed tendency. 

To extend this, one first needs to write a suitable function, which can be tested using the data stored at `assets/sample_sam_state.pt`. The following steps explore this data

```ipython
In [5]: state =  torch.load("assets/sample_sam_state.pt")                                                                                 

In [6]: state.keys()                                                                                                                      
Out[6]: dict_keys(['layer_mass', 'p', 'pi', 'caseid', 'case', 'liquid_ice_static_energy', '_DIMS', '_ATTRIBUTES', 'total_water_mixing_ratio', 'air_temperature', 'upward_air_velocity', 'x_wind', 'y_wind', 'tendency_of_total_water_mixing_ratio_due_to_dynamics', 'tendency_of_liquid_ice_static_energy_due_to_dynamics', 'tendency_of_x_wind_due_to_dynamics', 'tendency_of_y_wind_due_to_dynamics', 'latitude', 'longitude', 'sea_surface_temperature', 'surface_air_pressure', 'toa_incoming_shortwave_flux', 'surface_upward_sensible_heat_flux', 'surface_upward_latent_heat_flux', 'air_pressure', 'air_pressure_on_interface_levels', 'dt', 'time', 'day', 'nstep'])

In [7]: qt = state['total_water_mixing_ratio']                                                                                            

In [8]: qt.shape                                                                                                                          
Out[8]: (34, 64, 128)

In [9]: state['sea_surface_temperature'].shape                                                                                            
Out[9]: (1, 64, 128)

In [10]: state['air_pressure_on_interface_levels'].shape                                                                                  
Out[10]: (35,)

In [11]: state['p'].shape                                                                                                                 
Out[11]: (34,)

In [12]: state['_ATTRIBUTES']                                                                                                             
Out[12]: 
{'liquid_ice_static_energy': {'units': 'K'},
 'total_water_mixing_ratio': {'units': 'g/kg'},
 'air_temperature': {'units': 'K'},
 'upward_air_velocity': {'units': 'm/s'},
 'x_wind': {'units': 'm/s'},
 'y_wind': {'units': 'm/s'},
 'tendency_of_total_water_mixing_ratio_due_to_dynamics': {'units': 'm/s'},
 'tendency_of_liquid_ice_static_energy_due_to_dynamics': {'units': 'm/s'},
 'tendency_of_x_wind_due_to_dynamics': {'units': 'm/s'},
 'tendency_of_y_wind_due_to_dynamics': {'units': 'm/s'},
 'latitude': {'units': 'degreeN'},
 'longitude': {'units': 'degreeN'},
 'sea_surface_temperature': {'units': 'K'},
 'surface_air_pressure': {'units': 'mbar'},
 'toa_incoming_shortwave_flux': {'units': 'W m^-2'},
 'surface_upward_sensible_heat_flux': {'units': 'W m^-2'},
 'surface_upward_latent_heat_flux': {'units': 'W m^-2'},
 'air_pressure': {'units': 'hPa'}}

In [13]: # tendence of total water mixing ratio expected units = g/kg/day                                                                 

In [14]: # tendence of tendency_of_liquid_ice_static_energy expected units =K/day                                           
```    

## Configuring SAM to call this function


Write uwnet.sam_interface.call_random_forest 
 
rule sam_run in Snakefile actually runs the SAM model. 
 
parameters as a json file are passed to src.sam.create_case via -p flag.  
 
Example parameters at assets/parameters_sam_neural_network.json.  
 
parameters['python'] configures which python function is called. 
