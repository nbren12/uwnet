# The System for Atmospheric Modeling

The System for atmospheric modeling (SAM) solves a simplified version of the fluid equations governing the atmosphere known as the anelastic equations. Traditionally, this model is used for high resolution (< 4km) simulations of clouds and convection, but it can also be run in near-global domains at coarse resolution (Bretherton and Khairoutdinov, 2015).

## Prognostic Variables

The main prognostic variables that the System for Atmospheric Modeling uses are:

1. Liquid/ice static energy variable (SLI)
2. Non-precipitating water mixing ratio
3. The wind fields are collocated on an [Arakawa C-grid](https://en.wikipedia.org/wiki/Arakawa_grids#Arakawa_C-grid) 


For more details, see the appendix of Khairoutdinov and Randall (2003).

## Running SAM via docker

We run SAM using docker, but if you are not familiar with docker, we have
written scripts that abstract away most docker specific details. Therefore, it
should be possible to compile SAM, configure an initial value experiement, and
execute a similution without knowing any docker.

### Setting up the docker image and compiling SAM

To prepare the docker image and compile the SAM, enter the following commands in the project root directory
```
make build_image
make compile_sam
```
The `build_image` installs all the necessary dependencies in the docker image, but does not compile the SAM model. We do not do this because the SAM source code has been evolving rapidly in this project. Instead, we use "bind-mounts" to make the `ext/sam/` directory available to the docker image, and then compile SAM in place. This means, that 
1. The SAM executable is placed in the `ext/sam` folder of host system, and 
2. any changes to the SAM source in this folder can be quickly recompiled using `make compile_sam`.

### Setting up a SAM NG-Aqua prediction case

The SAM model is configured by setting up a so-called "case" directory. This
case directory needs to have a specific format that is described in the SAM User
Guide. UWNET provides a script named `src/criticism/run_sam_ic_nn.py` for
generating NG-Aqua prediction cases. For instance, the following command will
setup a coupled SAM+NN simulation initialized with the first available time step
of the NG-Aqua:
```
src/criticism/run_sam_ic_nn.py -nn <model file>.pkl -t 0 -p assets/parameters2.json <run directory>
```
Instead of using Fortran namelists to configure the model, this script uses a json file `assets/parameters2.json` with a very similar structure. Internally, this json file is parsed and saved into the Fortran namelist that SAM requires.

That command generates a directory structure like this:
```
<run directory>
├── CASE
│   ├── CASE_control.nml
│   ├── grd
│   ├── ic.nc
│   ├── prm
│   └── snd
├── CaseName
├── OUT_2D
├── OUT_3D
├── OUT_STAT
├── RESTART
├── RUNDATA -> /opt/sam/RUNDATA
├── model.pkl
├── run.sh

```
The SAM User Guide discusses this structure in more detail. The main additions that our script does are
- generates the initial condition file `CASE/ic.nc`
- copies the desired model to `model.pkl`
- generates the parameter namelist `CASE/prm`
- creates a script (`run.sh`) for running the model and converting its outputs to netCDF 

To learn more about how the SAM model is configured to run with the neural network, you can examine the 
`run.sh` and `CASE/prm` files.

### Running a case

You can run a simulation described by a case by executing

    setup/docker/execute_sam.sh <path to run directory>
    
All this script does is execute the `run.sh` script within docker.


## References

- Bretherton, C. S., & Khairoutdinov, M. F. (2015). Convective self-aggregation feedbacks in near-global cloud-resolving simulations of an aquaplanet. Journal of Advances in Modeling Earth Systems, 7(4), 1765–1787. Retrieved from http://onlinelibrary.wiley.com/doi/10.1002/2015MS000499/full

- Khairoutdinov, M. F., & Randall, D. A. (02/2003). Cloud Resolving Modeling of the ARM Summer 1997 IOP: Model Formulation, Results, Uncertainties, and Sensitivities. Journal of the Atmospheric Sciences, 60(4), 607–625. https://doi.org/10.1175/1520-0469(2003)060<0607:CRMOTA>2.0.CO;2
