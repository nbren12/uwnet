#!/bin/bash

. /etc/bashrc
# Make module command available
module purge
module load intel/17.0.4
module load openmpi/1.6.4

export UWNET=$(pwd)
export CALLPY_SRC=$(pwd)/ext/sam/ext/call_py_fort
export PFUNIT=${CONDA_PREFIX}
export CALL_PY_ROOT=$UWNET/ext/callpy
export LD_LIBRARY_PATH=$UWNET/ext/callpy/lib:$LD_LIBRARY_PATH

# NETCDF LIBDIR
NETCDF_ROOT=$(dirname $(dirname $(which mpif90)))
export LD_LIBRARY_PATH=$NETCDF_ROOT/lib/:$LD_LIBRARY_PATH

# Install pfunit
(
    export F90=ifort
    export F90_VENDOR=Intel

    mkdir -p build/
    cd build/
    tar xzf $UWNET/ext/pFUnit-3.2.9.tgz && \
        cd pFUnit-3.2.9 && \
        make &&\
        make install INSTALL_DIR=${CONDA_PREFIX}
)


# Install my callpy library
(
    rm -rf build/callpy
    mkdir -p build/callpy
    mkdir -p ext/callpy
    cd build/callpy
    export FC=ifort
    cmake $CALLPY_SRC -DCMAKE_INSTALL_PREFIX=$UWNET/ext/callpy \
      -DCMAKE_BUILD_TYPE=Release
    make install
)


# Compile SAM
(
    export LOCAL_FLAGS=$UWNET/setup/olympus/local_flags.mk
    cd ext/sam
    ./Build
)

# Setup SAM scripts
bash setup/olympus/compile_sam.sh

# setup sam post processing tools bin2d...
sh setup/olympus/compile_sam_utils.sh
