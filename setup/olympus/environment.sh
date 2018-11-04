. /etc/bashrc

# SAM segfaults on saving states without this
ulimit -s unlimited

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
export PATH=$UWNET/ext/sam/UTIL:$PATH
