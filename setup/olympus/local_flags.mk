# Determine platform
CALL_PY_ROOT ?= /usr/local


FF77 = mpif90 -c -fixed -extend_source -r8
FF90 = mpif90 -c -r8
CC = mpicc -c -DLINUX


FFLAGS = -O2 -fp-model source
# FFLAGS_NOOPT = -O0 -g -ftrapuv -check all -fp-model source
FFLAGS_NOOPT = -O0 -g -ftrapuv -check all -fp-model source -traceback
# FFLAGS = -g -ftrapuv -check all # for debugging
# FFLAGS = -g  # for debugging
# FFLAGS = ${FFLAGS_NOOPT}

ifeq ($(HOSTNAME),oly01)
   NCVERSION := 4.3.0-oly01
else
   NCVERSION := 4.3.0
endif

FFLAGS += -I/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/include
FFLAGS_NOOPT += -I/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/include
LD = mpif90
LDFLAGS = -L/usr/lib64 -L/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/lib -Wl,-rpath /usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/lib -lnetcdff -lnetcdf

# add callpy flags
FFLAGS += -I${CALL_PY_ROOT}/include
LDFLAGS += -L${CALL_PY_ROOT}/lib -lcallpy -lplugin  -Wl,-rpath ${CALL_PY_ROOT}/lib
