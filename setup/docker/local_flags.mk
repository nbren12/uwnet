INC_NETCDF := /usr/include
LIB_NETCDF := /usr/lib

DEBUG_FLAGS= -g -fbacktrace -ffpe-trap=invalid,denormal,zero,overflow,underflow

FF77 = mpif77 -c -ffixed-form -ffixed-line-length-0 -fdefault-real-8
FF90 = mpif90  -ffree-form -ffree-line-length-0 -fdefault-real-8
CC = mpicc -c -DLINUX

CPPFLAGS = -DNZ=$(NZ) -DNX=$(NX) -DNY=$(NZ)

FFLAGS = -O3
#FFLAGS = -g -fcheck=all

PYTHON_LIB_DIR=/sam/SRC/python
PYTHON_LIB=$(PYTHON_LIB_DIR)/libpython.so
PYTHON_LIB_SRC=$(PYTHON_LIB_DIR)/builder.py

FFLAGS += -c -I${INC_NETCDF} -I/usr/local/include/
LD = mpif90
LDFLAGS = -L${LIB_NETCDF} -lnetcdf -lnetcdff -lcallpy -lplugin
