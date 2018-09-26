# Makefile for various platforms
# Execute using Build csh-script only!
# Used together with Perl scripts in SRC/SCRIPT
# (C) 2005 Marat Khairoutdinov
#------------------------------------------------------------------
# uncomment to disable timers:
#
#NOTIMERS=-DDISABLE_TIMERS
#-----------------------------------------------------------------

SAM = SAM_$(ADV_DIR)_$(SGS_DIR)_$(RAD_DIR)_$(MICRO_DIR)

# Determine platform
PLATFORM := $(shell uname -s)

# Docker platform  compiler flags

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


#----------------------------------
#----------------------------------------------
# you dont need to edit below this line


#compute the search path
dirs := . $(shell cat Filepath)
VPATH    := $(foreach dir,$(dirs),$(wildcard $(dir)))

.SUFFIXES:
.SUFFIXES: .f .f90 .c .o



all: $(SAM_DIR)/$(SAM)


SOURCES   := $(shell cat Srcfiles)

Depends: Srcfiles Filepath
	$(SAM_SRC)/SCRIPT/mkDepends Filepath Srcfiles > $@

Srcfiles: Filepath
	$(SAM_SRC)/SCRIPT/mkSrcfiles > $@

OBJS      := $(addsuffix .o, $(basename $(SOURCES)))

$(SAM_DIR)/$(SAM): $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

tests :

test_read_sound: $(SAM_SRC)/TESTS/test_read_sound.f90 read_netcdf_3d.o grid.o task_util_MPI.o
	$(FF90)  $(LDFLAGS) -I${INC_NETCDF} -o $@  $^

.f90.o:
	${FF90}  ${FFLAGS} $<
.f.o:
	${FF77}  ${FFLAGS} $<
.c.o:
	${CC}  ${CFLAGS} -I$(SAM_SRC)/TIMING $(NOTIMERS) $<

domain.o : domain.f90
	${FF90} ${FFLAGS} ${CPPFLAGS} -cpp $<

include Depends



clean:
	rm ./OBJ/*
