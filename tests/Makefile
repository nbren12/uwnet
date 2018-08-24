all: tests.x
	./tests.x

include $(PFUNIT)/include/base.mk

LIBS = -lpfunit -L$(PFUNIT)/lib

INC_NETCDF := /usr/include
LIB_NETCDF := /usr/lib

FF77 = mpif77 -c -ffixed-form -ffixed-line-length-0
FF90 = mpif90  -ffree-form -ffree-line-length-0
CC = mpicc -c -DLINUX

CPPFLAGS = -DNZ=$(NZ) -DNX=$(NX) -DNY=$(NZ)

PYTHON_LIB_DIR=/sam/SRC/python
PYTHON_LIB=$(PYTHON_LIB_DIR)/libpython.so
PYTHON_LIB_SRC=$(PYTHON_LIB_DIR)/builder.py

FFLAGS += -I${INC_NETCDF}
LD = mpif90
LDFLAGS = -L$(PYTHON_LIB_DIR) -Wl,-rpath=$(PYTHON_LIB_DIR) \
	-L${LIB_NETCDF} -lnetcdf -lnetcdff -lplugin $(LIBS)



PFS = $(wildcard *.pf)
OBJS = $(PFS:.pf=.o)

FILTER= ../OBJ/main.o

SAM_OBJS = $(filter-out $(FILTER), $(wildcard ../OBJ/*.o))


%.F90: %.pf
	$(PFUNIT)/bin/pFUnitParser.py $<  $@

%$(OBJ_EXT): %.F90
	$(FC) -c $(FFLAGS) $(FPPFLAGS) $<

tests.x : testSuites.inc	helloWorld.F90
	$(FF90) -o $@ -I../OBJ/ -I./ -I$(PFUNIT)/mod -I$(PFUNIT)/include \
			$(PFUNIT)/include/driver.F90 helloWorld.F90 \
			$(SAM_OBJS) $(LDFLAGS) $(FFLAGS) 
