# Makefile to compile conversion utilities. You need to set include and library paths for NETCDF
#

# Docker platform  compiler flags

INC_NETCDF := /usr/include
LIB_NETCDF := /usr/lib

FF = gfortran -ffixed-form -ffixed-line-length-0


FFLAGS = -O3 
#FFLAGS = -g -fcheck=all

FFLAGS += -I${INC_NETCDF}
LD = mpif90
LDFLAGS = -L${LIB_NETCDF} -lnetcdf -lnetcdff
VPATH = ./SRC

all: bin2D2nc bin3D2nc 2Dbin2nc 2Dbin2nc_mean bin3D2nc_mean com3D2bin 2Dcom2nc 2Dcom2nc_mean com3D2nc com3D2nc_mean com2D2nc stat2nc isccp2nc modis2nc misr2nc com3D2nc_sep 2Dbin2nc_sep 2Dcom_sep2one 2Dbin_sep2one com3D_sep2one bin3D_sep2one glue_movie_raw
	cp $^ /usr/local/bin/

.f:   
	$(FF) $(FFLAGS) -o $@ -I./SRC $< ./SRC/hbuf_lib.f ./SRC/cape.f ./SRC/cin.f $(LDFLAGS) 

clean: 
	rm bin* com* stat* 2* isccp* modis* misr* *.o glue*
