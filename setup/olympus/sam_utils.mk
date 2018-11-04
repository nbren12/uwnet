# Makefile to compile conversion utilities. You need to set include and library paths for NETCDF
#

# Docker platform  compiler flags

INC_NETCDF := /usr/include
LIB_NETCDF := /usr/lib

FF = ifort -fixed -O3 -pad -extend_source

ifeq ($(HOSTNAME),oly01)
   NCVERSION := 4.3.0-oly01
else
   NCVERSION := 4.3.0
endif

FFLAGS += -I/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/include
FFLAGS_NOOPT += -I/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/include
LDFLAGS = -L/usr/lib64 -L/usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/lib -Wl,-rpath /usr/local/modules/netcdf/${NCVERSION}/intel/13.1.1/lib -lnetcdff -lnetcdf

#FFLAGS = -g -fcheck=all

VPATH = ./SRC

all: bin2D2nc bin3D2nc 2Dbin2nc 2Dbin2nc_mean bin3D2nc_mean com3D2bin 2Dcom2nc 2Dcom2nc_mean com3D2nc com3D2nc_mean com2D2nc stat2nc isccp2nc modis2nc misr2nc com3D2nc_sep 2Dbin2nc_sep 2Dcom_sep2one 2Dbin_sep2one com3D_sep2one bin3D_sep2one glue_movie_raw

.f:   
	$(FF) $(FFLAGS) -o $@ -I./SRC $< ./SRC/hbuf_lib.f ./SRC/cape.f ./SRC/cin.f $(LDFLAGS) 

clean: 
	rm bin* com* stat* 2* isccp* modis* misr* *.o glue*
