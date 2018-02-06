FROM ubuntu

RUN apt-get update && apt-get install -y \
  build-essential \
  libnetcdf-dev \
  libnetcdff-dev \
  gdb git subversion \
  gfortran


# Download and build CAM in single column mode 
# without chemistry
ENV INC_NETCDF /usr/include
ENV LIB_NETCDF /usr/lib/x86_64-linux-gnu
ENV CAM_ROOT /cesm
ENV camcfg /cesm/models/atm/cam/bld

RUN git clone https://github.com/earthers/cesm-1_2_2 /cesm
RUN mkdir -p /bld
RUN cd /bld && \
  $camcfg/configure -scam  -nlev 30 -fc gfortran \
       -nospmd -nospm -dyn eul \
       -res 64x128 -fflags '-fbacktrace -fcheck=all' \
       -ocn aquaplanet \ 
       -chem none  # -debug

RUN cd /bld && make -j 2 > compile_output 2>&1

# Add scripts
ADD  /scripts /scripts

# Download the input data
ADD /inputdata /inputdata
ENV CSMDATA /inputdata


WORKDIR /rundir
# CMD /bld/cam | tee run_output 
CMD /scripts/configure.sh
