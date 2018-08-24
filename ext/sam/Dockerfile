FROM debian:latest

RUN apt-get update && apt-get install -y \
    libnetcdff-dev libnetcdf-dev \
    build-essential gfortran \
    libmpich-dev  csh  \
    gdb curl git cmake

# compile sam utilities
ADD UTIL /sam/UTIL
RUN make -C /sam/UTIL

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

# install pfunit
ADD http://superb-sea2.dl.sourceforge.net/project/pfunit/Source/pFUnit-3.2.9.tgz /tmp/
ENV F90=gfortran
ENV F90_VENDOR=GNU
ENV PFUNIT=/opt/pfunit/pfunit-serial
RUN cd /tmp && \
    tar xzf pFUnit-3.2.9.tgz && \
    cd pFUnit-3.2.9 && \
    make &&\
    make install INSTALL_DIR=${PFUNIT}

# add conda packages
RUN conda update -y conda
RUN conda install -y -c pytorch \
    cffi numpy pytorch-cpu \
    torchvision-cpu toolz \
    xarray dask
RUN pip install zarr attrs

# add SAM
ADD . /sam

# add callpy library
ADD ext/call_py_fort /opt/call_py_fort
ENV PYTHONPATH=/opt/call_py_fort/src/:/opt/call_py_fort/test:$PYTHONPATH
ENV CALLPY=/opt/call_py_fort
RUN cd /opt/call_py_fort/ && make install
ENV LD_LIBRARY_PATH=/usr/local/lib

# Install UWNET code
ENV PYTHONPATH=/uwnet:$PYTHONPATH

# Install SAM Python modules
ADD SCRIPTS /sam/SCRIPTS
ENV PYTHONPATH=/sam/SRC/python:${PYTHONPATH}
RUN pip install -e /sam/SCRIPTS/python/

# compile SAM
RUN cd /sam && ./Build

# add CalcForcing
ADD NGCalcForcing /case/NGCalcForcing
ADD NG1 /case/NG1
ADD run.sh /case/run.sh

WORKDIR /case
CMD /sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

