FROM debian:latest

RUN apt-get update && apt-get install -y \
    libnetcdff-dev libnetcdf-dev \
    build-essential gfortran \
    libmpich-dev  csh  \
    gdb curl git cmake

# compile sam utilities
ADD ext/sam/UTIL /sam/UTIL
RUN make -C /sam/UTIL

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}

# install pfunit
# RUN git clone git://git.code.sf.net/p/pfunit/code /tmp/pFUnit
# ADD http://superb-sea2.dl.sourceforge.net/project/pfunit/Source/pFUnit-3.2.9.tgz /tmp/
ADD ext/pFUnit-3.2.9.tgz /tmp/
ENV F90=gfortran
ENV F90_VENDOR=GNU
ENV PFUNIT=/opt/pfunit/pfunit-serial
RUN cd /tmp/pFUnit-3.2.9 && \
    make &&\
    make install INSTALL_DIR=${PFUNIT}

# add conda packages
RUN conda update -y conda
RUN conda install -y -c pytorch pytorch-cpu python=3.6 numpy toolz xarray \
                                                   netcdf4 scipy scikit-learn
RUN pip install zarr cffi click attrs dask pytest sacred jinja2
# ADD environment.yml /opt/environment.yml
# RUN cd /opt && conda env create
# ENV PATH=/miniconda/envs/uwnet/bin:${PATH}

# add callpy library
ADD ext/sam/ext/call_py_fort /opt/call_py_fort
ENV PYTHONPATH=/opt/call_py_fort/src/:/opt/call_py_fort/test:$PYTHONPATH
ENV CALLPY=/opt/call_py_fort
RUN cd /opt/call_py_fort/ && make install
ENV LD_LIBRARY_PATH=/usr/local/lib

# Install SAM Python modules
ENV PYTHONPATH=/opt/sam/SRC/python:${PYTHONPATH}
# ADD UWNET to path
ENV PYTHONPATH=/opt/sam/SCRIPTS/python/:/opt/:${PYTHONPATH}

RUN pip install f90nml

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
