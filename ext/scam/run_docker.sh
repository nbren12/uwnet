#!/bin/sh
runpath=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
docker run --privileged \
    -v cesm_data:/inputdata \
    -v $runpath:/rundir \
    -it nbren12/cam $2
