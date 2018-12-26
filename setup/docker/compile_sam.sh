#!/bin/sh
# Compile SAM run on docker
# must be run from root directory of uwnet

UWNET=$(pwd)

docker run -it \
    -v $UWNET:/uwnet \
    -v $UWNET/ext/sam:/opt/sam \
    -w /opt/sam \
    -e LOCAL_FLAGS=/uwnet/setup/docker/local_flags.mk \
    nbren12/uwnet:latest ./Build
