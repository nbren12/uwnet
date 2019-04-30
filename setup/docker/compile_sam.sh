#!/bin/sh
# Compile SAM run on docker
# must be run from root directory of uwnet

UWNET=$(pwd)

docker-compose run -w /opt/sam -e NX=512 -e NY=256 -e NZ=34 \
    -e NSUBX=2 -e NSUBY=2 sam ./Build

docker-compose run -w /opt/sam -e NX=128 -e NY=64 -e NZ=34 \
    -e NSUBX=1 -e NSUBY=1 sam ./Build
