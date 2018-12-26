#!/bin/sh
# Compile SAM run on docker
# must be run from root directory of uwnet

UWNET=$(pwd)
docker-compose run -w /opt/sam sam  ./Build
