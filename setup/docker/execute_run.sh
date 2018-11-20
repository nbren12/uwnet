#!/bin/sh
# Execute SAM run on docker

UWNET=$(pwd)
run=$(realpath $1)

docker run -it \
    -v /Users:/Users  \
    -v $UWNET/uwnet:/opt/uwnet \
    -v $UWNET/ext/sam:/opt/sam \
    -w "$run" \
    -e LOCAL_FLAGS=$UWNET/setup/docker/local_flags.mk \
    nbren12/uwnet:latest ./run.sh
