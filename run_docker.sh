#!/bin/sh
runpath=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
docker run --privileged \
       -v $runpath:/rundir \
       -v /Users/noah/workspace/models/scam/scripts:/scripts \
       -it nbren12/cam $2
