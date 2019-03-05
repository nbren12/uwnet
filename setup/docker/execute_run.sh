#!/bin/sh
# Execute SAM run on docker

run=$1
docker-compose run -w /run -v $run:/run sam ./run.sh $run

