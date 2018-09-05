#!/bin/bash
# script to run from host

image="nbren12/uwnet"
docker_args=-i

# check if running in TTY
if [[ $- == *i* ]]
then
    docker_args=-it
fi



docker run \
       $docker_args  \
       -v $(pwd):/case \
       -v /Users:/Users \
       -v /Users/noah/workspace/research/uwnet:/uwnet \
       -v /Users/noah/workspace/research/uwnet/ext/sam:/opt/sam \
       -w /case \
       $image /case/NG1/run.sh $@

# log data to mongodb
# f=$(readlink $model)
# mongo uwnet --eval "db.samrun.insert({model: \"$f\", path:\"\$(pwd)\"})"
