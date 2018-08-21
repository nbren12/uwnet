#!/bin/sh
# script to run from host
#
# Usage:
#

if [[ $# < 2 ]]
then
    echo "Run SAM model with specified neural network"
    echo ""
    echo "Usage:"
    echo "    run_sam_nn_docker.sh <model> <samcase>"
    echo ""
    exit 1
fi

image="nbren12/samuwgh"
exe=/case/NG1/run.sh
model=$1
CASEDIR=$2

cp -r $CASEDIR NG1
cp $model NG1/data.pkl

docker run \
       -i  \
       -v $(pwd):/case \
       -v /Users/noah/workspace/research/uwnet:/uwnet \
       -v /Users/noah/workspace/models/SAMUWgh:/sam \
       -e UWNET_OUTPUT_INTERVAL=20 \
       -e UWNET_DEBUG= \
       -e UWNET_MODEL=/case/NG1/data.pkl \
       -e PYTHONPATH=/uwnet:/sam/SRC/python \
       -w /case \
       $image $exe

# log data to mongodb
f=$(readlink $model)
mongo uwnet --eval "db.samrun.insert({model: \"$f\", path:\"\$(pwd)\"})"
