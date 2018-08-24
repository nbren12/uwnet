#!/bin/sh
# script to run from host

image="nbren12/samuwgh"

docker run \
       -i  \
       -v $(pwd):/case \
       -v /Users:/Users \
       -v /Users/noah/workspace/research/uwnet:/uwnet \
       -v /Users/noah/workspace/models/SAMUWgh:/sam \
       -e UWNET_OUTPUT_INTERVAL=20 \
       -e UWNET_DEBUG= \
       -w /case \
       $image /case/NG1/run.sh $@

# log data to mongodb
f=$(readlink $model)
mongo uwnet --eval "db.samrun.insert({model: \"$f\", path:\"\$(pwd)\"})"
