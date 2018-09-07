#!/bin/sh

uwnet=/Users/noah/workspace/research/uwnet/
model=$(realpath u1)

model_dir_name=$(dirname $model)
sam_dir_name=$model_dir_name/SAM

# make running directory
if [[ -d $sam_dir_name ]]
then
    echo $sam_dir_name "already exists. Stopping execution"
    exit 1
fi

mkdir $sam_dir_name

echo "Running SAM in $sam_dir_name"

# setup necessary files for running SAM
cp -r $uwnet/assets/NG1 $sam_dir_name

# run same
(cd $sam_dir_name
 $uwnet/bin/run_sam_nn_docker.sh $model > out 2> err
)
