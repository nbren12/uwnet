#!/bin/bash
# script to run from docker image

if [[ $# < 1 ]]
then
    echo "Run SAM model with specified neural network"
    echo ""
    echo "Usage:"
    echo "    run.sh <model>"
    echo ""
    exit 1
else
    model=$1
fi

# setup up environment
export PYTHONPATH=/uwnet:/sam/SRC/python:$PYTHONPATH
exe=/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM
export UWNET_MODEL=$model

# build the model if needed
(
    cd /sam
    ! [ -x $exe ] && /sam/Build
)

# create directories needed for running
for file in OUT_3D OUT_2D OUT_STAT RESTART
do
    ! [ -d $file ] && mkdir $file
done

! [ -h RUNDATA ] && ln -s /sam/RUNDATA .

# setup/clean case
echo NG1 > CaseName
/sam/docker/cleancase.sh NG1_test
rm -rf dbg.zarr

# run the model
$exe

/sam/docker/convert_files.sh
