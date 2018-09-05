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
export UWNET_OUTPUT_INTERVAL=0
export UWNET_DEBUG=
export UWNET_MODEL=$model
export PATH=/opt/sam/docker:/opt/sam:$PATH
export PYTHONPATH=/uwnet:$PYTHONPATH

exe=SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

# build the model if needed
(
    cd /opt/sam
    ./Build
)

# create directories needed for running
for file in OUT_3D OUT_2D OUT_STAT RESTART
do
    ! [ -d $file ] && mkdir $file
done

! [ -h RUNDATA ] && ln -s /opt/sam/RUNDATA .

# setup/clean case
echo NG1 > CaseName
cleancase.sh NG1_test
rm -rf dbg.zarr

# run the model
$exe

convert_files.sh

(
    cd OUT_2D/
    for file in $(ls *.2Dbin)
    do
        name_no_ext=${file%.*}
        if ! [ -e $name_no_ext.nc ]; then
            echo "Converting $file"
            2Dbin2nc $file
        fi
    done
)
