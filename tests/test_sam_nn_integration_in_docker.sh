#!/bin/sh

# exit on errors
set -e

rundir="test_sam"

function checkRun() {
    echo "Checking that SAM succesfully completed"
    cd $1
    n=$(ls OUT_3D/*.nc | wc -l)

    if [ $n -ne 3 ]
    then
       echo "Expected 3 netCDF files"
       exit 1
    fi

    echo "Run completed succesfully"
}

rm -rf $rundir

python src/criticism/run_sam_ic_nn.py \
       -ic assets/NG1/ic.nc \
       -nn assets/2018-12-23_model.pkl \
       -p assets/test_nn_interface.json \
       $rundir

echo "Running SAM"
setup/docker/execute_run.sh $rundir > test_sam/out 2> test_sam/err

checkRun $rundir
