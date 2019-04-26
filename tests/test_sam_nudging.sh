#!/bin/sh

# exit on errors
set -e

rundir="test_sam"

checkRun()
{
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
       -p assets/test_nudge.json \
       $rundir

echo "Running SAM"
(
    cd $rundir
    ./run.sh
)
checkRun $rundir

echo "Cleaning up"
rm -rf $rundir
