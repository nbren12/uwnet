#!/bin/sh

rundir="test_sam"

checkRun()
{
    echo "Checking that SAM succesfully completed"
    cd $1
    n=$(ls OUT_3D/*.bin3D | wc -l)

    if [ $n -ne 3 ]
    then
       echo "Expected 3 bin3D files"
       exit 1
    fi

    echo "Run completed succesfully"
}

rm -rf $rundir

SAM=/opt/sam/OBJ/128x64x34/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

python -m src.sam.create_case \
       -ic assets/NG1/ic.nc \
       -nn assets/2018-12-23_model.pkl \
       -p assets/test_nn_interface.json \
       $rundir

echo "Running SAM"
(
    cd $rundir
    $SAM || exit 0
)

checkRun $rundir

echo "Cleaning up"
rm -rf $rundir
