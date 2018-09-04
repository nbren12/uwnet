#!/bin/sh
# script to run from docker image

cp $1 data.pkl

export UWNET_OUTPUT_INTERVAL=120
export UWNET_DEBUG=
export UWNET_MODEL=data.pkl


for file in OUT_3D OUT_2D OUT_STAT RESTART
do
    ! [ -d $file ] && mkdir $file
done
ln -s /sam/RUNDATA .

/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM


echo NG1 > CaseName
! [ -h RUNDATA ] && ln -s /sam/RUNDATA .

(! [ -x $exe ] && /sam/Build)

/sam/docker/cleancase.sh NG1_test
/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM
/sam/docker/convert_files.sh
