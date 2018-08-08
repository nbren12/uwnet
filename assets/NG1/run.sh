#!/bin/sh
# script to run from docker image

exe=/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM

for file in OUT_3D OUT_2D OUT_STAT RESTART
do
    ! [ -d $file ] && mkdir $file
done


echo NG1 > CaseName
! [ -h RUNDATA ] && ln -s /sam/RUNDATA .

/sam/docker/cleancase.sh NG1_test

cd /sam
! [ -x $exe ] && /sam/Build
cd -
$exe
