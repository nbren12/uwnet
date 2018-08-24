#!/bin/sh

export UWNET_DEBUG_INTERVAL=20
export UWNET_DEBUG=True
export UWNET_MODEL=/uwnet/13_actual_constraint/5.pkl
# export SAM_TARGET_DATA=/data/time_chunked
# export SAM_FORCING_OUTPUT=/data/forcings

export PYTHONPATH=/uwnet:$PYTHONPATH

ln -s /sam/OUT* .
ln -s /sam/RUNDATA .

/sam/docker/cleancase.sh NGCalcForcing_1
echo NGCalcForcing > CaseName
/sam/SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM
