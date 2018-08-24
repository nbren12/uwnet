#!/bin/sh

export UWNET_DEBUG_INTERVAL=20
export UWNET_DEBUG=True
export UWNET_MODEL=/uwnet/16_new_forcing/1.pkl

export PYTHONPATH=/uwnet:$PYTHONPATH


echo NG1 > CaseName
./SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM
