#!/bin/sh

export UWNET_DEBUG_INTERVAL=120
export UWNET_DEBUG=True
export UWNET_MODEL=/uwnet/data/samNN/NG1/data.pkl
export PYTHONPATH=/uwnet:$PYTHONPATH


./docker/cleancase.sh NG1_2018-08-15-A
./SAM_ADV_MPDATA_SGS_TKE_RAD_CAM_MICRO_SAM1MOM
./docker/convert_files.sh
