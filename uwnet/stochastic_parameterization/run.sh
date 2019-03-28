#!/bin/sh

export UWNET_MODEL=model.pkl


ln -s /opt/sam/RUNDATA .
# /opt/sam/docker/cleancase.sh CASE
/opt/sam/SAM_*
/opt/sam/docker/convert_files.sh