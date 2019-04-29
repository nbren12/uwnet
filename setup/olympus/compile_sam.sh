#!/bin/bash

export LOCAL_FLAGS=$UWNET/setup/olympus/local_flags.mk
cd ext/sam

export NX=128
export NY=64
export NZ=34
export NSUBX=1
export NSUBY=1
./Build


export NX=512
export NY=256
export NZ=34
export NSUBX=2
export NSUBY=2
./Build
