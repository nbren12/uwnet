#!/bin/sh

# SAM segfaults on saving states without this
ulimit -s unlimited

cd $1
./run.sh
