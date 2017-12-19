#!/bin/sh

if [ ! -d data ]
then
    wget atmos.washington.edu/~nbren12/data/id/3.tar.gz
    tar xzf 3.tar.gz
    rm 3.tar.gz
    mv 3 data
fi
