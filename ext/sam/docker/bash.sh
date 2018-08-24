#!/bin/sh

docker run -it --privileged \
       -v $(pwd):/sam \
       -v $UWNET:/uwnet \
       nbren12/samuwgh \
       bash
