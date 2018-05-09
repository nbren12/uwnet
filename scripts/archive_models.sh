#!/bin/sh

revid=$(git rev-parse HEAD)

find data  -name '*.torch' -o -name '*.json' | \
    tar -cvzf archives/models-$(date +%F)-$revid.tar.gz -T -
