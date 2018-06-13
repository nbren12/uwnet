#!/bin/sh

revid=$(git rev-parse HEAD)

find data  -name '*.torch' -o -name '*.json' -o -name 'log.txt' | \
    tar -cvzf archives/models-$(date +%F)-$revid.tar.gz -T -
