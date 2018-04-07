#!/bin/sh


find data  -name '*.torch' -o -name '*.json'\
          -o -path 'data/output/columns.nc'\
          -o -path 'data/output/test_error.nc'\
          -o -path 'data/output/scam.nc'\
          -o -path 'data/processed/forcings.nc'\
          -o -path 'data/processed/inputs.nc' | \
    tar -cvzf data-$(date +%F).tar.gz -T -
