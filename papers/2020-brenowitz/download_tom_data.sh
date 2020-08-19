#!/bin/bash

set -x

[[ -d data/PKL_DATA ]] && exit 0

rm -rf data/tom
git clone https://github.com/tbeucler/CBRAIN-CAM/
mv CBRAIN-CAM/notebooks/tbeucler_devlog/PKL_DATA data/
rm -rf CBRAIN-CAM
