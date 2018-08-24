#!/bin/bash

# cd OUT_2D/
# for file in $(ls *.2Dbin)
# do
#     2Dbin2nc $file > /dev/null
# done
# cd ..

cd OUT_3D/
for file in $(ls *.bin3D)
do
    name_no_ext=${file%.*}
    if ! [ -e $name_no_ext.nc ]; then
        echo "Converting $file"
        bin3D2nc $file > /dev/null
    fi
done
# bin3D2nc NGAqua_ngaqua_1_0000000001.bin3D
cd ..


cd OUT_STAT
stat2nc *.stat > /dev/null
cd ..
