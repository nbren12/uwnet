#!/bin/sh
docker_cmd(){
    echo docker run -w /tmp -v $(pwd):/tmp uwnet_sam
}

cd OUT_3D/
for file in $(ls *.bin3D)
do
    base=${file%.*}
    ext=${file##*.}

    if [ -e $base.nc ]
    then
        echo "$base.nc exists"
    else
        echo "Converting $file to $base.nc"
        `docker_cmd` bin3D2nc $file > /dev/null
    fi
done
cd ..
python combine_3d_files.py
