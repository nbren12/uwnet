#!/bin/sh
docker_cmd(){
    echo docker run -w /tmp -v $(pwd):/tmp nbren12/sam
}

# cd OUT_2D/
# for file in $(ls *.2Dbin)
# do
#     `docker_cmd` 2Dbin2nc $file > /dev/null
# done
# cd ..

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
# bin3D2nc NGAqua_ngaqua_1_0000000001.bin3D
cd ..


cd OUT_STAT
`docker_cmd` stat2nc *.stat > /dev/null
cd ..

cd OUT_2D/
for file in $(ls *.2Dbin)
do
    base=${file%.*}
    ext=${file##*.}

    echo "Converting $file to $base.nc"
    `docker_cmd` 2Dbin $file > /dev/null
done
cd ..
