#!/bin/sh

# downloading data
echo '************************************************************'
echo "Downloading data"
echo '************************************************************'
/scripts/download_inputdata.sh

nml=`cat /rundir/namelist.txt`
# configure
echo '************************************************************'
echo "Configuring SAM"
echo '************************************************************'
$camcfg/build-namelist -namelist "$nml" -config /bld/config_cache.xml -test > configure_output

if [ $? -ne 0 ]; then
    echo "Configuration failed. Showing output"
    cat configure_output
fi

echo '************************************************************'
echo "Editing start time/date"
echo '************************************************************'
sed -i.bak 's/\ *start_ymd.*/start_ymd = 990411/' drv_in

echo '************************************************************'
echo "Running CAM"
echo '************************************************************'
/bld/cam | tee run_output | grep DATE 

if [ $? -ne 0 ]; then
    cat run_output
fi

