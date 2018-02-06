#!/bin/sh

mkdir -p inputdata

cat << EOF >  files_needed.txt
inputdata/atm/cam/chem/trop_mozart_aero/aero/aero_1.9x2.5_L26_2000clim_c091112.nc
inputdata/atm/cam/chem/trop_mozart/ub/clim_p_trop.nc
inputdata/atm/cam/inic/gaus/cami_0000-01-01_64x128_L30_c090102.nc
inputdata/atm/cam/topo/USGS-gtopo30_64x128_c050520.nc
inputdata/atm/cam/rad/abs_ems_factors_fastvx.c030508.nc
inputdata/atm/cam/ozone/ozone_1.9x2.5_L26_2000clim_c091112.nc
inputdata/atm/cam/physprops/sulfate_camrt_c080918.nc
inputdata/atm/cam/physprops/dust1_camrt_c080918.nc
inputdata/atm/cam/physprops/dust2_camrt_c080918.nc
inputdata/atm/cam/physprops/dust3_camrt_c080918.nc
inputdata/atm/cam/physprops/dust4_camrt_c080918.nc
inputdata/atm/cam/physprops/bcpho_camrt_c080918.nc
inputdata/atm/cam/physprops/bcphi_camrt_c080918.nc
inputdata/atm/cam/physprops/ocpho_camrt_c080918.nc
inputdata/atm/cam/physprops/ocphi_camrt_c080918.nc
inputdata/atm/cam/physprops/ssam_camrt_c080918.nc
inputdata/atm/cam/physprops/sscm_camrt_c080918.nc
EOF


while read line
do
    file=$(basename $line)
    folder=$(dirname $line)

    if [ ! -e $folder ]
    then
        mkdir -p $folder
    fi

    if [ ! -e $folder/$file ]
    then
        echo "$folder/$file not present downloading"
        pushd $folder > /dev/null
        svn export --username guestuser https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/$folder/$file
        popd > /dev/null
    else
        echo "$folder/$file already present"
    fi

done <  files_needed.txt

rm files_needed.txt
