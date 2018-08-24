#!/bin/bash

mv SRC/domain.f90 SRC/domain.tmp
cp domain.f90 SRC/.

for ADV in "MPDATA" # "UM5"
  do
  echo ${ADV}
  for RAD in "RRTM" "CAM"
    do
    echo ${RAD}
    for MICRO in "M2005" "THOM"
      do
      echo ${MICRO}

      rm -f Build Build.[1-2]
      sed s/xxx/${ADV}/ Build.base > Build.1
      sed s/yyy/${RAD}/ Build.1 > Build.2
      sed s/zzz/${MICRO}/ Build.2 > Build
      chmod +x Build

      rm -f SAM_* # remove old executable(s)
      ./Build # build new executable

      for CASE in "KWAJEX3" # "KWAJEX2"  
        do
        echo ${CASE}

	# set up CaseName file
        rm -f CaseName
	sed s/aaa/${CASE}/ CaseName.base > CaseName

        # set up prm file
	rm -f ${CASE}/prm ${CASE}/prm.[1-2]
        sed s/xxx/${ADV}/ ${CASE}/prm.base > ${CASE}/prm.1
        sed s/yyy/${RAD}/ ${CASE}/prm.1 > ${CASE}/prm.2
        sed s/zzz/${MICRO}/ ${CASE}/prm.2 > ${CASE}/prm

        time mpirun -np 8 ./SAM_ADV_${ADV}_SGS_TKE_RAD_${RAD}_MICRO_${MICRO} > log_2d_${CASE}_${MICRO}_${RAD}_${ADV}
	cat timing.0 >> log_2d_${CASE}_${MICRO}_${RAD}_${ADV}

      done # for CASE

    done # for MICRO

  done # for RAD

done # for ADV

rm -f SRC/domain.f90
cp SRC/domain.tmp SRC/domain.f90

exit 0
