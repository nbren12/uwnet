#! /bin/csh -f

set filename = /gpfs/scratch1/marat/SAM6.7_WMO/OUT_3D/SuperCell_512x256x64PB_1000m_4s_QI0-5L_64_00000
set N1 = 150
set N2 = 3600
set NN = 150

while ($N1 <= $N2)

 echo $N1
 set M = ""
 if($N1 < 10) then
  set M = "0000"
 else if($N1 < 100) then
  set M = "000"
 else if($N1 < 1000) then
  set M = "00"
 else if($N1 < 10000) then
  set M = "0"
 endif

echo $M

set f = $filename$M$N1.com3D

../com3D_sep2one  $f

@ N1 = $N1 + $NN

end


