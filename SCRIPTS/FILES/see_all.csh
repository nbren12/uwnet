#! /bin/csh -f

set filename = /gpfs/scratch1/marat/SAM6.7_SR/OUT_3D/GATE_IDEAL_S_2048x2048x256_100m_2s_2048_00000
set N1 = 0
set N2 = 43200
set NN = 150

while ($N1 < $N2)

   @ N1 = $N1 + $NN
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

   ls -l $filename$M$N1.com3D

end

