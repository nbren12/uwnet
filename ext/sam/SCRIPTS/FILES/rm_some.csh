#! /bin/csh -f

set filename = GATE_IDEAL_S_2048x2048x256_100m_2s_2048_00000
set N1 = 150
set N2 = 43200
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

set f = $filename$M$N1.2Dcom_

ls  $f*

@ N1 = $N1 + $NN

end


