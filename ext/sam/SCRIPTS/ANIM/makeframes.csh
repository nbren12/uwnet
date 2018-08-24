#!/bin/csh

set echo

set N = 0
set NMAX = 719

while ($N <= $NMAX)

echo $N > number

if($N < 10) then
  set M = "000"
else if($N < 100) then
  set M = "00"
else if($N < 1000) then
  set M = "0"
else
  set M = ""
endif


ncl  NCL/plot_frame_xz.ncl
mv  gsnapp.ncgm  movie/FRAMES/image$M$N.ncgm

@ N = $N + 1

end
