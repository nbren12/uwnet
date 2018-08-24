#! /bin/csh -f

set NP = 2048
set N = 0

while ($N <= $NP)

echo $N 

rm  /gpfs-wan/scratch/ux454536/SAM6.7/OUT_2D/GATE_IDEAL_*.2Dbin_$N
@ N = $N + 1

end

