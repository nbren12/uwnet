#! /bin/csh -f

#set filename = /gpfs-wan/scratch/ux454536/SAM6.6.5d/DATA3D/BOMEX_1024x1024x192_20m_20m_0.5s_1024_00000
#set N1 = 36120
#set N2 = 43200
#set NN = 120
#set filename = /gpfs-wan/scratch/ux454536/SAM6.7/OUT_3D/GATE_IDEAL_512x512x128_500m_6s_512_00000
#set N1 = 10050
#set N2 = 21600
#set NN = 150
set filename = /gpfs-wan/scratch/ux454536/SAM6.7/OUT_MOMENTS/GATE_IDEAL_moments_512x512x128_500m_6s_512_00000
set N1 = 10000
set N2 = 21600
set NN = 100

while ($N1 <= $N2)

echo $N1
set f = $filename{$N1}.bin3D
@ N1 = $N1 + $NN

../bin3D_sep2one  $f

end

