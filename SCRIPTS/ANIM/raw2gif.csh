#!/bin/csh

 set echo
# set the caseid for the movies to process
set thepath = 'movie/'                   # caseid of run
set caseid = 'GATE_IDEAL_512x512x128_500m_6s_'
# set the total grid dimensions
set nx_gl = 512                            # TOTAL points in x direction
set ny_gl = 512                            # TOTAL points in y direction 

#####################################################
# END USER INPUT

#pgf90 convertmov.f90
#./a.out
#cd movie

foreach field (vsfc cldtop usfc thsfc qvsfc sfcprec cwp iwp mse)

#convert -depth 8 -size $nx_gl'x'$ny_gl $thepath$caseid$field'.raw' $thepath$caseid$field'.gif'
mv $thepath$caseid$field'_$.raw' $thepath$caseid$field'_S.raw'
#rm mv$thepath$caseid$field'.raw'

end
