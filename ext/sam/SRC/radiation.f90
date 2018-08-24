
subroutine radiation()

!	Radiation interface

use grid
use params, only: dosmoke, doradsimple
implicit none

call t_startf ('radiation')
	
if(doradsimple) then

!  A simple predefined radiation (longwave only)

    if(dosmoke) then
       call rad_simple_smoke()
    else
       call rad_simple()
    end if
	 
else


! Call full radiation package:
 

    call rad_full()	
 
endif

call t_stopf ('radiation')

end


