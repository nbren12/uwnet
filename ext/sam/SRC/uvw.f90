subroutine uvw
	
! update the velocity field 

use vars
use params
implicit none
	
call t_startf ('adams_uvw')

u(1:nx,1:ny,1:nzm) = dudt(1:nx,1:ny,1:nzm,nc)
v(1:nx,1:ny,1:nzm) = dvdt(1:nx,1:ny,1:nzm,nc)
w(1:nx,1:ny,1:nzm) = dwdt(1:nx,1:ny,1:nzm,nc)

call t_stopf ('adams_uvw')

end subroutine uvw
