subroutine advect_mom

use vars
use params, only: docolumn

implicit none
integer i,j,k
real du(nx,ny,nz,3)

if(docolumn) return

call t_startf ('advect_mom')

if(dostatis) then
	
  do k=1,nzm
   do j=1,ny
    do i=1,nx
     du(i,j,k,1)=dudt(i,j,k,na) 
     du(i,j,k,2)=dvdt(i,j,k,na) 
     du(i,j,k,3)=dwdt(i,j,k,na) 
    end do
   end do
  end do

endif

call advect2_mom_xy()
call advect2_mom_z()


if(dostatis) then
	
  do k=1,nzm
   do j=1,ny
    do i=1,nx
     du(i,j,k,1)=dudt(i,j,k,na)-du(i,j,k,1)
     du(i,j,k,2)=dvdt(i,j,k,na)-du(i,j,k,2)
     du(i,j,k,3)=dwdt(i,j,k,na)-du(i,j,k,3)
    end do
   end do
  end do

  call stat_tke(du,tkeleadv)
  call stat_mom(du,momleadv)
  call setvalue(twleadv,nzm,0.)
  call setvalue(qwleadv,nzm,0.)
  call stat_sw1(du,twleadv,qwleadv)

endif

call t_stopf ('advect_mom')

end subroutine advect_mom

