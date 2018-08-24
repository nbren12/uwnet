subroutine diffuse_mom

!  Interface to the diffusion routines

use vars
implicit none
integer i,j,k
real du(nx,ny,nz,3)

call t_startf ('diffuse_mom')

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

if(RUN3D) then
   call diffuse_mom3D()
else
   call diffuse_mom2D()
endif

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

  call stat_tke(du,tkelediff)
  call stat_mom(du,momlediff)
  call setvalue(twlediff,nzm,0.)
  call setvalue(qwlediff,nzm,0.)
  call stat_sw1(du,twlediff,qwlediff)

endif

call t_stopf ('diffuse_mom')

end subroutine diffuse_mom

