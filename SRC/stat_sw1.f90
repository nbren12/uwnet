! scalar flux budget: 	

	
subroutine stat_sw1(du,twleproc,qwleproc)

use vars

implicit none
real du(nx,ny,nz,3)
real twleproc(nzm), qwleproc(nzm), swleproc(nzm)
integer i,j,k	
do j=1,ny
 do i=1,nx
  du(i,j,nz,3)=0.
 end do
end do
do k=1,nzm
 do j=1,ny
  do i=1,nx
   twleproc(k)=twleproc(k)+0.5*(t(i,j,k)-t0(k))*(du(i,j,k,3)+du(i,j,k+1,3))
   qwleproc(k)=qwleproc(k)+0.5*(qv(i,j,k)+qcl(i,j,k)+qci(i,j,k)-q0(k))*(du(i,j,k,3)+du(i,j,k+1,3))
  end do
 end do
end do

end
