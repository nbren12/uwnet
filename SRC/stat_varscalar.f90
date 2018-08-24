! scalar variance budget:

subroutine stat_varscalar(f,df,f0,df0,s2le)

use grid
implicit none
real f(dimx1_s:dimx2_s,dimy1_s:dimy2_s,nzm)
real df(dimx1_s:dimx2_s,dimy1_s:dimy2_s,nzm)
real s2le(nzm)
real f0(nzm),df0(nzm),coef
integer i,j,k	
call averageXY_MPI(f,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,f0)
call averageXY_MPI(df,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,df0)
coef = 1./(dtn*nx*ny)	
do k=1,nzm
 s2le(k)=0.
 do j=1,ny
  do i=1,nx
   s2le(k)=s2le(k)+(f(i,j,k)-f0(k))**2-(df(i,j,k)-df0(k))**2
  end do
 end do
 s2le(k)=s2le(k)*coef
end do

end
	

