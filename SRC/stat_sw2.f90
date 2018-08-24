! scalar flux budget: Part II

subroutine stat_sw2(f,df,swle)

use vars
implicit none
real f(dimx1_s:dimx2_s,dimy1_s:dimy2_s,nzm)
real df(dimx1_s:dimx2_s,dimy1_s:dimy2_s,nzm)
real swle(nzm), coef
integer i,j,k	
coef=0.5/dtn
do j=1,ny
 do i=1,nx
  misc(i,j,nz) = 0.
 end do
end do
do k=1,nzm
 do j=1,ny
  do i=1,nx
   swle(k)=swle(k)+coef*(misc(i,j,k)+misc(i,j,k+1))*(f(i,j,k)-df(i,j,k))
  end do
 end do
end do

end




