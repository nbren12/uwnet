! CKE budget stuff


subroutine stat_tke(du,tkele)

use vars
implicit none
real du(nx,ny,nz,3)
real tkele(nzm)
real d_u(nz), d_v(nz),d_w(nz),coef
integer i,j,k	
coef = 1./float(nx*ny)
do k=1,nz
 d_u(k)=0.
 d_v(k)=0.
 d_w(k)=0.
end do
do k=1,nzm
 do j=1,ny
  do i=1,nx
   d_u(k)=d_u(k)+(u(i,j,k)-u0(k))*du(i,j,k,1)
   d_v(k)=d_v(k)+(v(i,j,k)-v0(k))*du(i,j,k,2)
   d_w(k)=d_w(k)+ w(i,j,k) *      du(i,j,k,3)
  end do
 end do
 d_u(k)=d_u(k)*coef
 d_v(k)=d_v(k)*coef
 d_w(k)=d_w(k)*coef
end do
do k=1,nzm
 tkele(k)=0.5*(d_w(k)+d_w(k+1))+d_u(k)+d_v(k)*YES3D
end do

end




