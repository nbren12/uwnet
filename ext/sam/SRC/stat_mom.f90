! Momentum flux budget stuff:

subroutine stat_mom(du,momle)

use vars
implicit none
real du(nx,ny,nz,3)
real momle(nz,3),coef
integer i,j,k,jb,jc	
coef = 1./float(nx*ny)	
do k=1,nz
 momle(k,1)=0.
 momle(k,2)=0.
 momle(k,3)=0.
end do
do j=1,ny
 do i=1,nx
   du(i,j,nz,3) = 0.
 end do
end do
do k=1,nzm
 do j=1,ny
  jb=j-YES3D
  jc=j+YES3D
  do i=1,nx
   momle(k,1)=momle(k,1)+ &
          0.25*(w(i,j,k)+w(i,j,k+1)+w(i-1,j,k)+w(i-1,j,k+1))*du(i,j,k,1)+ &
          0.25*(u(i,j,k)+u(i+1,j,k)-2.*u0(k))*(du(i,j,k,3)+du(i,j,k+1,3))
   momle(k,2)=momle(k,2)+ & 
          0.25*(w(i,j,k)+w(i,j,k+1)+w(i,jb,k)+w(i,jb,k+1))*du(i,j,k,2)+ &
          0.25*(v(i,j,k)+v(i,jc,k)-2.*v0(k))*(du(i,j,k,3)+du(i,j,k+1,3))
   momle(k,3)=momle(k,3)+w(i,j,k)*du(i,j,k,3)
  end do
 end do
 momle(k,1)=momle(k,1)*coef
 momle(k,2)=momle(k,2)*coef
 momle(k,3)=momle(k,3)*coef
end do
do k=1,nzm
  momle(k,3)=0.5*(momle(k,3)+momle(k+1,3))
end do
end




