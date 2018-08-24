
subroutine rad_simple_smoke
 	
!	Simple Interactive Radiation for Smoke-cloud GCSS WG1 case

use vars
use params
use rad, only: qrad
implicit none
	
real lwp(nx,ny), fr(nx,ny), f0, xk, fr1, dtrad, coef, coef1
integer i,j,k
	
if(.not.dolongwave) return

f0 = 60 ! W/m2
xk = 20. 
	
fr(:,:) = f0
lwp(:,:) = 0.
radlwdn(:) =0.
radqrlw(:) =0.

	
radlwdn(nz) = f0 * nx*ny
	
do k = nzm,1,-1
  coef1 = rho(k)*dz*adz(k)
  coef = 1./(coef1 * cp)
  do j=1,ny
  do i=1,nx
    lwp(i,j) = lwp(i,j) + qv(i,j,k)*coef1
    fr1 = f0 * exp(-xk * lwp(i,j))
    dtrad = -(fr(i,j) - fr1) * coef
    fr(i,j) = fr1
    t(i,j,k) = t(i,j,k) + dtrad * dtn
    radlwdn(k) = radlwdn(k) + fr1
    radqrlw(k) = radqrlw(k) + dtrad
    qrad(i,j,k) = dtrad
  end do
  end do
end do
	
end 


