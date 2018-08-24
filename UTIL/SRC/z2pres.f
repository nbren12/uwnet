c
c  pressure levels from heights, temperature and mosture.
c

	subroutine z2pres(pres0, z, tabs, q, p)

	implicit none
	include 'grid.inc'

! input:
	real pres0	! surface pressure
	real z(nzm)	! height
	real tabs(nzm)	! absolute temperature, K
	real q(nzm)	! vapor mixing ratio, kg/kg
!output
	real p(nzm)	! pressure levels, mB
!local
	real pi(0:nzm), tv(nzm), coef, tmp
	integer k

	coef = 2.*9.81/1004.

	do k=1,nzm
	 tv(k)=tabs(k)*(1.+0.61*q(k))
	end do

	pi(0)=(pres0/1000.)**0.28586
	pi(1)=pi(0)*tv(1)/(tv(1)+coef/2.*z(1))
	p(1) = 1000.*pi(1)**3.498278 
	do k=2,nzm
	 tmp = (tv(k)-tv(k-1)+coef*(z(k)-z(k-1)))*pi(k-1)
	 pi(k)= (-tmp+sqrt(tmp**2+4.*tv(k)*tv(k-1)*pi(k-1)**2))
     &	                                            /(2.*tv(k-1))
	 p(k) = 1000.*pi(k)**3.498278
	end do

	return
	end
