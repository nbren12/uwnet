  
subroutine precip_init

! Initialize precipitation related stuff

use vars
use microphysics
use micro_params
use params

implicit none

real coef1, coef2,estw,rrr1,rrr2
integer k

do k=1,nzm

  rrr1=393./(tabs0(k)+120.)*(tabs0(k)/273.)**1.5
  rrr2=(tabs0(k)/273.)**1.94*(1000./pres(k))
  estw = 100.*esatw(tabs0(k))

! evaporation of rain:

  coef1  =(lcond/(tabs0(k)*rv)-1.)*lcond/(therco*rrr1*tabs0(k))
  coef2  = rv*tabs0(k)/(diffelq * rrr2 * estw)

  evapr1(k) = 3.*0.86 / (coef1+coef2) / rhor /coefrv**0.6666  ! (KK 2000)
  evapr2(k) = 0.

end do

           
end subroutine precip_init


