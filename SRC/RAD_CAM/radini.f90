!bloss#include <misc.h>
!bloss#include <params.h>
subroutine radini(gravx   ,cpairx  ,epsilox ,stebolx, pstdx )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Initialize various constants for radiation scheme; note that
! the radiation scheme uses cgs units.
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: W. Collins (H2O parameterization) and J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid
!bloss   use radae,        only: radaeini
!bloss   use comozp,       only: cplos, cplol
!bloss   use pmgrid,       only: masterproc, plev, plevp
!bloss   use physconst,    only: mwdry, mwco2
!bloss#if ( defined SPMD )
!bloss    use mpishorthand
!bloss#endif
   implicit none

!bloss#include <crdcon.h>
!bloss#include <comhyb.h>
!bloss#include <comctl.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   real(r4), intent(in) :: gravx      ! Acceleration of gravity (MKS)
   real(r4), intent(in) :: cpairx     ! Specific heat of dry air (MKS)
   real(r4), intent(in) :: epsilox    ! Ratio of mol. wght of H2O to dry air
   real(r4), intent(in) :: stebolx    ! Stefan-Boltzmann's constant (MKS)
   real(r4), intent(in) :: pstdx      ! Standard pressure (Pascals)
!
!---------------------------Local variables-----------------------------
!
   integer k       ! Loop variable

   real(r4) v0         ! Volume of a gas at stp (m**3/kmol)
   real(r4) p0         ! Standard pressure (pascals)
   real(r4) amd        ! Effective molecular weight of dry air (kg/kmol)
   real(r4) goz        ! Acceleration of gravity (m/s**2)
!
!-----------------------------------------------------------------------
!
! Set general radiation consts; convert to cgs units where appropriate:
!
   gravit  =  100.*gravx
   rga     =  1./gravit
   cpair   =  1.e4*cpairx
   epsilo  =  epsilox
   sslp    =  1.013250e6
   stebol  =  1.e3*stebolx
   rgsslp  =  0.5/(gravit*sslp)
   dpfo3   =  2.5e-3
   dpfco2  =  5.0e-3
   dayspy  =  365.
   pie     =  4.*atan(1.)
!
!
! Initialize ozone data.
!
   v0  = 22.4136         ! Volume of a gas at stp (m**3/kmol)
   p0  = 0.1*sslp        ! Standard pressure (pascals)
   amd = 28.9644         ! Molecular weight of dry air (kg/kmol)
   goz = gravx           ! Acceleration of gravity (m/s**2)
!
! Constants for ozone path integrals (multiplication by 100 for unit
! conversion to cgs from mks):
!
   cplos = v0/(amd*goz)       *100.0
   cplol = v0/(amd*goz*p0)*0.5*100.0
!
! Derived constants
   ntoplw = 1 ! hard-wire ntoplw to 1 since zi(nz) is unlikely to be above 90km

!bloss! If the top model level is above ~90 km (0.1 Pa), set the top level to compute
!bloss! longwave cooling to about 80 km (1 Pa)
!bloss   if (hypm(1) .lt. 0.1) then
!bloss      do k = 1, pver
!bloss         if (hypm(k) .lt. 1.) ntoplw  = k
!bloss      end do
!bloss   else
!bloss      ntoplw = 1
!bloss   end if
!bloss   if (masterproc) then
!bloss      write (6,*) 'RADINI: ntoplw =',ntoplw, ' pressure:',hypm(ntoplw)
!bloss   endif

!bloss   call radaeini( pstdx, mwdry, mwco2 )
   return
end subroutine radini
