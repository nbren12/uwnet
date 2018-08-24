!bloss#include <misc.h>
!bloss#include <params.h>
subroutine radinp(lchnk   ,ncol    ,                            &
                  pmid    ,pint    ,o3vmr   , pmidrd  ,&
                  pintrd  ,eccf_in    ,o3mmr   )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Set latitude and time dependent arrays for input to solar
! and longwave radiation.
! Convert model pressures to cgs, and compute ozone mixing ratio, needed for
! the solar radiation.
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: CCM1, CMS Contact J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid
   use shr_orb_mod
!bloss   use time_manager, only: get_curr_calday

   implicit none

!bloss#include <crdcon.h>
!bloss#include <comsol.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                ! chunk identifier
   integer, intent(in) :: ncol                 ! number of atmospheric columns

   real(r4), intent(in) :: pmid(pcols,pver)    ! Pressure at model mid-levels (pascals)
   real(r4), intent(in) :: pint(pcols,pverp)   ! Pressure at model interfaces (pascals)
   real(r4), intent(in) :: o3vmr(pcols,pver)   ! ozone volume mixing ratio
!
! Output arguments
!
   real(r4), intent(out) :: pmidrd(pcols,pver)  ! Pressure at mid-levels (dynes/cm*2)
   real(r4), intent(out) :: pintrd(pcols,pverp) ! Pressure at interfaces (dynes/cm*2)
   real(r4), intent(out) :: eccf_in                ! Earth-sun distance factor
   real(r4), intent(out) :: o3mmr(pcols,pver)   ! Ozone mass mixing ratio

!
!---------------------------Local variables-----------------------------
!
   integer i                ! Longitude loop index
   integer k                ! Vertical loop index

   real(r4) :: calday           ! current calendar day
   real(r4) amd                 ! Effective molecular weight of dry air (g/mol)
   real(r4) amo                 ! Molecular weight of ozone (g/mol)
   real(r4) vmmr                ! Ozone volume mixing ratio
   real(r4) delta               ! Solar declination angle

   save     amd   ,amo

   data amd   /  28.9644   /
   data amo   /  48.0000   /
!
!-----------------------------------------------------------------------
!
!bloss   calday = get_curr_calday()
   calday = day
   call shr_orb_decl (calday  ,eccen     ,mvelpp  ,lambm0  ,obliqr  , &
                      delta   ,eccf_in)

!
! Convert pressure from pascals to dynes/cm2
!
   do k=1,pver
      do i=1,ncol
         pmidrd(i,k) = pmid(i,k)*10.0
         pintrd(i,k) = pint(i,k)*10.0
      end do
   end do
   do i=1,ncol
      pintrd(i,pverp) = pint(i,pverp)*10.0
   end do
!
! Convert ozone volume mixing ratio to mass mixing ratio:
!
   vmmr = amo/amd
   do k=1,pver
      do i=1,ncol
         o3mmr(i,k) = vmmr*o3vmr(i,k)
      end do
   end do
!
   return
end subroutine radinp
