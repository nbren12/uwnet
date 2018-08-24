!bloss#include <misc.h>
!bloss#include <params.h>
subroutine zenith(calday  ,clat    , clon   ,coszrs  ,ncol    )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Compute cosine of solar zenith angle for albedo and radiation
!   computations.
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use shr_orb_mod
!bloss
   use ppgrid, only: eccen, mvelpp, lambm0, obliqr, eccf

   implicit none

!bloss#include <crdcon.h>
!bloss#include <comsol.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: ncol                 ! number of positions
   real(r4), intent(in) :: calday              ! Calendar day, including fraction
   real(r4), intent(in) :: clat(ncol)          ! Current centered latitude (radians)
   real(r4), intent(in) :: clon(ncol)          ! Centered longitude (radians)
!
! Output arguments
!
   real(r4), intent(out) :: coszrs(ncol)       ! Cosine solar zenith angle
!
!---------------------------Local variables-----------------------------
!
   integer i         ! Position loop index
   real(r4) delta    ! Solar declination angle  in radians
!bloss   real(r4) eccf     ! Earth orbit eccentricity factor
!
!-----------------------------------------------------------------------
!
   call shr_orb_decl (calday  ,eccen     ,mvelpp  ,lambm0  ,obliqr  , &
                      delta   ,eccf      )
!
! Compute local cosine solar zenith angle,
!
   do i=1,ncol
      coszrs(i) = shr_orb_cosz( calday, clat(i), clon(i), delta )
   end do
!
   return
end subroutine zenith

