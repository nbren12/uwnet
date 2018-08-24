!bloss#include <misc.h>
!bloss#include <params.h>
subroutine radoz2(lchnk   ,ncol    ,o3vmr   ,pint    ,plol    ,plos, ntoplw    )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Computes the path length integrals to the model interfaces given the
! ozone volume mixing ratio
! 
! Method: 
! <Describe the algorithm(s) used in the routine.> 
! <Also include any applicable external references.> 
! 
! Author: CCM1, CMS Contact J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid, only: cplos, cplol, pcols, pver, pverp
!bloss   use comozp

   implicit none
!------------------------------Input arguments--------------------------
!
   integer, intent(in) :: lchnk                ! chunk identifier
   integer, intent(in) :: ncol                 ! number of atmospheric columns

   real(r4), intent(in) :: o3vmr(pcols,pver)   ! ozone volume mixing ratio
   real(r4), intent(in) :: pint(pcols,pverp)   ! Model interface pressures

   integer, intent(in) :: ntoplw               ! topmost level/layer longwave is solved for

!
!----------------------------Output arguments---------------------------
!
   real(r4), intent(out) :: plol(pcols,pverp)   ! Ozone prs weighted path length (cm)
   real(r4), intent(out) :: plos(pcols,pverp)   ! Ozone path length (cm)

!
!---------------------------Local workspace-----------------------------
!
   integer i                ! longitude index
   integer k                ! level index
!
!-----------------------------------------------------------------------
!
! Evaluate the ozone path length integrals to interfaces;
! factors of .1 and .01 to convert pressures from cgs to mks:
!
   do i=1,ncol
      plos(i,ntoplw) = 0.1 *cplos*o3vmr(i,ntoplw)*pint(i,ntoplw)
      plol(i,ntoplw) = 0.01*cplol*o3vmr(i,ntoplw)*pint(i,ntoplw)*pint(i,ntoplw)
   end do
   do k=ntoplw+1,pverp
      do i=1,ncol
         plos(i,k) = plos(i,k-1) + 0.1*cplos*o3vmr(i,k-1)*(pint(i,k) - pint(i,k-1))
         plol(i,k) = plol(i,k-1) + 0.01*cplol*o3vmr(i,k-1)* &
                    (pint(i,k)*pint(i,k) - pint(i,k-1)*pint(i,k-1))
      end do
   end do
!
   return
end subroutine radoz2

