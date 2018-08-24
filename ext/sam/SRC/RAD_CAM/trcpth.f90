!bloss#include <misc.h>
!bloss#include <params.h>
subroutine trcpth(lchnk   ,ncol    ,                            &
                  tnm     ,pnm     ,cfc11   ,cfc12   ,n2o     , &
                  ch4     ,qnm     ,ucfc11  ,ucfc12  ,un2o0   , &
                  un2o1   ,uch4    ,uco211  ,uco212  ,uco213  , &
                  uco221  ,uco222  ,uco223  ,bn2o0   ,bn2o1   , &
                  bch4    ,uptype  )
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Calculate path lengths and pressure factors for CH4, N2O, CFC11
! and CFC12.
! 
! Method: 
! See CCM3 description for details
! 
! Author: J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid
!bloss   use ghg_surfvals, only: ghg_surfvals_get_co2mmr

   implicit none

!bloss#include <crdcon.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: ncol                  ! number of atmospheric columns

   real(r4), intent(in) :: tnm(pcols,pver)      ! Model level temperatures
   real(r4), intent(in) :: pnm(pcols,pverp)     ! Pres. at model interfaces (dynes/cm2)
   real(r4), intent(in) :: qnm(pcols,pver)      ! h2o specific humidity
   real(r4), intent(in) :: cfc11(pcols,pver)    ! CFC11 mass mixing ratio
!
   real(r4), intent(in) :: cfc12(pcols,pver)    ! CFC12 mass mixing ratio
   real(r4), intent(in) :: n2o(pcols,pver)      ! N2O mass mixing ratio
   real(r4), intent(in) :: ch4(pcols,pver)      ! CH4 mass mixing ratio

!
! Output arguments
!
   real(r4), intent(out) :: ucfc11(pcols,pverp)  ! CFC11 path length
   real(r4), intent(out) :: ucfc12(pcols,pverp)  ! CFC12 path length
   real(r4), intent(out) :: un2o0(pcols,pverp)   ! N2O path length
   real(r4), intent(out) :: un2o1(pcols,pverp)   ! N2O path length (hot band)
   real(r4), intent(out) :: uch4(pcols,pverp)    ! CH4 path length
!
   real(r4), intent(out) :: uco211(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(out) :: uco212(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(out) :: uco213(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(out) :: uco221(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(out) :: uco222(pcols,pverp)  ! CO2 10.4 micron band path length
!
   real(r4), intent(out) :: uco223(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(out) :: bn2o0(pcols,pverp)   ! pressure factor for n2o
   real(r4), intent(out) :: bn2o1(pcols,pverp)   ! pressure factor for n2o
   real(r4), intent(out) :: bch4(pcols,pverp)    ! pressure factor for ch4
   real(r4), intent(out) :: uptype(pcols,pverp)  ! p-type continuum path length

!
!---------------------------Local variables-----------------------------
!
   integer   i               ! Longitude index
   integer   k               ! Level index
!
   real(r4) co2fac(pcols,1)      ! co2 factor
   real(r4) alpha1(pcols)        ! stimulated emission term
   real(r4) alpha2(pcols)        ! stimulated emission term
   real(r4) rt(pcols)            ! reciprocal of local temperature
   real(r4) rsqrt(pcols)         ! reciprocal of sqrt of temp
!
   real(r4) pbar(pcols)          ! mean pressure
   real(r4) dpnm(pcols)          ! difference in pressure
   real(r4) diff                 ! diffusivity factor
!bloss   real(r4) co2mmr               ! co2 mass mixing ratio
!
!--------------------------Data Statements------------------------------
!
   data diff /1.66/
!
!-----------------------------------------------------------------------
!
!  Calculate path lengths for the trace gases at model top
!

!bloss   co2mmr = ghg_surfvals_get_co2mmr()    ! co2 mass mixing ratio 

   do i = 1,ncol
      ucfc11(i,ntoplw) = 1.8 * cfc11(i,ntoplw) * pnm(i,ntoplw) * rga
      ucfc12(i,ntoplw) = 1.8 * cfc12(i,ntoplw) * pnm(i,ntoplw) * rga
      un2o0(i,ntoplw) = diff * 1.02346e5 * n2o(i,ntoplw) * pnm(i,ntoplw) * rga / sqrt(tnm(i,ntoplw))
      un2o1(i,ntoplw) = diff * 2.01909 * un2o0(i,ntoplw) * exp(-847.36/tnm(i,ntoplw))
      uch4(i,ntoplw)  = diff * 8.60957e4 * ch4(i,ntoplw) * pnm(i,ntoplw) * rga / sqrt(tnm(i,ntoplw))
      co2fac(i,1)     = diff * co2mmr * pnm(i,ntoplw) * rga
      alpha1(i) = (1.0 - exp(-1540.0/tnm(i,ntoplw)))**3.0/sqrt(tnm(i,ntoplw))
      alpha2(i) = (1.0 - exp(-1360.0/tnm(i,ntoplw)))**3.0/sqrt(tnm(i,ntoplw))
      uco211(i,ntoplw) = 3.42217e3 * co2fac(i,1) * alpha1(i) * exp(-1849.7/tnm(i,ntoplw))
      uco212(i,ntoplw) = 6.02454e3 * co2fac(i,1) * alpha1(i) * exp(-2782.1/tnm(i,ntoplw))
      uco213(i,ntoplw) = 5.53143e3 * co2fac(i,1) * alpha1(i) * exp(-3723.2/tnm(i,ntoplw))
      uco221(i,ntoplw) = 3.88984e3 * co2fac(i,1) * alpha2(i) * exp(-1997.6/tnm(i,ntoplw))
      uco222(i,ntoplw) = 3.67108e3 * co2fac(i,1) * alpha2(i) * exp(-3843.8/tnm(i,ntoplw))
      uco223(i,ntoplw) = 6.50642e3 * co2fac(i,1) * alpha2(i) * exp(-2989.7/tnm(i,ntoplw))
      bn2o0(i,ntoplw) = diff * 19.399 * pnm(i,ntoplw)**2.0 * n2o(i,ntoplw) * &
                   1.02346e5 * rga / (sslp*tnm(i,ntoplw))
      bn2o1(i,ntoplw) = bn2o0(i,ntoplw) * exp(-847.36/tnm(i,ntoplw)) * 2.06646e5
      bch4(i,ntoplw) = diff * 2.94449 * ch4(i,ntoplw) * pnm(i,ntoplw)**2.0 * rga * &
                  8.60957e4 / (sslp*tnm(i,ntoplw))
      uptype(i,ntoplw) = diff * qnm(i,ntoplw) * pnm(i,ntoplw)**2.0 *  &
                    exp(1800.0*(1.0/tnm(i,ntoplw) - 1.0/296.0)) * rga / sslp
   end do
!
! Calculate trace gas path lengths through model atmosphere
!
   do k = ntoplw,pver
      do i = 1,ncol
         rt(i) = 1./tnm(i,k)
         rsqrt(i) = sqrt(rt(i))
         pbar(i) = 0.5 * (pnm(i,k+1) + pnm(i,k)) / sslp
         dpnm(i) = (pnm(i,k+1) - pnm(i,k)) * rga
         alpha1(i) = diff * rsqrt(i) * (1.0 - exp(-1540.0/tnm(i,k)))**3.0
         alpha2(i) = diff * rsqrt(i) * (1.0 - exp(-1360.0/tnm(i,k)))**3.0
         ucfc11(i,k+1) = ucfc11(i,k) +  1.8 * cfc11(i,k) * dpnm(i)
         ucfc12(i,k+1) = ucfc12(i,k) +  1.8 * cfc12(i,k) * dpnm(i)
         un2o0(i,k+1) = un2o0(i,k) + diff * 1.02346e5 * n2o(i,k) * rsqrt(i) * dpnm(i)
         un2o1(i,k+1) = un2o1(i,k) + diff * 2.06646e5 * n2o(i,k) * &
                        rsqrt(i) * exp(-847.36/tnm(i,k)) * dpnm(i)
         uch4(i,k+1) = uch4(i,k) + diff * 8.60957e4 * ch4(i,k) * rsqrt(i) * dpnm(i)
         uco211(i,k+1) = uco211(i,k) + 1.15*3.42217e3 * alpha1(i) * &
                         co2mmr * exp(-1849.7/tnm(i,k)) * dpnm(i)
         uco212(i,k+1) = uco212(i,k) + 1.15*6.02454e3 * alpha1(i) * &
                         co2mmr * exp(-2782.1/tnm(i,k)) * dpnm(i)
         uco213(i,k+1) = uco213(i,k) + 1.15*5.53143e3 * alpha1(i) * &
                         co2mmr * exp(-3723.2/tnm(i,k)) * dpnm(i)
         uco221(i,k+1) = uco221(i,k) + 1.15*3.88984e3 * alpha2(i) * &
                         co2mmr * exp(-1997.6/tnm(i,k)) * dpnm(i)
         uco222(i,k+1) = uco222(i,k) + 1.15*3.67108e3 * alpha2(i) * &
                         co2mmr * exp(-3843.8/tnm(i,k)) * dpnm(i)
         uco223(i,k+1) = uco223(i,k) + 1.15*6.50642e3 * alpha2(i) * &
                         co2mmr * exp(-2989.7/tnm(i,k)) * dpnm(i)
         bn2o0(i,k+1) = bn2o0(i,k) + diff * 19.399 * pbar(i) * rt(i) &
                        * 1.02346e5 * n2o(i,k) * dpnm(i)
         bn2o1(i,k+1) = bn2o1(i,k) + diff * 19.399 * pbar(i) * rt(i) &
                        * 2.06646e5 * exp(-847.36/tnm(i,k)) * n2o(i,k)*dpnm(i)
         bch4(i,k+1) = bch4(i,k) + diff * 2.94449 * rt(i) * pbar(i) &
                       * 8.60957e4 * ch4(i,k) * dpnm(i)
         uptype(i,k+1) = uptype(i,k) + diff *qnm(i,k) * &
                         exp(1800.0*(1.0/tnm(i,k) - 1.0/296.0)) * pbar(i) * dpnm(i)
      end do
   end do
!
   return
end subroutine trcpth

