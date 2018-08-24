!bloss#include <misc.h>
!bloss#include <params.h>
subroutine trcems(lchnk   ,ncol    ,                            &
                  k       ,co2t    ,pnm     ,ucfc11  ,ucfc12  , &
                  un2o0   ,un2o1   ,bn2o0   ,bn2o1   ,uch4    , &
                  bch4    ,uco211  ,uco212  ,uco213  ,uco221  , &
                  uco222  ,uco223  ,uptype  ,w       ,s2c     , &
                  up2     ,emplnk  ,th2o    ,tco2    ,to3     , &
                  emstrc  , &
                 aer_trn_ttl)
!----------------------------------------------------------------------- 
! 
! Purpose: 
!  Calculate emissivity for CH4, N2O, CFC11 and CFC12 bands.
! 
! Method: 
!  See CCM3 Description for equations.
! 
! Author: J. Kiehl
! 
!-----------------------------------------------------------------------
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid
   use volcrad

   implicit none

!bloss#include <crdcon.h>
!
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: ncol                  ! number of atmospheric columns

   real(r4), intent(in) :: co2t(pcols,pverp)    ! pressure weighted temperature
   real(r4), intent(in) :: pnm(pcols,pverp)     ! interface pressure
   real(r4), intent(in) :: ucfc11(pcols,pverp)  ! CFC11 path length
   real(r4), intent(in) :: ucfc12(pcols,pverp)  ! CFC12 path length
   real(r4), intent(in) :: un2o0(pcols,pverp)   ! N2O path length
!
   real(r4), intent(in) :: un2o1(pcols,pverp)   ! N2O path length (hot band)
   real(r4), intent(in) :: uch4(pcols,pverp)    ! CH4 path length
   real(r4), intent(in) :: uco211(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco212(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco213(pcols,pverp)  ! CO2 9.4 micron band path length
!
   real(r4), intent(in) :: uco221(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco222(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco223(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uptype(pcols,pverp)  ! continuum path length
   real(r4), intent(in) :: bn2o0(pcols,pverp)   ! pressure factor for n2o
!
   real(r4), intent(in) :: bn2o1(pcols,pverp)   ! pressure factor for n2o
   real(r4), intent(in) :: bch4(pcols,pverp)    ! pressure factor for ch4
   real(r4), intent(in) :: emplnk(14,pcols)     ! emissivity Planck factor
   real(r4), intent(in) :: th2o(pcols)          ! water vapor overlap factor
   real(r4), intent(in) :: tco2(pcols)          ! co2 overlap factor
!
   real(r4), intent(in) :: to3(pcols)           ! o3 overlap factor
   real(r4), intent(in) :: s2c(pcols,pverp)     ! h2o continuum path length
   real(r4), intent(in) :: w(pcols,pverp)       ! h2o path length
   real(r4), intent(in) :: up2(pcols)           ! pressure squared h2o path length
!
   integer, intent(in) :: k                 ! level index

   real(r4), intent(in) :: aer_trn_ttl(pcols,pverp,pverp,bnd_nbr_LW) ! aer trn.

!
!  Output Arguments
!
   real(r4), intent(out) :: emstrc(pcols,pverp)  ! total trace gas emissivity

!
!--------------------------Local Variables------------------------------
!
   integer i,l               ! loop counters
!
   real(r4) sqti(pcols)          ! square root of mean temp
   real(r4) ecfc1                ! emissivity of cfc11 798 cm-1 band
   real(r4) ecfc2                !     "      "    "   846 cm-1 band
   real(r4) ecfc3                !     "      "    "   933 cm-1 band
   real(r4) ecfc4                !     "      "    "   1085 cm-1 band
!
   real(r4) ecfc5                !     "      "  cfc12 889 cm-1 band
   real(r4) ecfc6                !     "      "    "   923 cm-1 band
   real(r4) ecfc7                !     "      "    "   1102 cm-1 band
   real(r4) ecfc8                !     "      "    "   1161 cm-1 band
   real(r4) u01                  ! n2o path length
!
   real(r4) u11                  ! n2o path length
   real(r4) beta01               ! n2o pressure factor
   real(r4) beta11               ! n2o pressure factor
   real(r4) en2o1                ! emissivity of the 1285 cm-1 N2O band
   real(r4) u02                  ! n2o path length
!
   real(r4) u12                  ! n2o path length
   real(r4) beta02               ! n2o pressure factor
   real(r4) en2o2                ! emissivity of the 589 cm-1 N2O band
   real(r4) u03                  ! n2o path length
   real(r4) beta03               ! n2o pressure factor
!
   real(r4) en2o3                ! emissivity of the 1168 cm-1 N2O band
   real(r4) betac                ! ch4 pressure factor
   real(r4) ech4                 ! emissivity of 1306 cm-1 CH4 band
   real(r4) betac1               ! co2 pressure factor
   real(r4) betac2               ! co2 pressure factor
!
   real(r4) eco21                ! emissivity of 1064 cm-1 CO2 band
   real(r4) eco22                ! emissivity of 961 cm-1 CO2 band
   real(r4) tt(pcols)            ! temp. factor for h2o overlap factor
   real(r4) psi1                 ! narrow band h2o temp. factor
   real(r4) phi1                 !             "
!
   real(r4) p1                   ! h2o line overlap factor
   real(r4) w1                   !          "
   real(r4) tw(pcols,6)          ! h2o transmission overlap
   real(r4) g1(6)                ! h2o overlap factor
   real(r4) g2(6)                !          "
!
   real(r4) g3(6)                !          "
   real(r4) g4(6)                !          "
   real(r4) ab(6)                !          "
   real(r4) bb(6)                !          "
   real(r4) abp(6)               !          "
!
   real(r4) bbp(6)               !          "
   real(r4) tcfc3                ! transmission for cfc11 band
   real(r4) tcfc4                !          "
   real(r4) tcfc6                ! transmission for cfc12 band
   real(r4) tcfc7                !          "
!
   real(r4) tcfc8                !          "
   real(r4) tlw                  ! h2o overlap factor
   real(r4) tch4                 ! ch4 overlap factor
!
!--------------------------Data Statements------------------------------
!
   data g1 /0.0468556,0.0397454,0.0407664,0.0304380,0.0540398,0.0321962/
   data g2 /14.4832,4.30242,5.23523,3.25342,0.698935,16.5599/
   data g3 /26.1898,18.4476,15.3633,12.1927,9.14992,8.07092/
   data g4 /0.0261782,0.0369516,0.0307266,0.0243854,0.0182932,0.0161418/
   data ab /3.0857e-2,2.3524e-2,1.7310e-2,2.6661e-2,2.8074e-2,2.2915e-2/
   data bb /-1.3512e-4,-6.8320e-5,-3.2609e-5,-1.0228e-5,-9.5743e-5,-1.0304e-4/
   data abp/2.9129e-2,2.4101e-2,1.9821e-2,2.6904e-2,2.9458e-2,1.9892e-2/
   data bbp/-1.3139e-4,-5.5688e-5,-4.6380e-5,-8.0362e-5,-1.0115e-4,-8.8061e-5/
!
!--------------------------Statement Functions--------------------------
!
   real(r4) func, u, b
   func(u,b) = u/sqrt(4.0 + u*(1.0 + 1.0 / b))
!
!-----------------------------------------------------------------------
!
   do i = 1,ncol
      sqti(i) = sqrt(co2t(i,k))
!
! Transmission for h2o
!
      tt(i) = abs(co2t(i,k) - 250.0)
   end do
!
   do l = 1,6
      do i = 1,ncol
         psi1 = exp(abp(l)*tt(i)+bbp(l)*tt(i)*tt(i))
         phi1 = exp(ab(l)*tt(i)+bb(l)*tt(i)*tt(i))
         p1 = pnm(i,k) * (psi1/phi1) / sslp
         w1 = w(i,k) * phi1
         tw(i,l) = exp(- g1(l)*p1*(sqrt(1.0+g2(l)*(w1/p1))-1.0) &
                   - g3(l)*s2c(i,k)-g4(l)*uptype(i,k))
      end do
   end do

!     Overlap H2O tranmission with STRAER continuum in 6 trace gas 
!                 subbands

      do i=1,ncol
         tw(i,1)=tw(i,1)*(0.7*aer_trn_ttl(i,k,1,idx_LW_0650_0800)+&! l=1: 0750--0820 cm-1
                          0.3*aer_trn_ttl(i,k,1,idx_LW_0800_1000))
         tw(i,2)=tw(i,2)*aer_trn_ttl(i,k,1,idx_LW_0800_1000) ! l=2: 0820--0880 cm-1
         tw(i,3)=tw(i,3)*aer_trn_ttl(i,k,1,idx_LW_0800_1000) ! l=3: 0880--0900 cm-1
         tw(i,4)=tw(i,4)*aer_trn_ttl(i,k,1,idx_LW_0800_1000) ! l=4: 0900--1000 cm-1
         tw(i,5)=tw(i,5)*aer_trn_ttl(i,k,1,idx_LW_1000_1200) ! l=5: 1000--1120 cm-1
         tw(i,6)=tw(i,6)*aer_trn_ttl(i,k,1,idx_LW_1000_1200) ! l=6: 1120--1170 cm-1
      end do                    ! end loop over lon
!
   do i = 1,ncol
!
! transmission due to cfc bands
!
      tcfc3 = exp(-175.005*ucfc11(i,k))
      tcfc4 = exp(-1202.18*ucfc11(i,k))
      tcfc6 = exp(-5786.73*ucfc12(i,k))
      tcfc7 = exp(-2873.51*ucfc12(i,k))
      tcfc8 = exp(-2085.59*ucfc12(i,k))
!
! Emissivity for CFC11 bands
!
      ecfc1 = 50.0*(1.0 - exp(-54.09*ucfc11(i,k))) * tw(i,1) * emplnk(7,i)
      ecfc2 = 60.0*(1.0 - exp(-5130.03*ucfc11(i,k)))* tw(i,2) * emplnk(8,i)
      ecfc3 = 60.0*(1.0 - tcfc3)*tw(i,4)*tcfc6*emplnk(9,i)
      ecfc4 = 100.0*(1.0 - tcfc4)*tw(i,5)*emplnk(10,i)
!
! Emissivity for CFC12 bands
!
      ecfc5 = 45.0*(1.0 - exp(-1272.35*ucfc12(i,k)))*tw(i,3)*emplnk(11,i)
      ecfc6 = 50.0*(1.0 - tcfc6)*tw(i,4)*emplnk(12,i)
      ecfc7 = 80.0*(1.0 - tcfc7)*tw(i,5)* tcfc4 * emplnk(13,i)
      ecfc8 = 70.0*(1.0 - tcfc8)*tw(i,6) * emplnk(14,i)
!
! Emissivity for CH4 band 1306 cm-1
!
      tlw = exp(-1.0*sqrt(up2(i)))

!     Overlap H2O vibration rotation band with STRAER continuum 
!             for CH4 1306 cm-1 and N2O 1285 cm-1 bands

            tlw=tlw*aer_trn_ttl(i,k,1,idx_LW_1200_2000)
!bloss: add epsilon to denominator in case ch4 mixing ratio is zero.
      betac = (bch4(i,k)+TINY(uch4(i,k)))/(uch4(i,k)+TINY(uch4(i,k)))
!bloss      betac = bch4(i,k)/uch4(i,k)
      ech4 = 6.00444*sqti(i)*log(1.0 + func(uch4(i,k),betac)) *tlw * emplnk(3,i)
      tch4 = 1.0/(1.0 + 0.02*func(uch4(i,k),betac))
!
! Emissivity for N2O bands
!
      u01 = un2o0(i,k)
      u11 = un2o1(i,k)
      beta01 = (bn2o0(i,k)+TINY(un2o0(i,k)))/(un2o0(i,k)+TINY(un2o0(i,k)))
      beta11 = (bn2o1(i,k)+TINY(un2o1(i,k)))/(un2o1(i,k)+TINY(un2o1(i,k)))
!
! 1285 cm-1 band
!
      en2o1 = 2.35558*sqti(i)*log(1.0 + func(u01,beta01) + &
              func(u11,beta11))*tlw*tch4*emplnk(4,i)
      u02 = 0.100090*u01
      u12 = 0.0992746*u11
      beta02 = 0.964282*beta01
!
! 589 cm-1 band
!
      en2o2 = 2.65581*sqti(i)*log(1.0 + func(u02,beta02) + &
              func(u12,beta02)) * tco2(i) * th2o(i) * emplnk(5,i)
      u03 = 0.0333767*u01
      beta03 = 0.982143*beta01
!
! 1168 cm-1 band
!
      en2o3 = 2.54034*sqti(i)*log(1.0 + func(u03,beta03)) * &
              tw(i,6) * tcfc8 * emplnk(6,i)
!
! Emissivity for 1064 cm-1 band of CO2
!
      betac1 = 2.97558*pnm(i,k) / (sslp*sqti(i))
      betac2 = 2.0 * betac1
      eco21 = 3.7571*sqti(i)*log(1.0 + func(uco211(i,k),betac1) &
              + func(uco212(i,k),betac2) + func(uco213(i,k),betac2)) &
              * to3(i) * tw(i,5) * tcfc4 * tcfc7 * emplnk(2,i)
!
! Emissivity for 961 cm-1 band
!
      eco22 = 3.8443*sqti(i)*log(1.0 + func(uco221(i,k),betac1) &
              + func(uco222(i,k),betac1) + func(uco223(i,k),betac2)) &
              * tw(i,4) * tcfc3 * tcfc6 * emplnk(1,i)
!
! total trace gas emissivity
!
      emstrc(i,k) = ecfc1 + ecfc2 + ecfc3 + ecfc4 + ecfc5 +ecfc6 + &
                    ecfc7 + ecfc8 + en2o1 + en2o2 + en2o3 + ech4 + &
                    eco21 + eco22
   end do
!
   return
!
end subroutine trcems

