!bloss#include <misc.h>
!bloss#include <params.h>
subroutine trcabn(lchnk   ,ncol    ,                            &
                  k2      ,kn      ,ucfc11  ,ucfc12  ,un2o0   , &
                  un2o1   ,uch4    ,uco211  ,uco212  ,uco213  , &
                  uco221  ,uco222  ,uco223  ,tbar    ,bplnk   , &
                  winpl   ,pinpl   ,tco2    ,th2o    ,to3     , &
                  uptype  ,dw      ,s2c     ,up2     ,pnew    , &
                  abstrc  ,uinpl   , &
                  aer_trn_ngh)
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Calculate nearest layer absorptivity due to CH4, N2O, CFC11 and CFC12
! 
! Method: 
! Equations in CCM3 description
! 
! Author: J. Kiehl
! 
!-----------------------------------------------------------------------
!
   use shr_kind_mod, only: r4 => shr_kind_r4
   use ppgrid
   use volcrad

   implicit none
 
!bloss#include <crdcon.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: ncol                  ! number of atmospheric columns
   integer, intent(in) :: k2                    ! level index
   integer, intent(in) :: kn                    ! level index
!
   real(r4), intent(in) :: tbar(pcols,4)        ! pressure weighted temperature
   real(r4), intent(in) :: ucfc11(pcols,pverp)  ! CFC11 path length
   real(r4), intent(in) :: ucfc12(pcols,pverp)  ! CFC12 path length
   real(r4), intent(in) :: un2o0(pcols,pverp)   ! N2O path length
   real(r4), intent(in) :: un2o1(pcols,pverp)   ! N2O path length (hot band)
!
   real(r4), intent(in) :: uch4(pcols,pverp)    ! CH4 path length
   real(r4), intent(in) :: uco211(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco212(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco213(pcols,pverp)  ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco221(pcols,pverp)  ! CO2 10.4 micron band path length
!
   real(r4), intent(in) :: uco222(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco223(pcols,pverp)  ! CO2 10.4 micron band path length
   real(r4), intent(in) :: bplnk(14,pcols,4)    ! weighted Planck fnc. for absorptivity
   real(r4), intent(in) :: winpl(pcols,4)       ! fractional path length
   real(r4), intent(in) :: pinpl(pcols,4)       ! pressure factor for subdivided layer
!
   real(r4), intent(in) :: tco2(pcols)          ! co2 transmission
   real(r4), intent(in) :: th2o(pcols)          ! h2o transmission
   real(r4), intent(in) :: to3(pcols)           ! o3 transmission
   real(r4), intent(in) :: dw(pcols)            ! h2o path length
   real(r4), intent(in) :: pnew(pcols)          ! pressure factor
!
   real(r4), intent(in) :: s2c(pcols,pverp)     ! h2o continuum factor
   real(r4), intent(in) :: uptype(pcols,pverp)  ! p-type path length
   real(r4), intent(in) :: up2(pcols)           ! p squared path length
   real(r4), intent(in) :: uinpl(pcols,4)       ! Nearest layer subdivision factor
   real(r4), intent(in) :: aer_trn_ngh(pcols,bnd_nbr_LW) 
                             ! [fraction] Total transmission between 
                             !            nearest neighbor sub-levels
!
!  Output Arguments
!
   real(r4), intent(out) :: abstrc(pcols)        ! total trace gas absorptivity

!
!--------------------------Local Variables------------------------------
!
   integer i,l                   ! loop counters
!
   real(r4) sqti(pcols)          ! square root of mean temp
   real(r4) rsqti(pcols)         ! reciprocal of sqti
   real(r4) du1                  ! cfc11 path length
   real(r4) du2                  ! cfc12 path length
   real(r4) acfc1                ! absorptivity of cfc11 798 cm-1 band
!
   real(r4) acfc2                ! absorptivity of cfc11 846 cm-1 band
   real(r4) acfc3                ! absorptivity of cfc11 933 cm-1 band
   real(r4) acfc4                ! absorptivity of cfc11 1085 cm-1 band
   real(r4) acfc5                ! absorptivity of cfc11 889 cm-1 band
   real(r4) acfc6                ! absorptivity of cfc11 923 cm-1 band
!
   real(r4) acfc7                ! absorptivity of cfc11 1102 cm-1 band
   real(r4) acfc8                ! absorptivity of cfc11 1161 cm-1 band
   real(r4) du01                 ! n2o path length
   real(r4) dbeta01              ! n2o pressure factors
   real(r4) dbeta11              !        "
!
   real(r4)  an2o1               ! absorptivity of the 1285 cm-1 n2o band
   real(r4) du02                 ! n2o path length
   real(r4) dbeta02              ! n2o pressure factor
   real(r4) an2o2                ! absorptivity of the 589 cm-1 n2o band
   real(r4) du03                 ! n2o path length
!
   real(r4) dbeta03              ! n2o pressure factor
   real(r4) an2o3                ! absorptivity of the 1168 cm-1 n2o band
   real(r4) duch4                ! ch4 path length
   real(r4) dbetac               ! ch4 pressure factor
   real(r4) ach4                 ! absorptivity of the 1306 cm-1 ch4 band
!
   real(r4) du11                 ! co2 path length
   real(r4) du12                 !       "
   real(r4) du13                 !       "
   real(r4) dbetc1               ! co2 pressure factor
   real(r4) dbetc2               ! co2 pressure factor
!
   real(r4) aco21                ! absorptivity of the 1064 cm-1 co2 band
   real(r4) du21                 ! co2 path length
   real(r4) du22                 !       "
   real(r4) du23                 !       "
   real(r4) aco22                ! absorptivity of the 961 cm-1 co2 band
!
   real(r4) tt(pcols)            ! temp. factor for h2o overlap
   real(r4) psi1                 !          "
   real(r4) phi1                 !          "
   real(r4) p1                   ! factor for h2o overlap
   real(r4) w1                   !          "
!
   real(r4) ds2c(pcols)          ! continuum path length
   real(r4) duptyp(pcols)        ! p-type path length
   real(r4) tw(pcols,6)          ! h2o transmission overlap
   real(r4) g1(6)                ! h2o overlap factor
   real(r4) g2(6)                !         "
!
   real(r4) g3(6)                !         "
   real(r4) g4(6)                !         "
   real(r4) ab(6)                ! h2o temp. factor
   real(r4) bb(6)                !         "
   real(r4) abp(6)               !         "
!
   real(r4) bbp(6)               !         "
   real(r4) tcfc3                ! transmission of cfc11 band
   real(r4) tcfc4                ! transmission of cfc11 band
   real(r4) tcfc6                ! transmission of cfc12 band
   real(r4) tcfc7                !         "
!
   real(r4) tcfc8                !         "
   real(r4) tlw                  ! h2o transmission
   real(r4) tch4                 ! ch4 transmission
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
!------------------------------------------------------------------
!
   do i = 1,ncol
      sqti(i) = sqrt(tbar(i,kn))
      rsqti(i) = 1. / sqti(i)
!
! h2o transmission
!
      tt(i) = abs(tbar(i,kn) - 250.0)
      ds2c(i) = abs(s2c(i,k2+1) - s2c(i,k2))*uinpl(i,kn)
      duptyp(i) = abs(uptype(i,k2+1) - uptype(i,k2))*uinpl(i,kn)
   end do
!
   do l = 1,6
      do i = 1,ncol
         psi1 = exp(abp(l)*tt(i)+bbp(l)*tt(i)*tt(i))
         phi1 = exp(ab(l)*tt(i)+bb(l)*tt(i)*tt(i))
         p1 = pnew(i) * (psi1/phi1) / sslp
         w1 = dw(i) * winpl(i,kn) * phi1
         tw(i,l) = exp(- g1(l)*p1*(sqrt(1.0+g2(l)*(w1/p1))-1.0) &
                   - g3(l)*ds2c(i)-g4(l)*duptyp(i))
      end do
   end do
!
   do i=1,ncol
      tw(i,1)=tw(i,1)*(0.7*aer_trn_ngh(i,idx_LW_0650_0800)+&! l=1: 0750--0820 cm-1
                       0.3*aer_trn_ngh(i,idx_LW_0800_1000))
      tw(i,2)=tw(i,2)*aer_trn_ngh(i,idx_LW_0800_1000) ! l=2: 0820--0880 cm-1
      tw(i,3)=tw(i,3)*aer_trn_ngh(i,idx_LW_0800_1000) ! l=3: 0880--0900 cm-1
      tw(i,4)=tw(i,4)*aer_trn_ngh(i,idx_LW_0800_1000) ! l=4: 0900--1000 cm-1
      tw(i,5)=tw(i,5)*aer_trn_ngh(i,idx_LW_1000_1200) ! l=5: 1000--1120 cm-1
      tw(i,6)=tw(i,6)*aer_trn_ngh(i,idx_LW_1000_1200) ! l=6: 1120--1170 cm-1
   end do                    ! end loop over lon

   do i = 1,ncol
!
      du1 = abs(ucfc11(i,k2+1) - ucfc11(i,k2)) * winpl(i,kn)
      du2 = abs(ucfc12(i,k2+1) - ucfc12(i,k2)) * winpl(i,kn)
!
! cfc transmissions
!
      tcfc3 = exp(-175.005*du1)
      tcfc4 = exp(-1202.18*du1)
      tcfc6 = exp(-5786.73*du2)
      tcfc7 = exp(-2873.51*du2)
      tcfc8 = exp(-2085.59*du2)
!
! Absorptivity for CFC11 bands
!
      acfc1 = 50.0*(1.0 - exp(-54.09*du1)) * tw(i,1)*bplnk(7,i,kn)
      acfc2 = 60.0*(1.0 - exp(-5130.03*du1))*tw(i,2)*bplnk(8,i,kn)
      acfc3 = 60.0*(1.0 - tcfc3)*tw(i,4)*tcfc6 * bplnk(9,i,kn)
      acfc4 = 100.0*(1.0 - tcfc4)* tw(i,5) * bplnk(10,i,kn)
!
! Absorptivity for CFC12 bands
!
      acfc5 = 45.0*(1.0 - exp(-1272.35*du2))*tw(i,3)*bplnk(11,i,kn)
      acfc6 = 50.0*(1.0 - tcfc6)*tw(i,4)*bplnk(12,i,kn)
      acfc7 = 80.0*(1.0 - tcfc7)* tw(i,5)*tcfc4 *bplnk(13,i,kn)
      acfc8 = 70.0*(1.0 - tcfc8)*tw(i,6)*bplnk(14,i,kn)
!
! Absorptivity for CH4 band 1306 cm-1
!
      tlw = exp(-1.0*sqrt(up2(i)))
      tlw=tlw*aer_trn_ngh(i,idx_LW_1200_2000)
      duch4 = abs(uch4(i,k2+1) - uch4(i,k2)) * winpl(i,kn)
      dbetac = 2.94449 * pinpl(i,kn) * rsqti(i) / sslp
      ach4 = 6.00444*sqti(i)*log(1.0 + func(duch4,dbetac)) * tlw * bplnk(3,i,kn)
      tch4 = 1.0/(1.0 + 0.02*func(duch4,dbetac))
!
! Absorptivity for N2O bands
!
      du01 = abs(un2o0(i,k2+1) - un2o0(i,k2)) * winpl(i,kn)
      du11 = abs(un2o1(i,k2+1) - un2o1(i,k2)) * winpl(i,kn)
      dbeta01 = 19.399 *  pinpl(i,kn) * rsqti(i) / sslp
      dbeta11 = dbeta01
!
! 1285 cm-1 band
!
      an2o1 = 2.35558*sqti(i)*log(1.0 + func(du01,dbeta01) &
              + func(du11,dbeta11)) * tlw * tch4 * bplnk(4,i,kn)
      du02 = 0.100090*du01
      du12 = 0.0992746*du11
      dbeta02 = 0.964282*dbeta01
!
! 589 cm-1 band
!
      an2o2 = 2.65581*sqti(i)*log(1.0 + func(du02,dbeta02) &
              +  func(du12,dbeta02)) * tco2(i) * th2o(i) * bplnk(5,i,kn)
      du03 = 0.0333767*du01
      dbeta03 = 0.982143*dbeta01
!
! 1168 cm-1 band
!
      an2o3 = 2.54034*sqti(i)*log(1.0 + func(du03,dbeta03)) * &
              tw(i,6) * tcfc8 * bplnk(6,i,kn)
!
! Absorptivity for 1064 cm-1 band of CO2
!
      du11 = abs(uco211(i,k2+1) - uco211(i,k2)) * winpl(i,kn)
      du12 = abs(uco212(i,k2+1) - uco212(i,k2)) * winpl(i,kn)
      du13 = abs(uco213(i,k2+1) - uco213(i,k2)) * winpl(i,kn)
      dbetc1 = 2.97558 * pinpl(i,kn) * rsqti(i) / sslp
      dbetc2 = 2.0 * dbetc1
      aco21 = 3.7571*sqti(i)*log(1.0 + func(du11,dbetc1) &
              + func(du12,dbetc2) + func(du13,dbetc2)) &
              * to3(i) * tw(i,5) * tcfc4 * tcfc7 * bplnk(2,i,kn)
!
! Absorptivity for 961 cm-1 band of co2
!
      du21 = abs(uco221(i,k2+1) - uco221(i,k2)) * winpl(i,kn)
      du22 = abs(uco222(i,k2+1) - uco222(i,k2)) * winpl(i,kn)
      du23 = abs(uco223(i,k2+1) - uco223(i,k2)) * winpl(i,kn)
      aco22 = 3.8443*sqti(i)*log(1.0 + func(du21,dbetc1) &
              + func(du22,dbetc1) + func(du23,dbetc2)) &
              * tw(i,4) * tcfc3 * tcfc6 * bplnk(1,i,kn)
!
! total trace gas absorptivity
!
      abstrc(i) = acfc1 + acfc2 + acfc3 + acfc4 + acfc5 + acfc6 + &
                  acfc7 + acfc8 + an2o1 + an2o2 + an2o3 + ach4 + &
                  aco21 + aco22
   end do
!
   return
!
end subroutine trcabn





