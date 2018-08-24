!bloss #include <misc.h>
!bloss #include <params.h>
module radae
!------------------------------------------------------------------------------
!
! Description:
!
! Data and subroutines to calculate absorptivities and emissivity needed
! for the LW radiation calculation.
!
! Public interfaces are: 
!
! radaeini -------------- Initialization
! initialize_radbuffer -- Initialize the 3D abs/emis arrays.
! radabs ---------------- Compute absorptivities.
! radems ---------------- Compute emissivity.
! radtpl ---------------- Compute Temperatures and path lengths.
!
! Author:  B. Collins
!
! $Id: radae.f90,v 1.1 2004/09/08 19:46:03 samcvs Exp $
!------------------------------------------------------------------------------
  use shr_kind_mod, only: r4 => shr_kind_r4
  use ppgrid
!bloss  use infnan
!bloss  use ghg_surfvals, only: co2vmr
  use volcrad
  use abortutils, only: endrun
!bloss  use prescribed_aerosols, only: strat_volcanic

  implicit none

  save
!-----------------------------------------------------------------------------
! PUBLIC:: By default data and interfaces are private
!-----------------------------------------------------------------------------
  private
  public radabs, radems, radtpl, radaeini, initialize_radbuffer ! Public interface routines

  integer, public, parameter :: nbands = 2          ! Number of spectral bands
!
! Following data needed for restarts and in radclwmx
!
  real(r4), public, allocatable, target :: abstot_3d(:,:,:,:) ! Non-adjacent layer absorptivites
  real(r4), public, allocatable, target :: absnxt_3d(:,:,:,:) ! Nearest layer absorptivities
  real(r4), public, allocatable, target :: emstot_3d(:,:,:)   ! Total emissivity
!-----------------------------------------------------------------------------
! PRIVATE:: The rest of the data is private to this module.
!-----------------------------------------------------------------------------
  real(r4) :: p0    ! Standard pressure (dynes/cm**2)
  real(r4) :: amd   ! Molecular weight of dry air (g/mol)
  real(r4) :: amco2 ! Molecular weight of co2   (g/mol)
  integer, parameter :: n_u = 25   ! Number of U in abs/emis tables
  integer, parameter :: n_p = 10   ! Number of P in abs/emis tables
  integer, parameter :: n_tp = 10  ! Number of T_p in abs/emis tables
  integer, parameter :: n_te = 21  ! Number of T_e in abs/emis tables
  integer, parameter :: n_rh = 7   ! Number of RH in abs/emis tables

  real(r4):: ah2onw(n_p, n_tp, n_u, n_te, n_rh)   ! absorptivity (non-window)
  real(r4):: eh2onw(n_p, n_tp, n_u, n_te, n_rh)   ! emissivity   (non-window)
  real(r4):: ah2ow(n_p, n_tp, n_u, n_te, n_rh)    ! absorptivity (window, for adjacent layers)
  real(r4):: cn_ah2ow(n_p, n_tp, n_u, n_te, n_rh)    ! continuum transmission for absorptivity (window)
  real(r4):: cn_eh2ow(n_p, n_tp, n_u, n_te, n_rh)    ! continuum transmission for emissivity   (window)
  real(r4):: ln_ah2ow(n_p, n_tp, n_u, n_te, n_rh)    ! line-only transmission for absorptivity (window)
  real(r4):: ln_eh2ow(n_p, n_tp, n_u, n_te, n_rh)    ! line-only transmission for emissivity   (window)
!
! Constant coefficients for water vapor overlap with trace gases.
! Reference: Ramanathan, V. and  P.Downey, 1986: A Nonisothermal
!            Emissivity and Absorptivity Formulation for Water Vapor
!            Journal of Geophysical Research, vol. 91., D8, pp 8649-8666
!
  real(r4):: coefh(2,4) = reshape(  &
         (/ (/5.46557e+01,-7.30387e-02/), &
            (/1.09311e+02,-1.46077e-01/), &
            (/5.11479e+01,-6.82615e-02/), &
            (/1.02296e+02,-1.36523e-01/) /), (/2,4/) )
!
  real(r4):: coefj(3,2) = reshape( &
            (/ (/2.82096e-02,2.47836e-04,1.16904e-06/), &
               (/9.27379e-02,8.04454e-04,6.88844e-06/) /), (/3,2/) )
!
  real(r4):: coefk(3,2) = reshape( &
            (/ (/2.48852e-01,2.09667e-03,2.60377e-06/) , &
               (/1.03594e+00,6.58620e-03,4.04456e-06/) /), (/3,2/) )
  real(r4):: c16,c17,c26,c27,c28,c29,c30,c31
!
! Farwing correction constants for narrow-band emissivity model,
! introduced to account for the deficiencies in narrow-band model
! used to derive the emissivity; tuned with Arkings line-by-line
! calculations.  Just used for water vapor overlap with trace gases.
!
  real(r4):: fwcoef      ! Farwing correction constant
  real(r4):: fwc1,fwc2   ! Farwing correction constants 
  real(r4):: fc1         ! Farwing correction constant 
!
! Collins/Hackney/Edwards (C/H/E) & Collins/Lee-Taylor/Edwards (C/LT/E) 
!       H2O parameterization
!
! Notation:
! U   = integral (P/P_0 dW)  eq. 15 in Ramanathan/Downey 1986
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
! absorptivity/emissivity in window are fit using an expression:
!
!      a/e = f_a/e * {1.0 - ln_a/e * cn_a/e} 
!
! absorptivity/emissivity in non-window are fit using:
! 
!      a/e = f_a/e * a/e_norm
!
! where
!      a/e = absorptivity/emissivity
! a/e_norm = absorptivity/emissivity normalized to 1
!    f_a/e = value of a/e as U->infinity = f(T_e) only
!   cn_a/e = continuum transmission
!   ln_a/e = line transmission
!
! spectral interval:
!   1 = 0-800 cm^-1 and 1200-2200 cm^-1 (rotation and rotation-vibration)
!   2 = 800-1200 cm^-1                  (window)
!
! The H2O saturation table spans 160K to 351K in 1K intervals).
!
  real(r4), parameter:: min_tp_h2o = 160.0        ! min T_p for pre-calculated abs/emis 
  real(r4), parameter:: max_tp_h2o = 349.999999   ! max T_p for pre-calculated abs/emis 
  integer, parameter :: ntemp = 192 ! Number of temperatures in H2O sat. table for Tp
  real(r4) :: estblh2o(0:ntemp)       ! saturation vapor pressure for H2O for Tp rang
  integer, parameter :: o_fa = 6   ! Degree+1 of poly of T_e for absorptivity as U->inf.
  integer, parameter :: o_fe = 6   ! Degree+1 of poly of T_e for emissivity as U->inf.
!-----------------------------------------------------------------------------
! Data for f in C/H/E fit -- value of A and E as U->infinity
! New C/LT/E fit (Hitran 2K, CKD 2.4) -- no change
!     These values are determined by integrals of Planck functions or
!     derivatives of Planck functions only.
!-----------------------------------------------------------------------------
!
! fa/fe coefficients for 2 bands (0-800 & 1200-2200, 800-1200 cm^-1)
!
! Coefficients of polynomial for f_a in T_e
!
  real(r4), parameter:: fat(o_fa,nbands) = reshape( (/ &
       (/-1.06665373E-01,  2.90617375E-02, -2.70642049E-04,   &   ! 0-800&1200-2200 cm^-1
          1.07595511E-06, -1.97419681E-09,  1.37763374E-12/), &   !   0-800&1200-2200 cm^-1
       (/ 1.10666537E+00, -2.90617375E-02,  2.70642049E-04,   &   ! 800-1200 cm^-1
         -1.07595511E-06,  1.97419681E-09, -1.37763374E-12/) /) & !   800-1200 cm^-1
       , (/o_fa,nbands/) )
!
! Coefficients of polynomial for f_e in T_e
!
  real(r4), parameter:: fet(o_fe,nbands) = reshape( (/ & 
      (/3.46148163E-01,  1.51240299E-02, -1.21846479E-04,   &   ! 0-800&1200-2200 cm^-1
        4.04970123E-07, -6.15368936E-10,  3.52415071E-13/), &   !   0-800&1200-2200 cm^-1
      (/6.53851837E-01, -1.51240299E-02,  1.21846479E-04,   &   ! 800-1200 cm^-1
       -4.04970123E-07,  6.15368936E-10, -3.52415071E-13/) /) & !   800-1200 cm^-1
      , (/o_fa,nbands/) )
!
! Note: max values should be slightly underestimated to avoid index bound violations
!
  real(r4), parameter:: min_lp_h2o = -3.0         ! min log_10(P) for pre-calculated abs/emis 
  real(r4), parameter:: min_p_h2o = 1.0e-3        ! min log_10(P) for pre-calculated abs/emis 
  real(r4), parameter:: max_lp_h2o = -0.0000001   ! max log_10(P) for pre-calculated abs/emis 
  real(r4), parameter:: dlp_h2o = 0.3333333333333 ! difference in adjacent elements of lp_h2o
 
  real(r4), parameter:: dtp_h2o = 21.111111111111 ! difference in adjacent elements of tp_h2o

  real(r4), parameter:: min_rh_h2o = 0.0          ! min RH for pre-calculated abs/emis 
  real(r4), parameter:: max_rh_h2o = 1.19999999   ! max RH for pre-calculated abs/emis 
  real(r4), parameter:: drh_h2o = 0.2             ! difference in adjacent elements of RH

  real(r4), parameter:: min_te_h2o = -120.0       ! min T_e-T_p for pre-calculated abs/emis 
  real(r4), parameter:: max_te_h2o = 79.999999    ! max T_e-T_p for pre-calculated abs/emis 
  real(r4), parameter:: dte_h2o  = 10.0           ! difference in adjacent elements of te_h2o

  real(r4), parameter:: min_lu_h2o = -8.0         ! min log_10(U) for pre-calculated abs/emis 
  real(r4), parameter:: min_u_h2o  = 1.0e-8       ! min pressure-weighted path-length
  real(r4), parameter:: max_lu_h2o =  3.9999999   ! max log_10(U) for pre-calculated abs/emis 
  real(r4), parameter:: dlu_h2o  = 0.5            ! difference in adjacent elements of lu_h2o

!-----------------------------------------------------------------------------
! Public Interfaces
!-----------------------------------------------------------------------------
CONTAINS

subroutine radabs(lchnk   ,ncol    ,             &
   pbr    ,pnm     ,co2em    ,co2eml  ,tplnka  , &
   s2c    ,tcg     ,w        ,h2otr   ,plco2   , &
   plh2o  ,co2t    ,tint     ,tlayr   ,plol    , &
   plos   ,pmln    ,piln     ,ucfc11  ,ucfc12  , &
   un2o0  ,un2o1   ,uch4     ,uco211  ,uco212  , &
   uco213 ,uco221  ,uco222   ,uco223  ,uptype  , &
   bn2o0  ,bn2o1   ,bch4    ,abplnk1  ,abplnk2 , &
   abstot ,absnxt  ,plh2ob  ,wb       , &
   aer_mpp ,aer_trn_ttl)
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Compute absorptivities for h2o, co2, o3, ch4, n2o, cfc11 and cfc12
! 
! Method: 
! h2o  ....  Uses nonisothermal emissivity method for water vapor from
!            Ramanathan, V. and  P.Downey, 1986: A Nonisothermal
!            Emissivity and Absorptivity Formulation for Water Vapor
!            Journal of Geophysical Research, vol. 91., D8, pp 8649-8666
!
!            Implementation updated by Collins, Hackney, and Edwards (2001)
!               using line-by-line calculations based upon Hitran 1996 and
!               CKD 2.1 for absorptivity and emissivity
!
!            Implementation updated by Collins, Lee-Taylor, and Edwards (2003)
!               using line-by-line calculations based upon Hitran 2000 and
!               CKD 2.4 for absorptivity and emissivity
!
! co2  ....  Uses absorptance parameterization of the 15 micro-meter
!            (500 - 800 cm-1) band system of Carbon Dioxide, from
!            Kiehl, J.T. and B.P.Briegleb, 1991: A New Parameterization
!            of the Absorptance Due to the 15 micro-meter Band System
!            of Carbon Dioxide Jouranl of Geophysical Research,
!            vol. 96., D5, pp 9013-9019.
!            Parameterizations for the 9.4 and 10.4 mircon bands of CO2
!            are also included.
!
! o3   ....  Uses absorptance parameterization of the 9.6 micro-meter
!            band system of ozone, from Ramanathan, V. and R.Dickinson,
!            1979: The Role of stratospheric ozone in the zonal and
!            seasonal radiative energy balance of the earth-troposphere
!            system. Journal of the Atmospheric Sciences, Vol. 36,
!            pp 1084-1104
!
! ch4  ....  Uses a broad band model for the 7.7 micron band of methane.
!
! n20  ....  Uses a broad band model for the 7.8, 8.6 and 17.0 micron
!            bands of nitrous oxide
!
! cfc11 ...  Uses a quasi-linear model for the 9.2, 10.7, 11.8 and 12.5
!            micron bands of CFC11
!
! cfc12 ...  Uses a quasi-linear model for the 8.6, 9.1, 10.8 and 11.2
!            micron bands of CFC12
!
!
! Computes individual absorptivities for non-adjacent layers, accounting
! for band overlap, and sums to obtain the total; then, computes the
! nearest layer contribution.
! 
! Author: W. Collins (H2O absorptivity) and J. Kiehl
! 
!-----------------------------------------------------------------------
!bloss #include <crdcon.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                       ! chunk identifier
   integer, intent(in) :: ncol                        ! number of atmospheric columns

   real(r4), intent(in) :: pbr(pcols,pver)            ! Prssr at mid-levels (dynes/cm2)
   real(r4), intent(in) :: pnm(pcols,pverp)           ! Prssr at interfaces (dynes/cm2)
   real(r4), intent(in) :: co2em(pcols,pverp)         ! Co2 emissivity function
   real(r4), intent(in) :: co2eml(pcols,pver)         ! Co2 emissivity function
   real(r4), intent(in) :: tplnka(pcols,pverp)        ! Planck fnctn level temperature
   real(r4), intent(in) :: s2c(pcols,pverp)           ! H2o continuum path length
   real(r4), intent(in) :: tcg(pcols,pverp)           ! H2o-mass-wgted temp. (Curtis-Godson approx.)
   real(r4), intent(in) :: w(pcols,pverp)             ! H2o prs wghted path
   real(r4), intent(in) :: h2otr(pcols,pverp)         ! H2o trnsmssn fnct for o3 overlap
   real(r4), intent(in) :: plco2(pcols,pverp)         ! Co2 prs wghted path length
   real(r4), intent(in) :: plh2o(pcols,pverp)         ! H2o prs wfhted path length
   real(r4), intent(in) :: co2t(pcols,pverp)          ! Tmp and prs wghted path length
   real(r4), intent(in) :: tint(pcols,pverp)          ! Interface temperatures
   real(r4), intent(in) :: tlayr(pcols,pverp)         ! K-1 level temperatures
   real(r4), intent(in) :: plol(pcols,pverp)          ! Ozone prs wghted path length
   real(r4), intent(in) :: plos(pcols,pverp)          ! Ozone path length
   real(r4), intent(in) :: pmln(pcols,pver)           ! Ln(pmidm1)
   real(r4), intent(in) :: piln(pcols,pverp)          ! Ln(pintm1)
   real(r4), intent(in) :: plh2ob(nbands,pcols,pverp) ! Pressure weighted h2o path with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 
   real(r4), intent(in) :: wb(nbands,pcols,pverp)     ! H2o path length with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 

   real(r4), intent(in) :: aer_mpp(pcols,pverp) ! STRAER path above kth interface level
   real(r4), intent(in) :: aer_trn_ttl(pcols,pverp,pverp,bnd_nbr_LW) ! aer trn.


!
! Trace gas variables
!
   real(r4), intent(in) :: ucfc11(pcols,pverp)        ! CFC11 path length
   real(r4), intent(in) :: ucfc12(pcols,pverp)        ! CFC12 path length
   real(r4), intent(in) :: un2o0(pcols,pverp)         ! N2O path length
   real(r4), intent(in) :: un2o1(pcols,pverp)         ! N2O path length (hot band)
   real(r4), intent(in) :: uch4(pcols,pverp)          ! CH4 path length
   real(r4), intent(in) :: uco211(pcols,pverp)        ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco212(pcols,pverp)        ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco213(pcols,pverp)        ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco221(pcols,pverp)        ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco222(pcols,pverp)        ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco223(pcols,pverp)        ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uptype(pcols,pverp)        ! continuum path length
   real(r4), intent(in) :: bn2o0(pcols,pverp)         ! pressure factor for n2o
   real(r4), intent(in) :: bn2o1(pcols,pverp)         ! pressure factor for n2o
   real(r4), intent(in) :: bch4(pcols,pverp)          ! pressure factor for ch4
   real(r4), intent(in) :: abplnk1(14,pcols,pverp)    ! non-nearest layer Planck factor
   real(r4), intent(in) :: abplnk2(14,pcols,pverp)    ! nearest layer factor
!
! Output arguments
!
   real(r4), intent(out) :: abstot(pcols,pverp,pverp) ! Total absorptivity
   real(r4), intent(out) :: absnxt(pcols,pver,4)      ! Total nearest layer absorptivity
!
!---------------------------Local variables-----------------------------
!
   integer i                   ! Longitude index
   integer k                   ! Level index
   integer k1                  ! Level index
   integer k2                  ! Level index
   integer kn                  ! Nearest level index
   integer wvl                 ! Wavelength index

   real(r4) abstrc(pcols)              ! total trace gas absorptivity
   real(r4) bplnk(14,pcols,4)          ! Planck functions for sub-divided layers
   real(r4) pnew(pcols)        ! Effective pressure for H2O vapor linewidth
   real(r4) pnewb(nbands)      ! Effective pressure for h2o linewidth w/
                               !    Hulst-Curtis-Godson correction for
                               !    each band
   real(r4) u(pcols)           ! Pressure weighted H2O path length
   real(r4) ub(nbands)         ! Pressure weighted H2O path length with
                               !    Hulst-Curtis-Godson correction for
                               !    each band
   real(r4) tbar(pcols,4)      ! Mean layer temperature
   real(r4) emm(pcols,4)       ! Mean co2 emissivity
   real(r4) o3emm(pcols,4)     ! Mean o3 emissivity
   real(r4) o3bndi             ! Ozone band parameter
   real(r4) temh2o(pcols,4)    ! Mean layer temperature equivalent to tbar
   real(r4) k21                ! Exponential coefficient used to calculate
!                              !  rotation band transmissvty in the 650-800
!                              !  cm-1 region (tr1)
   real(r4) k22                ! Exponential coefficient used to calculate
!                              !  rotation band transmissvty in the 500-650
!                              !  cm-1 region (tr2)
   real(r4) uc1(pcols)         ! H2o continuum pathlength in 500-800 cm-1
   real(r4) to3h2o(pcols)      ! H2o trnsmsn for overlap with o3
   real(r4) pi                 ! For co2 absorptivity computation
   real(r4) sqti(pcols)        ! Used to store sqrt of mean temperature
   real(r4) et                 ! Co2 hot band factor
   real(r4) et2                ! Co2 hot band factor squared
   real(r4) et4                ! Co2 hot band factor to fourth power
   real(r4) omet               ! Co2 stimulated emission term
   real(r4) f1co2              ! Co2 central band factor
   real(r4) f2co2(pcols)       ! Co2 weak band factor
   real(r4) f3co2(pcols)       ! Co2 weak band factor
   real(r4) t1co2(pcols)       ! Overlap factr weak bands on strong band
   real(r4) sqwp               ! Sqrt of co2 pathlength
   real(r4) f1sqwp(pcols)      ! Main co2 band factor
   real(r4) oneme              ! Co2 stimulated emission term
   real(r4) alphat             ! Part of the co2 stimulated emission term
   real(r4) wco2               ! Constants used to define co2 pathlength
   real(r4) posqt              ! Effective pressure for co2 line width
   real(r4) u7(pcols)          ! Co2 hot band path length
   real(r4) u8                 ! Co2 hot band path length
   real(r4) u9                 ! Co2 hot band path length
   real(r4) u13                ! Co2 hot band path length
   real(r4) rbeta7(pcols)      ! Inverse of co2 hot band line width par
   real(r4) rbeta8             ! Inverse of co2 hot band line width par
   real(r4) rbeta9             ! Inverse of co2 hot band line width par
   real(r4) rbeta13            ! Inverse of co2 hot band line width par
   real(r4) tpatha             ! For absorptivity computation
   real(r4) abso(pcols,4)      ! Absorptivity for various gases/bands
   real(r4) dtx(pcols)         ! Planck temperature minus 250 K
   real(r4) dty(pcols)         ! Path temperature minus 250 K
   real(r4) term7(pcols,2)     ! Kl_inf(i) in eq(r4) of table A3a of R&D
   real(r4) term8(pcols,2)     ! Delta kl_inf(i) in eq(r4)
   real(r4) tr1                ! Eqn(6) in table A2 of R&D for 650-800
   real(r4) tr10(pcols)        ! Eqn (6) times eq(4) in table A2
!                              !  of R&D for 500-650 cm-1 region
   real(r4) tr2                ! Eqn(6) in table A2 of R&D for 500-650
   real(r4) tr5                ! Eqn(4) in table A2 of R&D for 650-800
   real(r4) tr6                ! Eqn(4) in table A2 of R&D for 500-650
   real(r4) tr9(pcols)         ! Equation (6) times eq(4) in table A2
!                              !  of R&D for 650-800 cm-1 region
   real(r4) sqrtu(pcols)       ! Sqrt of pressure weighted h20 pathlength
   real(r4) fwk(pcols)         ! Equation(33) in R&D far wing correction
   real(r4) fwku(pcols)        ! GU term in eqs(1) and (6) in table A2
   real(r4) to3co2(pcols)      ! P weighted temp in ozone band model
   real(r4) dpnm(pcols)        ! Pressure difference between two levels
   real(r4) pnmsq(pcols,pverp) ! Pressure squared
   real(r4) dw(pcols)          ! Amount of h2o between two levels
   real(r4) uinpl(pcols,4)     ! Nearest layer subdivision factor
   real(r4) winpl(pcols,4)     ! Nearest layer subdivision factor
   real(r4) zinpl(pcols,4)     ! Nearest layer subdivision factor
   real(r4) pinpl(pcols,4)     ! Nearest layer subdivision factor
   real(r4) dplh2o(pcols)      ! Difference in press weighted h2o amount
   real(r4) r293               ! 1/293
   real(r4) r250               ! 1/250
   real(r4) r3205              ! Line width factor for o3 (see R&Di)
   real(r4) r300               ! 1/300
   real(r4) rsslp              ! Reciprocal of sea level pressure
   real(r4) r2sslp             ! 1/2 of rsslp
   real(r4) ds2c               ! Y in eq(7) in table A2 of R&D
   real(r4)  dplos             ! Ozone pathlength eq(A2) in R&Di
   real(r4) dplol              ! Presure weighted ozone pathlength
   real(r4) tlocal             ! Local interface temperature
   real(r4) beta               ! Ozone mean line parameter eq(A3) in R&Di
!                               (includes Voigt line correction factor)
   real(r4) rphat              ! Effective pressure for ozone beta
   real(r4) tcrfac             ! Ozone temperature factor table 1 R&Di
   real(r4) tmp1               ! Ozone band factor see eq(A1) in R&Di
   real(r4) u1                 ! Effective ozone pathlength eq(A2) in R&Di
   real(r4) realnu             ! 1/beta factor in ozone band model eq(A1)
   real(r4) tmp2               ! Ozone band factor see eq(A1) in R&Di
   real(r4) u2                 ! Effective ozone pathlength eq(A2) in R&Di
   real(r4) rsqti              ! Reciprocal of sqrt of path temperature
   real(r4) tpath              ! Path temperature used in co2 band model
   real(r4) tmp3               ! Weak band factor see K&B
   real(r4) rdpnmsq            ! Reciprocal of difference in press^2
   real(r4) rdpnm              ! Reciprocal of difference in press
   real(r4) p1                 ! Mean pressure factor
   real(r4) p2                 ! Mean pressure factor
   real(r4) dtym10             ! T - 260 used in eq(9) and (10) table A3a
   real(r4) dplco2             ! Co2 path length
   real(r4) te                 ! A_0 T factor in ozone model table 1 of R&Di
   real(r4) denom              ! Denominator in eq(r4) of table A3a of R&D
   real(r4) th2o(pcols)        ! transmission due to H2O
   real(r4) tco2(pcols)        ! transmission due to CO2
   real(r4) to3(pcols)         ! transmission due to O3
!
! Transmission terms for various spectral intervals:
!
   real(r4) trab2(pcols)       ! H2o   500 -  800 cm-1
   real(r4) absbnd             ! Proportional to co2 band absorptance
   real(r4) dbvtit(pcols,pverp)! Intrfc drvtv plnck fnctn for o3
   real(r4) dbvtly(pcols,pver) ! Level drvtv plnck fnctn for o3
!
! Variables for Collins/Hackney/Edwards (C/H/E) & 
!       Collins/Lee-Taylor/Edwards (C/LT/E) H2O parameterization

!
! Notation:
! U   = integral (P/P_0 dW)  eq. 15 in Ramanathan/Downey 1986
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
   real(r4) fa               ! asymptotic value of abs. as U->infinity
   real(r4) a_star           ! normalized absorptivity for non-window
   real(r4) l_star           ! interpolated line transmission
   real(r4) c_star           ! interpolated continuum transmission

   real(r4) te1              ! emission temperature
   real(r4) te2              ! te^2
   real(r4) te3              ! te^3
   real(r4) te4              ! te^4
   real(r4) te5              ! te^5

   real(r4) log_u            ! log base 10 of U 
   real(r4) log_uc           ! log base 10 of H2O continuum path
   real(r4) log_p            ! log base 10 of P
   real(r4) t_p              ! T_p
   real(r4) t_e              ! T_e (offset by T_p)

   integer iu                ! index for log10(U)
   integer iu1               ! iu + 1
   integer iuc               ! index for log10(H2O continuum path)
   integer iuc1              ! iuc + 1
   integer ip                ! index for log10(P)
   integer ip1               ! ip + 1
   integer itp               ! index for T_p
   integer itp1              ! itp + 1
   integer ite               ! index for T_e
   integer ite1              ! ite + 1
   integer irh               ! index for RH
   integer irh1              ! irh + 1

   real(r4) dvar             ! normalized variation in T_p/T_e/P/U
   real(r4) uvar             ! U * diffusivity factor
   real(r4) uscl             ! factor for lineary scaling as U->0

   real(r4) wu               ! weight for U
   real(r4) wu1              ! 1 - wu
   real(r4) wuc              ! weight for H2O continuum path
   real(r4) wuc1             ! 1 - wuc
   real(r4) wp               ! weight for P
   real(r4) wp1              ! 1 - wp
   real(r4) wtp              ! weight for T_p
   real(r4) wtp1             ! 1 - wtp
   real(r4) wte              ! weight for T_e
   real(r4) wte1             ! 1 - wte
   real(r4) wrh              ! weight for RH
   real(r4) wrh1             ! 1 - wrh

   real(r4) w_0_0_           ! weight for Tp/Te combination
   real(r4) w_0_1_           ! weight for Tp/Te combination
   real(r4) w_1_0_           ! weight for Tp/Te combination
   real(r4) w_1_1_           ! weight for Tp/Te combination

   real(r4) w_0_00           ! weight for Tp/Te/RH combination
   real(r4) w_0_01           ! weight for Tp/Te/RH combination
   real(r4) w_0_10           ! weight for Tp/Te/RH combination
   real(r4) w_0_11           ! weight for Tp/Te/RH combination
   real(r4) w_1_00           ! weight for Tp/Te/RH combination
   real(r4) w_1_01           ! weight for Tp/Te/RH combination
   real(r4) w_1_10           ! weight for Tp/Te/RH combination
   real(r4) w_1_11           ! weight for Tp/Te/RH combination

   real(r4) w00_00           ! weight for P/Tp/Te/RH combination
   real(r4) w00_01           ! weight for P/Tp/Te/RH combination
   real(r4) w00_10           ! weight for P/Tp/Te/RH combination
   real(r4) w00_11           ! weight for P/Tp/Te/RH combination
   real(r4) w01_00           ! weight for P/Tp/Te/RH combination
   real(r4) w01_01           ! weight for P/Tp/Te/RH combination
   real(r4) w01_10           ! weight for P/Tp/Te/RH combination
   real(r4) w01_11           ! weight for P/Tp/Te/RH combination
   real(r4) w10_00           ! weight for P/Tp/Te/RH combination
   real(r4) w10_01           ! weight for P/Tp/Te/RH combination
   real(r4) w10_10           ! weight for P/Tp/Te/RH combination
   real(r4) w10_11           ! weight for P/Tp/Te/RH combination
   real(r4) w11_00           ! weight for P/Tp/Te/RH combination
   real(r4) w11_01           ! weight for P/Tp/Te/RH combination
   real(r4) w11_10           ! weight for P/Tp/Te/RH combination
   real(r4) w11_11           ! weight for P/Tp/Te/RH combination

   integer ib                ! spectral interval:
                             !   1 = 0-800 cm^-1 and 1200-2200 cm^-1
                             !   2 = 800-1200 cm^-1


   real(r4) pch2o            ! H2O continuum path
   real(r4) fch2o            ! temp. factor for continuum
   real(r4) uch2o            ! U corresponding to H2O cont. path (window)

   real(r4) fdif             ! secant(zenith angle) for diffusivity approx.

   real(r4) sslp_mks         ! Sea-level pressure in MKS units
   real(r4) esx              ! saturation vapor pressure returned by vqsatd
   real(r4) qsx              ! saturation mixing ratio returned by vqsatd
   real(r4) pnew_mks         ! pnew in MKS units
   real(r4) q_path           ! effective specific humidity along path
   real(r4) rh_path          ! effective relative humidity along path
   real(r4) omeps            ! 1 - epsilo

   integer  iest             ! index in estblh2o

      integer bnd_idx        ! LW band index
      real(r4) aer_pth_dlt   ! [kg m-2] STRAER path between interface levels k1 and k2
      real(r4) aer_pth_ngh(pcols)
                             ! [kg m-2] STRAER path between neighboring layers
      real(r4) odap_aer_ttl  ! [fraction] Total path absorption optical depth
      real(r4) aer_trn_ngh(pcols,bnd_nbr_LW) 
                             ! [fraction] Total transmission between 
                             !            nearest neighbor sub-levels
!
!--------------------------Statement function---------------------------
!
   real(r4) dbvt,t             ! Planck fnctn tmp derivative for o3
!
   dbvt(t)=(-2.8911366682e-4+(2.3771251896e-6+1.1305188929e-10*t)*t)/ &
      (1.0+(-6.1364820707e-3+1.5550319767e-5*t)*t)
!
!
!-----------------------------------------------------------------------
!
! Initialize
!
   do k2=1,ntoplw-1
      do k1=1,ntoplw-1
!        abstot(:,k1,k2) = inf    ! set unused portions for lf95 restart write
         abstot(:,k1,k2) = 0.     ! set unused portions for lf95 restart write
      end do
   end do
   do k2=1,4
      do k1=1,ntoplw-1
!        absnxt(:,k1,k2) = inf    ! set unused portions for lf95 restart write
         absnxt(:,k1,k2) = 0.     ! set unused portions for lf95 restart write
      end do
   end do

   do k=ntoplw,pverp
!     abstot(:,k,k) = inf         ! set unused portions for lf95 restart write
      abstot(:,k,k) = 0.          ! set unused portions for lf95 restart write
   end do

   do k=ntoplw,pver
      do i=1,ncol
         dbvtly(i,k) = dbvt(tlayr(i,k+1))
         dbvtit(i,k) = dbvt(tint(i,k))
      end do
   end do
   do i=1,ncol
      dbvtit(i,pverp) = dbvt(tint(i,pverp))
   end do
!
   r293    = 1./293.
   r250    = 1./250.
   r3205   = 1./.3205
   r300    = 1./300.
   rsslp   = 1./sslp
   r2sslp  = 1./(2.*sslp)
!
!Constants for computing U corresponding to H2O cont. path
!
   fdif       = 1.66
   sslp_mks   = sslp / 10.0
   omeps      = 1.0 - epsilo
!
! Non-adjacent layer absorptivity:
!
! abso(i,1)     0 -  800 cm-1   h2o rotation band
! abso(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! abso(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O absorptivity
!
! 500-800 cm^-1 H2o continuum/line overlap already included
!                in abso(i,1).  This used to be in abso(i,4)
!
! abso(i,3)   o3  9.6 micrometer band (nu3 and nu1 bands)
! abso(i,4)   co2 15  micrometer band system
!
   do k=ntoplw,pverp
      do i=1,ncol
         pnmsq(i,k) = pnm(i,k)**2
         dtx(i) = tplnka(i,k) - 250.
      end do
   end do
!
! Non-nearest layer level loops
!
   do k1=pverp,ntoplw,-1
      do k2=pverp,ntoplw,-1
         if (k1 == k2) cycle
         do i=1,ncol
            dplh2o(i) = plh2o(i,k1) - plh2o(i,k2)
            u(i)      = abs(dplh2o(i))
            sqrtu(i)  = sqrt(u(i))
            ds2c      = abs(s2c(i,k1) - s2c(i,k2))
            dw(i)     = abs(w(i,k1) - w(i,k2))
            uc1(i)    = (ds2c + 1.7e-3*u(i))*(1. +  2.*ds2c)/(1. + 15.*ds2c)
            pch2o     = ds2c
            pnew(i)   = u(i)/dw(i)
            pnew_mks  = pnew(i) * sslp_mks
!
! Changed effective path temperature to std. Curtis-Godson form
!
            tpatha = abs(tcg(i,k1) - tcg(i,k2))/dw(i)
            t_p = min(max(tpatha, min_tp_h2o), max_tp_h2o)
            iest = floor(t_p) - min_tp_h2o
            esx = estblh2o(iest) + (estblh2o(iest+1)-estblh2o(iest)) * &
                 (t_p - min_tp_h2o - iest)
            qsx = epsilo * esx / (pnew_mks - omeps * esx)
!
! Compute effective RH along path
!
            q_path = dw(i) / abs(pnm(i,k1) - pnm(i,k2)) / rga
!
! Calculate effective u, pnew for each band using
!        Hulst-Curtis-Godson approximation:
! Formulae: Goody and Yung, Atmospheric Radiation: Theoretical Basis, 
!           2nd edition, Oxford University Press, 1989.
! Effective H2O path (w)
!      eq. 6.24, p. 228
! Effective H2O path pressure (pnew = u/w):
!      eq. 6.29, p. 228
!
            ub(1) = abs(plh2ob(1,i,k1) - plh2ob(1,i,k2)) / psi(t_p,1)
            ub(2) = abs(plh2ob(2,i,k1) - plh2ob(2,i,k2)) / psi(t_p,2)
            
            pnewb(1) = ub(1) / abs(wb(1,i,k1) - wb(1,i,k2)) * phi(t_p,1)
            pnewb(2) = ub(2) / abs(wb(2,i,k1) - wb(2,i,k2)) * phi(t_p,2)

            dtx(i)      = tplnka(i,k2) - 250.
            dty(i)      = tpatha       - 250.

            fwk(i)  = fwcoef + fwc1/(1. + fwc2*u(i))
            fwku(i) = fwk(i)*u(i)
!
! Define variables for C/H/E (now C/LT/E) fit
!
! abso(i,1)     0 -  800 cm-1   h2o rotation band
! abso(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! abso(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O absorptivity
!
! Notation:
! U   = integral (P/P_0 dW)  
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
!
! Terms for asymptotic value of emissivity
!
            te1  = tplnka(i,k2)
            te2  = te1 * te1
            te3  = te2 * te1
            te4  = te3 * te1
            te5  = te4 * te1

!
!  Band-independent indices for lines and continuum tables
!
            dvar = (t_p - min_tp_h2o) / dtp_h2o
            itp = min(max(int(aint(dvar,r4)) + 1, 1), n_tp - 1)
            itp1 = itp + 1
            wtp = dvar - floor(dvar)
            wtp1 = 1.0 - wtp
            
            t_e = min(max(tplnka(i,k2)-t_p, min_te_h2o), max_te_h2o)
            dvar = (t_e - min_te_h2o) / dte_h2o
            ite = min(max(int(aint(dvar,r4)) + 1, 1), n_te - 1)
            ite1 = ite + 1
            wte = dvar - floor(dvar)
            wte1 = 1.0 - wte
            
            rh_path = min(max(q_path / qsx, min_rh_h2o), max_rh_h2o)
            dvar = (rh_path - min_rh_h2o) / drh_h2o
            irh = min(max(int(aint(dvar,r4)) + 1, 1), n_rh - 1)
            irh1 = irh + 1
            wrh = dvar - floor(dvar)
            wrh1 = 1.0 - wrh

            w_0_0_ = wtp  * wte
            w_0_1_ = wtp  * wte1
            w_1_0_ = wtp1 * wte 
            w_1_1_ = wtp1 * wte1
            
            w_0_00 = w_0_0_ * wrh
            w_0_01 = w_0_0_ * wrh1
            w_0_10 = w_0_1_ * wrh
            w_0_11 = w_0_1_ * wrh1
            w_1_00 = w_1_0_ * wrh
            w_1_01 = w_1_0_ * wrh1
            w_1_10 = w_1_1_ * wrh
            w_1_11 = w_1_1_ * wrh1

!
! H2O Continuum path for 0-800 and 1200-2200 cm^-1
!
!    Assume foreign continuum dominates total H2O continuum in these bands
!    per Clough et al, JGR, v. 97, no. D14 (Oct 20, 1992), p. 15776
!    Then the effective H2O path is just 
!         U_c = integral[ f(P) dW ]
!    where 
!           W = water-vapor mass and 
!        f(P) = dependence of foreign continuum on pressure 
!             = P / sslp
!    Then 
!         U_c = U (the same effective H2O path as for lines)
!
!
! Continuum terms for 800-1200 cm^-1
!
!    Assume self continuum dominates total H2O continuum for this band
!    per Clough et al, JGR, v. 97, no. D14 (Oct 20, 1992), p. 15776
!    Then the effective H2O self-continuum path is 
!         U_c = integral[ h(e,T) dW ]                        (*eq. 1*)
!    where 
!           W = water-vapor mass and 
!           e = partial pressure of H2O along path
!           T = temperature along path
!      h(e,T) = dependence of foreign continuum on e,T
!             = e / sslp * f(T)
!
!    Replacing
!           e =~ q * P / epsilo
!           q = mixing ratio of H2O
!     epsilo = 0.622
!
!    and using the definition
!           U = integral [ (P / sslp) dW ]
!             = (P / sslp) W                                 (homogeneous path)
!
!    the effective path length for the self continuum is
!         U_c = (q / epsilo) f(T) U                         (*eq. 2*)
!
!    Once values of T, U, and q have been calculated for the inhomogeneous
!        path, this sets U_c for the corresponding
!        homogeneous atmosphere.  However, this need not equal the
!        value of U_c' defined by eq. 1 for the actual inhomogeneous atmosphere
!        under consideration.
!
!    Solution: hold T and q constant, solve for U' that gives U_c' by
!        inverting eq. (2):
!
!        U' = (U_c * epsilo) / (q * f(T))
!
            fch2o = fh2oself(t_p) 
            uch2o = (pch2o * epsilo) / (q_path * fch2o)

!
! Band-dependent indices for non-window
!
            ib = 1

            uvar = ub(ib) * fdif
            log_u  = min(log10(max(uvar, min_u_h2o)), max_lu_h2o)
            dvar = (log_u - min_lu_h2o) / dlu_h2o
            iu = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
            iu1 = iu + 1
            wu = dvar - floor(dvar)
            wu1 = 1.0 - wu
            
            log_p  = min(log10(max(pnewb(ib), min_p_h2o)), max_lp_h2o)
            dvar = (log_p - min_lp_h2o) / dlp_h2o
            ip = min(max(int(aint(dvar,r4)) + 1, 1), n_p - 1)
            ip1 = ip + 1
            wp = dvar - floor(dvar)
            wp1 = 1.0 - wp
         
            w00_00 = wp  * w_0_00 
            w00_01 = wp  * w_0_01 
            w00_10 = wp  * w_0_10 
            w00_11 = wp  * w_0_11 
            w01_00 = wp  * w_1_00 
            w01_01 = wp  * w_1_01 
            w01_10 = wp  * w_1_10 
            w01_11 = wp  * w_1_11 
            w10_00 = wp1 * w_0_00 
            w10_01 = wp1 * w_0_01 
            w10_10 = wp1 * w_0_10 
            w10_11 = wp1 * w_0_11 
            w11_00 = wp1 * w_1_00 
            w11_01 = wp1 * w_1_01 
            w11_10 = wp1 * w_1_10 
            w11_11 = wp1 * w_1_11 
!
! Asymptotic value of absorptivity as U->infinity
!
            fa = fat(1,ib) + &
                 fat(2,ib) * te1 + &
                 fat(3,ib) * te2 + &
                 fat(4,ib) * te3 + &
                 fat(5,ib) * te4 + &
                 fat(6,ib) * te5

            a_star = &
                 ah2onw(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
                 ah2onw(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
                 ah2onw(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
                 ah2onw(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
                 ah2onw(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
                 ah2onw(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
                 ah2onw(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
                 ah2onw(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
                 ah2onw(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
                 ah2onw(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
                 ah2onw(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
                 ah2onw(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
                 ah2onw(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
                 ah2onw(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
                 ah2onw(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
                 ah2onw(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
                 ah2onw(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
                 ah2onw(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
                 ah2onw(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
                 ah2onw(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
                 ah2onw(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
                 ah2onw(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
                 ah2onw(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
                 ah2onw(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
                 ah2onw(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
                 ah2onw(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu 
            abso(i,ib) = min(max(fa * (1.0 - (1.0 - a_star) * &
                                 aer_trn_ttl(i,k1,k2,ib)), &
                             0.0), 1.0)
!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
            if (uvar < min_u_h2o) then
               uscl = uvar / min_u_h2o
               abso(i,ib) = abso(i,ib) * uscl
            endif
                         
!
! Band-dependent indices for window
!
            ib = 2

            uvar = ub(ib) * fdif
            log_u  = min(log10(max(uvar, min_u_h2o)), max_lu_h2o)
            dvar = (log_u - min_lu_h2o) / dlu_h2o
            iu = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
            iu1 = iu + 1
            wu = dvar - floor(dvar)
            wu1 = 1.0 - wu
            
            log_p  = min(log10(max(pnewb(ib), min_p_h2o)), max_lp_h2o)
            dvar = (log_p - min_lp_h2o) / dlp_h2o
            ip = min(max(int(aint(dvar,r4)) + 1, 1), n_p - 1)
            ip1 = ip + 1
            wp = dvar - floor(dvar)
            wp1 = 1.0 - wp
         
            w00_00 = wp  * w_0_00 
            w00_01 = wp  * w_0_01 
            w00_10 = wp  * w_0_10 
            w00_11 = wp  * w_0_11 
            w01_00 = wp  * w_1_00 
            w01_01 = wp  * w_1_01 
            w01_10 = wp  * w_1_10 
            w01_11 = wp  * w_1_11 
            w10_00 = wp1 * w_0_00 
            w10_01 = wp1 * w_0_01 
            w10_10 = wp1 * w_0_10 
            w10_11 = wp1 * w_0_11 
            w11_00 = wp1 * w_1_00 
            w11_01 = wp1 * w_1_01 
            w11_10 = wp1 * w_1_10 
            w11_11 = wp1 * w_1_11 

            log_uc  = min(log10(max(uch2o * fdif, min_u_h2o)), max_lu_h2o)
            dvar = (log_uc - min_lu_h2o) / dlu_h2o
            iuc = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
            iuc1 = iuc + 1
            wuc = dvar - floor(dvar)
            wuc1 = 1.0 - wuc
!
! Asymptotic value of absorptivity as U->infinity
!
            fa = fat(1,ib) + &
                 fat(2,ib) * te1 + &
                 fat(3,ib) * te2 + &
                 fat(4,ib) * te3 + &
                 fat(5,ib) * te4 + &
                 fat(6,ib) * te5

            l_star = &
                 ln_ah2ow(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
                 ln_ah2ow(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
                 ln_ah2ow(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
                 ln_ah2ow(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
                 ln_ah2ow(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
                 ln_ah2ow(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
                 ln_ah2ow(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
                 ln_ah2ow(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
                 ln_ah2ow(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
                 ln_ah2ow(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
                 ln_ah2ow(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
                 ln_ah2ow(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
                 ln_ah2ow(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
                 ln_ah2ow(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
                 ln_ah2ow(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
                 ln_ah2ow(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
                 ln_ah2ow(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
                 ln_ah2ow(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
                 ln_ah2ow(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
                 ln_ah2ow(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
                 ln_ah2ow(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
                 ln_ah2ow(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
                 ln_ah2ow(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
                 ln_ah2ow(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
                 ln_ah2ow(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
                 ln_ah2ow(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
                 ln_ah2ow(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
                 ln_ah2ow(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
                 ln_ah2ow(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
                 ln_ah2ow(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
                 ln_ah2ow(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
                 ln_ah2ow(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu 

            c_star = &
                 cn_ah2ow(ip , itp , iuc , ite , irh ) * w11_11 * wuc1 + &
                 cn_ah2ow(ip , itp , iuc , ite , irh1) * w11_10 * wuc1 + &
                 cn_ah2ow(ip , itp , iuc , ite1, irh ) * w11_01 * wuc1 + &
                 cn_ah2ow(ip , itp , iuc , ite1, irh1) * w11_00 * wuc1 + &
                 cn_ah2ow(ip , itp , iuc1, ite , irh ) * w11_11 * wuc  + &
                 cn_ah2ow(ip , itp , iuc1, ite , irh1) * w11_10 * wuc  + &
                 cn_ah2ow(ip , itp , iuc1, ite1, irh ) * w11_01 * wuc  + &
                 cn_ah2ow(ip , itp , iuc1, ite1, irh1) * w11_00 * wuc  + &
                 cn_ah2ow(ip , itp1, iuc , ite , irh ) * w10_11 * wuc1 + &
                 cn_ah2ow(ip , itp1, iuc , ite , irh1) * w10_10 * wuc1 + &
                 cn_ah2ow(ip , itp1, iuc , ite1, irh ) * w10_01 * wuc1 + &
                 cn_ah2ow(ip , itp1, iuc , ite1, irh1) * w10_00 * wuc1 + &
                 cn_ah2ow(ip , itp1, iuc1, ite , irh ) * w10_11 * wuc  + &
                 cn_ah2ow(ip , itp1, iuc1, ite , irh1) * w10_10 * wuc  + &
                 cn_ah2ow(ip , itp1, iuc1, ite1, irh ) * w10_01 * wuc  + &
                 cn_ah2ow(ip , itp1, iuc1, ite1, irh1) * w10_00 * wuc  + &
                 cn_ah2ow(ip1, itp , iuc , ite , irh ) * w01_11 * wuc1 + &
                 cn_ah2ow(ip1, itp , iuc , ite , irh1) * w01_10 * wuc1 + &
                 cn_ah2ow(ip1, itp , iuc , ite1, irh ) * w01_01 * wuc1 + &
                 cn_ah2ow(ip1, itp , iuc , ite1, irh1) * w01_00 * wuc1 + &
                 cn_ah2ow(ip1, itp , iuc1, ite , irh ) * w01_11 * wuc  + &
                 cn_ah2ow(ip1, itp , iuc1, ite , irh1) * w01_10 * wuc  + &
                 cn_ah2ow(ip1, itp , iuc1, ite1, irh ) * w01_01 * wuc  + &
                 cn_ah2ow(ip1, itp , iuc1, ite1, irh1) * w01_00 * wuc  + &
                 cn_ah2ow(ip1, itp1, iuc , ite , irh ) * w00_11 * wuc1 + &
                 cn_ah2ow(ip1, itp1, iuc , ite , irh1) * w00_10 * wuc1 + &
                 cn_ah2ow(ip1, itp1, iuc , ite1, irh ) * w00_01 * wuc1 + &
                 cn_ah2ow(ip1, itp1, iuc , ite1, irh1) * w00_00 * wuc1 + &
                 cn_ah2ow(ip1, itp1, iuc1, ite , irh ) * w00_11 * wuc  + &
                 cn_ah2ow(ip1, itp1, iuc1, ite , irh1) * w00_10 * wuc  + &
                 cn_ah2ow(ip1, itp1, iuc1, ite1, irh ) * w00_01 * wuc  + &
                 cn_ah2ow(ip1, itp1, iuc1, ite1, irh1) * w00_00 * wuc 
            abso(i,ib) = min(max(fa * (1.0 - l_star * c_star * &
                                 aer_trn_ttl(i,k1,k2,ib)), &
                             0.0), 1.0) 
!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
            if (uvar < min_u_h2o) then
               uscl = uvar / min_u_h2o
               abso(i,ib) = abso(i,ib) * uscl
            endif

         end do
!
! Line transmission in 800-1000 and 1000-1200 cm-1 intervals
!
         do i=1,ncol
            term7(i,1) = coefj(1,1) + coefj(2,1)*dty(i)*(1. + c16*dty(i))
            term8(i,1) = coefk(1,1) + coefk(2,1)*dty(i)*(1. + c17*dty(i))
            term7(i,2) = coefj(1,2) + coefj(2,2)*dty(i)*(1. + c26*dty(i))
            term8(i,2) = coefk(1,2) + coefk(2,2)*dty(i)*(1. + c27*dty(i))
         end do
!
! 500 -  800 cm-1   h2o rotation band overlap with co2
!
         do i=1,ncol
            k21    = term7(i,1) + term8(i,1)/ &
               (1. + (c30 + c31*(dty(i)-10.)*(dty(i)-10.))*sqrtu(i))
            k22    = term7(i,2) + term8(i,2)/ &
               (1. + (c28 + c29*(dty(i)-10.))*sqrtu(i))
            tr1    = exp(-(k21*(sqrtu(i) + fc1*fwku(i))))
            tr2    = exp(-(k22*(sqrtu(i) + fc1*fwku(i))))
            tr1=tr1*aer_trn_ttl(i,k1,k2,idx_LW_0650_0800) 
!                                          ! H2O line+STRAER trn 650--800 cm-1
            tr2=tr2*aer_trn_ttl(i,k1,k2,idx_LW_0500_0650)
!                                          ! H2O line+STRAER trn 500--650 cm-1
            tr5    = exp(-((coefh(1,3) + coefh(2,3)*dtx(i))*uc1(i)))
            tr6    = exp(-((coefh(1,4) + coefh(2,4)*dtx(i))*uc1(i)))
            tr9(i)   = tr1*tr5
            tr10(i)  = tr2*tr6
            th2o(i) = tr10(i)
            trab2(i) = 0.65*tr9(i) + 0.35*tr10(i)
         end do
         if (k2 < k1) then
            do i=1,ncol
               to3h2o(i) = h2otr(i,k1)/h2otr(i,k2)
            end do
         else
            do i=1,ncol
               to3h2o(i) = h2otr(i,k2)/h2otr(i,k1)
            end do
         end if
!
! abso(i,3)   o3  9.6 micrometer band (nu3 and nu1 bands)
!
         do i=1,ncol
            dpnm(i)  = pnm(i,k1) - pnm(i,k2)
            to3co2(i) = (pnm(i,k1)*co2t(i,k1) - pnm(i,k2)*co2t(i,k2))/dpnm(i)
            te       = (to3co2(i)*r293)**.7
            dplos    = plos(i,k1) - plos(i,k2)
            dplol    = plol(i,k1) - plol(i,k2)
            u1       = 18.29*abs(dplos)/te
            u2       = .5649*abs(dplos)/te
            rphat    = dplol/dplos
            tlocal   = tint(i,k2)
            tcrfac   = sqrt(tlocal*r250)*te
            beta     = r3205*(rphat + dpfo3*tcrfac)
            realnu   = te/beta
            tmp1     = u1/sqrt(4. + u1*(1. + realnu))
            tmp2     = u2/sqrt(4. + u2*(1. + realnu))
            o3bndi    = 74.*te*log(1. + tmp1 + tmp2)
            abso(i,3) = o3bndi*to3h2o(i)*dbvtit(i,k2)
            to3(i)   = 1.0/(1. + 0.1*tmp1 + 0.1*tmp2)
         end do
!
! abso(i,4)      co2 15  micrometer band system
!
         do i=1,ncol
            sqwp      = sqrt(abs(plco2(i,k1) - plco2(i,k2)))
            et        = exp(-480./to3co2(i))
            sqti(i)   = sqrt(to3co2(i))
            rsqti     = 1./sqti(i)
            et2       = et*et
            et4       = et2*et2
            omet      = 1. - 1.5*et2
            f1co2     = 899.70*omet*(1. + 1.94774*et + 4.73486*et2)*rsqti
            f1sqwp(i) = f1co2*sqwp
            t1co2(i)  = 1./(1. + (245.18*omet*sqwp*rsqti))
            oneme     = 1. - et2
            alphat    = oneme**3*rsqti
            pi        = abs(dpnm(i))
            wco2      =  2.5221*co2vmr*pi*rga
            u7(i)     =  4.9411e4*alphat*et2*wco2
            u8        =  3.9744e4*alphat*et4*wco2
            u9        =  1.0447e5*alphat*et4*et2*wco2
            u13       = 2.8388e3*alphat*et4*wco2
            tpath     = to3co2(i)
            tlocal    = tint(i,k2)
            tcrfac    = sqrt(tlocal*r250*tpath*r300)
            posqt     = ((pnm(i,k2) + pnm(i,k1))*r2sslp + dpfco2*tcrfac)*rsqti
            rbeta7(i) = 1./(5.3228*posqt)
            rbeta8    = 1./(10.6576*posqt)
            rbeta9    = rbeta7(i)
            rbeta13   = rbeta9
            f2co2(i)  = (u7(i)/sqrt(4. + u7(i)*(1. + rbeta7(i)))) + &
               (u8   /sqrt(4. + u8*(1. + rbeta8))) + &
               (u9   /sqrt(4. + u9*(1. + rbeta9)))
            f3co2(i)  = u13/sqrt(4. + u13*(1. + rbeta13))
         end do
         if (k2 >= k1) then
            do i=1,ncol
               sqti(i) = sqrt(tlayr(i,k2))
            end do
         end if
!
         do i=1,ncol
            tmp1      = log(1. + f1sqwp(i))
            tmp2      = log(1. + f2co2(i))
            tmp3      = log(1. + f3co2(i))
            absbnd    = (tmp1 + 2.*t1co2(i)*tmp2 + 2.*tmp3)*sqti(i)
            abso(i,4) = trab2(i)*co2em(i,k2)*absbnd
            tco2(i)   = 1./(1.0+10.0*(u7(i)/sqrt(4. + u7(i)*(1. + rbeta7(i)))))
         end do
!
! Calculate absorptivity due to trace gases, abstrc
!
         call trcab( lchnk   ,ncol    ,                   &
            k1      ,k2      ,ucfc11  ,ucfc12  ,un2o0   , &
            un2o1   ,uch4    ,uco211  ,uco212  ,uco213  , &
            uco221  ,uco222  ,uco223  ,bn2o0   ,bn2o1   , &
            bch4    ,to3co2  ,pnm     ,dw      ,pnew    , &
            s2c     ,uptype  ,u       ,abplnk1 ,tco2    , &
            th2o    ,to3     ,abstrc  , &
            aer_trn_ttl)
!
! Sum total absorptivity
!
         do i=1,ncol
            abstot(i,k1,k2) = abso(i,1) + abso(i,2) + &
               abso(i,3) + abso(i,4) + abstrc(i)
         end do
      end do ! do k2 = 
   end do ! do k1 = 
!
! Adjacent layer absorptivity:
!
! abso(i,1)     0 -  800 cm-1   h2o rotation band
! abso(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! abso(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O absorptivity
!
! 500-800 cm^-1 H2o continuum/line overlap already included
!                in abso(i,1).  This used to be in abso(i,4)
!
! abso(i,3)   o3  9.6 micrometer band (nu3 and nu1 bands)
! abso(i,4)   co2 15  micrometer band system
!
! Nearest layer level loop
!
   do k2=pver,ntoplw,-1
      do i=1,ncol
         tbar(i,1)   = 0.5*(tint(i,k2+1) + tlayr(i,k2+1))
         emm(i,1)    = 0.5*(co2em(i,k2+1) + co2eml(i,k2))
         tbar(i,2)   = 0.5*(tlayr(i,k2+1) + tint(i,k2))
         emm(i,2)    = 0.5*(co2em(i,k2) + co2eml(i,k2))
         tbar(i,3)   = 0.5*(tbar(i,2) + tbar(i,1))
         emm(i,3)    = emm(i,1)
         tbar(i,4)   = tbar(i,3)
         emm(i,4)    = emm(i,2)
         o3emm(i,1)  = 0.5*(dbvtit(i,k2+1) + dbvtly(i,k2))
         o3emm(i,2)  = 0.5*(dbvtit(i,k2) + dbvtly(i,k2))
         o3emm(i,3)  = o3emm(i,1)
         o3emm(i,4)  = o3emm(i,2)
         temh2o(i,1) = tbar(i,1)
         temh2o(i,2) = tbar(i,2)
         temh2o(i,3) = tbar(i,1)
         temh2o(i,4) = tbar(i,2)
         dpnm(i)     = pnm(i,k2+1) - pnm(i,k2)
      end do
!
!  Weighted Planck functions for trace gases
!
      do wvl = 1,14
         do i = 1,ncol
            bplnk(wvl,i,1) = 0.5*(abplnk1(wvl,i,k2+1) + abplnk2(wvl,i,k2))
            bplnk(wvl,i,2) = 0.5*(abplnk1(wvl,i,k2) + abplnk2(wvl,i,k2))
            bplnk(wvl,i,3) = bplnk(wvl,i,1)
            bplnk(wvl,i,4) = bplnk(wvl,i,2)
         end do
      end do
      
      do i=1,ncol
         rdpnmsq    = 1./(pnmsq(i,k2+1) - pnmsq(i,k2))
         rdpnm      = 1./dpnm(i)
         p1         = .5*(pbr(i,k2) + pnm(i,k2+1))
         p2         = .5*(pbr(i,k2) + pnm(i,k2  ))
         uinpl(i,1) =  (pnmsq(i,k2+1) - p1**2)*rdpnmsq
         uinpl(i,2) = -(pnmsq(i,k2  ) - p2**2)*rdpnmsq
         uinpl(i,3) = -(pnmsq(i,k2  ) - p1**2)*rdpnmsq
         uinpl(i,4) =  (pnmsq(i,k2+1) - p2**2)*rdpnmsq
         winpl(i,1) = (.5*( pnm(i,k2+1) - pbr(i,k2)))*rdpnm
         winpl(i,2) = (.5*(-pnm(i,k2  ) + pbr(i,k2)))*rdpnm
         winpl(i,3) = (.5*( pnm(i,k2+1) + pbr(i,k2)) - pnm(i,k2  ))*rdpnm
         winpl(i,4) = (.5*(-pnm(i,k2  ) - pbr(i,k2)) + pnm(i,k2+1))*rdpnm
         tmp1       = 1./(piln(i,k2+1) - piln(i,k2))
         tmp2       = piln(i,k2+1) - pmln(i,k2)
         tmp3       = piln(i,k2  ) - pmln(i,k2)
         zinpl(i,1) = (.5*tmp2          )*tmp1
         zinpl(i,2) = (        - .5*tmp3)*tmp1
         zinpl(i,3) = (.5*tmp2 -    tmp3)*tmp1
         zinpl(i,4) = (   tmp2 - .5*tmp3)*tmp1
         pinpl(i,1) = 0.5*(p1 + pnm(i,k2+1))
         pinpl(i,2) = 0.5*(p2 + pnm(i,k2  ))
         pinpl(i,3) = 0.5*(p1 + pnm(i,k2  ))
         pinpl(i,4) = 0.5*(p2 + pnm(i,k2+1))
         if(strat_volcanic) then
           aer_pth_ngh(i) = abs(aer_mpp(i,k2)-aer_mpp(i,k2+1))
         endif
      end do
      do kn=1,4
         do i=1,ncol
            u(i)     = uinpl(i,kn)*abs(plh2o(i,k2) - plh2o(i,k2+1))
            sqrtu(i) = sqrt(u(i))
            dw(i)    = abs(w(i,k2) - w(i,k2+1))
            pnew(i)  = u(i)/(winpl(i,kn)*dw(i))
            pnew_mks  = pnew(i) * sslp_mks
            t_p = min(max(tbar(i,kn), min_tp_h2o), max_tp_h2o)
            iest = floor(t_p) - min_tp_h2o
            esx = estblh2o(iest) + (estblh2o(iest+1)-estblh2o(iest)) * &
                 (t_p - min_tp_h2o - iest)
            qsx = epsilo * esx / (pnew_mks - omeps * esx)
            q_path = dw(i) / ABS(dpnm(i)) / rga
            
            ds2c     = abs(s2c(i,k2) - s2c(i,k2+1))
            uc1(i)   = uinpl(i,kn)*ds2c
            pch2o    = uc1(i)
            uc1(i)   = (uc1(i) + 1.7e-3*u(i))*(1. +  2.*uc1(i))/(1. + 15.*uc1(i))
            dtx(i)      = temh2o(i,kn) - 250.
            dty(i)      = tbar(i,kn) - 250.
            
            fwk(i)    = fwcoef + fwc1/(1. + fwc2*u(i))
            fwku(i)   = fwk(i)*u(i)

            if(strat_volcanic) then
              aer_pth_dlt=uinpl(i,kn)*aer_pth_ngh(i)
  
              do bnd_idx=1,bnd_nbr_LW
                 odap_aer_ttl=abs_cff_mss_aer(bnd_idx) * aer_pth_dlt 
                 aer_trn_ngh(i,bnd_idx)=exp(-fdif * odap_aer_ttl)
              end do
            else
              aer_trn_ngh(i,:) = 1.0
            endif

!
! Define variables for C/H/E (now C/LT/E) fit
!
! abso(i,1)     0 -  800 cm-1   h2o rotation band
! abso(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! abso(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O absorptivity
!
! Notation:
! U   = integral (P/P_0 dW)  
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
!
! Terms for asymptotic value of emissivity
!
            te1  = temh2o(i,kn)
            te2  = te1 * te1
            te3  = te2 * te1
            te4  = te3 * te1
            te5  = te4 * te1

!
! Indices for lines and continuum tables 
! Note: because we are dealing with the nearest layer,
!       the Hulst-Curtis-Godson corrections
!       for inhomogeneous paths are not applied.
!
            uvar = u(i)*fdif
            log_u  = min(log10(max(uvar, min_u_h2o)), max_lu_h2o)
            dvar = (log_u - min_lu_h2o) / dlu_h2o
            iu = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
            iu1 = iu + 1
            wu = dvar - floor(dvar)
            wu1 = 1.0 - wu
            
            log_p  = min(log10(max(pnew(i), min_p_h2o)), max_lp_h2o)
            dvar = (log_p - min_lp_h2o) / dlp_h2o
            ip = min(max(int(aint(dvar,r4)) + 1, 1), n_p - 1)
            ip1 = ip + 1
            wp = dvar - floor(dvar)
            wp1 = 1.0 - wp
            
            dvar = (t_p - min_tp_h2o) / dtp_h2o
            itp = min(max(int(aint(dvar,r4)) + 1, 1), n_tp - 1)
            itp1 = itp + 1
            wtp = dvar - floor(dvar)
            wtp1 = 1.0 - wtp
            
            t_e = min(max(temh2o(i,kn)-t_p,min_te_h2o),max_te_h2o)
            dvar = (t_e - min_te_h2o) / dte_h2o
            ite = min(max(int(aint(dvar,r4)) + 1, 1), n_te - 1)
            ite1 = ite + 1
            wte = dvar - floor(dvar)
            wte1 = 1.0 - wte
            
            rh_path = min(max(q_path / qsx, min_rh_h2o), max_rh_h2o)
            dvar = (rh_path - min_rh_h2o) / drh_h2o
            irh = min(max(int(aint(dvar,r4)) + 1, 1), n_rh - 1)
            irh1 = irh + 1
            wrh = dvar - floor(dvar)
            wrh1 = 1.0 - wrh
            
            w_0_0_ = wtp  * wte
            w_0_1_ = wtp  * wte1
            w_1_0_ = wtp1 * wte 
            w_1_1_ = wtp1 * wte1
            
            w_0_00 = w_0_0_ * wrh
            w_0_01 = w_0_0_ * wrh1
            w_0_10 = w_0_1_ * wrh
            w_0_11 = w_0_1_ * wrh1
            w_1_00 = w_1_0_ * wrh
            w_1_01 = w_1_0_ * wrh1
            w_1_10 = w_1_1_ * wrh
            w_1_11 = w_1_1_ * wrh1
            
            w00_00 = wp  * w_0_00 
            w00_01 = wp  * w_0_01 
            w00_10 = wp  * w_0_10 
            w00_11 = wp  * w_0_11 
            w01_00 = wp  * w_1_00 
            w01_01 = wp  * w_1_01 
            w01_10 = wp  * w_1_10 
            w01_11 = wp  * w_1_11 
            w10_00 = wp1 * w_0_00 
            w10_01 = wp1 * w_0_01 
            w10_10 = wp1 * w_0_10 
            w10_11 = wp1 * w_0_11 
            w11_00 = wp1 * w_1_00 
            w11_01 = wp1 * w_1_01 
            w11_10 = wp1 * w_1_10 
            w11_11 = wp1 * w_1_11 

!
! Non-window absorptivity
!
            ib = 1
            
            fa = fat(1,ib) + &
                 fat(2,ib) * te1 + &
                 fat(3,ib) * te2 + &
                 fat(4,ib) * te3 + &
                 fat(5,ib) * te4 + &
                 fat(6,ib) * te5
            
            a_star = &
                 ah2onw(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
                 ah2onw(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
                 ah2onw(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
                 ah2onw(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
                 ah2onw(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
                 ah2onw(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
                 ah2onw(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
                 ah2onw(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
                 ah2onw(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
                 ah2onw(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
                 ah2onw(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
                 ah2onw(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
                 ah2onw(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
                 ah2onw(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
                 ah2onw(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
                 ah2onw(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
                 ah2onw(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
                 ah2onw(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
                 ah2onw(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
                 ah2onw(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
                 ah2onw(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
                 ah2onw(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
                 ah2onw(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
                 ah2onw(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
                 ah2onw(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
                 ah2onw(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
                 ah2onw(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
                 ah2onw(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu
            
            abso(i,ib) = min(max(fa * (1.0 - (1.0 - a_star) * &
                                 aer_trn_ngh(i,ib)), &
                             0.0), 1.0)

!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
            if (uvar < min_u_h2o) then
               uscl = uvar / min_u_h2o
               abso(i,ib) = abso(i,ib) * uscl
            endif
            
!
! Window absorptivity
!
            ib = 2
            
            fa = fat(1,ib) + &
                 fat(2,ib) * te1 + &
                 fat(3,ib) * te2 + &
                 fat(4,ib) * te3 + &
                 fat(5,ib) * te4 + &
                 fat(6,ib) * te5
            
            a_star = &
                 ah2ow(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
                 ah2ow(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
                 ah2ow(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
                 ah2ow(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
                 ah2ow(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
                 ah2ow(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
                 ah2ow(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
                 ah2ow(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
                 ah2ow(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
                 ah2ow(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
                 ah2ow(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
                 ah2ow(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
                 ah2ow(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
                 ah2ow(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
                 ah2ow(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
                 ah2ow(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
                 ah2ow(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
                 ah2ow(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
                 ah2ow(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
                 ah2ow(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
                 ah2ow(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
                 ah2ow(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
                 ah2ow(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
                 ah2ow(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
                 ah2ow(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
                 ah2ow(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
                 ah2ow(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
                 ah2ow(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
                 ah2ow(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
                 ah2ow(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
                 ah2ow(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
                 ah2ow(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu
            
            abso(i,ib) = min(max(fa * (1.0 - (1.0 - a_star) * &
                                 aer_trn_ngh(i,ib)), &
                             0.0), 1.0)

!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
            if (uvar < min_u_h2o) then
               uscl = uvar / min_u_h2o
               abso(i,ib) = abso(i,ib) * uscl
            endif
            
         end do
!
! Line transmission in 800-1000 and 1000-1200 cm-1 intervals
!
         do i=1,ncol
            term7(i,1) = coefj(1,1) + coefj(2,1)*dty(i)*(1. + c16*dty(i))
            term8(i,1) = coefk(1,1) + coefk(2,1)*dty(i)*(1. + c17*dty(i))
            term7(i,2) = coefj(1,2) + coefj(2,2)*dty(i)*(1. + c26*dty(i))
            term8(i,2) = coefk(1,2) + coefk(2,2)*dty(i)*(1. + c27*dty(i))
         end do
!
! 500 -  800 cm-1   h2o rotation band overlap with co2
!
         do i=1,ncol
            dtym10     = dty(i) - 10.
            denom      = 1. + (c30 + c31*dtym10*dtym10)*sqrtu(i)
            k21        = term7(i,1) + term8(i,1)/denom
            denom      = 1. + (c28 + c29*dtym10       )*sqrtu(i)
            k22        = term7(i,2) + term8(i,2)/denom
            tr1     = exp(-(k21*(sqrtu(i) + fc1*fwku(i))))
            tr2     = exp(-(k22*(sqrtu(i) + fc1*fwku(i))))
            tr1=tr1*aer_trn_ngh(i,idx_LW_0650_0800) 
!                                         ! H2O line+STRAER trn 650--800 cm-1
            tr2=tr2*aer_trn_ngh(i,idx_LW_0500_0650) 
!                                         ! H2O line+STRAER trn 500--650 cm-1
            tr5     = exp(-((coefh(1,3) + coefh(2,3)*dtx(i))*uc1(i)))
            tr6     = exp(-((coefh(1,4) + coefh(2,4)*dtx(i))*uc1(i)))
            tr9(i)  = tr1*tr5
            tr10(i) = tr2*tr6
            trab2(i)= 0.65*tr9(i) + 0.35*tr10(i)
            th2o(i) = tr10(i)
         end do
!
! abso(i,3)  o3  9.6 micrometer (nu3 and nu1 bands)
!
         do i=1,ncol
            te        = (tbar(i,kn)*r293)**.7
            dplos     = abs(plos(i,k2+1) - plos(i,k2))
            u1        = zinpl(i,kn)*18.29*dplos/te
            u2        = zinpl(i,kn)*.5649*dplos/te
            tlocal    = tbar(i,kn)
            tcrfac    = sqrt(tlocal*r250)*te
            beta      = r3205*(pinpl(i,kn)*rsslp + dpfo3*tcrfac)
            realnu    = te/beta
            tmp1      = u1/sqrt(4. + u1*(1. + realnu))
            tmp2      = u2/sqrt(4. + u2*(1. + realnu))
            o3bndi    = 74.*te*log(1. + tmp1 + tmp2)
            abso(i,3) = o3bndi*o3emm(i,kn)*(h2otr(i,k2+1)/h2otr(i,k2))
            to3(i)    = 1.0/(1. + 0.1*tmp1 + 0.1*tmp2)
         end do
!
! abso(i,4)   co2 15  micrometer band system
!
         do i=1,ncol
            dplco2   = plco2(i,k2+1) - plco2(i,k2)
            sqwp     = sqrt(uinpl(i,kn)*dplco2)
            et       = exp(-480./tbar(i,kn))
            sqti(i)  = sqrt(tbar(i,kn))
            rsqti    = 1./sqti(i)
            et2      = et*et
            et4      = et2*et2
            omet     = (1. - 1.5*et2)
            f1co2    = 899.70*omet*(1. + 1.94774*et + 4.73486*et2)*rsqti
            f1sqwp(i)= f1co2*sqwp
            t1co2(i) = 1./(1. + (245.18*omet*sqwp*rsqti))
            oneme    = 1. - et2
            alphat   = oneme**3*rsqti
            pi       = abs(dpnm(i))*winpl(i,kn)
            wco2     = 2.5221*co2vmr*pi*rga
            u7(i)    = 4.9411e4*alphat*et2*wco2
            u8       = 3.9744e4*alphat*et4*wco2
            u9       = 1.0447e5*alphat*et4*et2*wco2
            u13      = 2.8388e3*alphat*et4*wco2
            tpath    = tbar(i,kn)
            tlocal   = tbar(i,kn)
            tcrfac   = sqrt((tlocal*r250)*(tpath*r300))
            posqt    = (pinpl(i,kn)*rsslp + dpfco2*tcrfac)*rsqti
            rbeta7(i)= 1./(5.3228*posqt)
            rbeta8   = 1./(10.6576*posqt)
            rbeta9   = rbeta7(i)
            rbeta13  = rbeta9
            f2co2(i) = u7(i)/sqrt(4. + u7(i)*(1. + rbeta7(i))) + &
                 u8   /sqrt(4. + u8*(1. + rbeta8)) + &
                 u9   /sqrt(4. + u9*(1. + rbeta9))
            f3co2(i) = u13/sqrt(4. + u13*(1. + rbeta13))
            tmp1     = log(1. + f1sqwp(i))
            tmp2     = log(1. + f2co2(i))
            tmp3     = log(1. + f3co2(i))
            absbnd   = (tmp1 + 2.*t1co2(i)*tmp2 + 2.*tmp3)*sqti(i)
            abso(i,4)= trab2(i)*emm(i,kn)*absbnd
            tco2(i)  = 1.0/(1.0+ 10.0*u7(i)/sqrt(4. + u7(i)*(1. + rbeta7(i))))
         end do ! do i =
!
! Calculate trace gas absorptivity for nearest layer, abstrc
!
         call trcabn(lchnk   ,ncol    ,                   &
              k2      ,kn      ,ucfc11  ,ucfc12  ,un2o0   , &
              un2o1   ,uch4    ,uco211  ,uco212  ,uco213  , &
              uco221  ,uco222  ,uco223  ,tbar    ,bplnk   , &
              winpl   ,pinpl   ,tco2    ,th2o    ,to3     , &
              uptype  ,dw      ,s2c     ,u       ,pnew    , &
              abstrc  ,uinpl   , &
              aer_trn_ngh)
!
! Total next layer absorptivity:
!
         do i=1,ncol
            absnxt(i,k2,kn) = abso(i,1) + abso(i,2) + &
                 abso(i,3) + abso(i,4) + abstrc(i)
         end do
      end do ! do kn =
   end do ! do k2 =

   return
end subroutine radabs

subroutine radems(lchnk   ,ncol    ,                            &
                  s2c     ,tcg     ,w       ,tplnke  ,plh2o   , &
                  pnm     ,plco2   ,tint    ,tint4   ,tlayr   , &
                  tlayr4  ,plol    ,plos    ,ucfc11  ,ucfc12  , &
                  un2o0   ,un2o1   ,uch4    ,uco211 ,uco212   , &
                  uco213  ,uco221  ,uco222  ,uco223  ,uptype  , &
                  bn2o0   ,bn2o1   ,bch4    ,co2em   ,co2eml  , &
                  co2t    ,h2otr   ,abplnk1 ,abplnk2 ,emstot  , &
                  plh2ob  ,wb      , &
                  aer_trn_ttl)
!----------------------------------------------------------------------- 
! 
! Purpose: 
! Compute emissivity for H2O, CO2, O3, CH4, N2O, CFC11 and CFC12
! 
! Method: 
! H2O  ....  Uses nonisothermal emissivity method for water vapor from
!            Ramanathan, V. and  P.Downey, 1986: A Nonisothermal
!            Emissivity and Absorptivity Formulation for Water Vapor
!            Jouranl of Geophysical Research, vol. 91., D8, pp 8649-8666
!
!            Implementation updated by Collins,Hackney, and Edwards 2001
!               using line-by-line calculations based upon Hitran 1996 and
!               CKD 2.1 for absorptivity and emissivity
!
!            Implementation updated by Collins, Lee-Taylor, and Edwards (2003)
!               using line-by-line calculations based upon Hitran 2000 and
!               CKD 2.4 for absorptivity and emissivity
!
! CO2  ....  Uses absorptance parameterization of the 15 micro-meter
!            (500 - 800 cm-1) band system of Carbon Dioxide, from
!            Kiehl, J.T. and B.P.Briegleb, 1991: A New Parameterization
!            of the Absorptance Due to the 15 micro-meter Band System
!            of Carbon Dioxide Jouranl of Geophysical Research,
!            vol. 96., D5, pp 9013-9019. Also includes the effects
!            of the 9.4 and 10.4 micron bands of CO2.
!
! O3   ....  Uses absorptance parameterization of the 9.6 micro-meter
!            band system of ozone, from Ramanathan, V. and R. Dickinson,
!            1979: The Role of stratospheric ozone in the zonal and
!            seasonal radiative energy balance of the earth-troposphere
!            system. Journal of the Atmospheric Sciences, Vol. 36,
!            pp 1084-1104
!
! ch4  ....  Uses a broad band model for the 7.7 micron band of methane.
!
! n20  ....  Uses a broad band model for the 7.8, 8.6 and 17.0 micron
!            bands of nitrous oxide
!
! cfc11 ...  Uses a quasi-linear model for the 9.2, 10.7, 11.8 and 12.5
!            micron bands of CFC11
!
! cfc12 ...  Uses a quasi-linear model for the 8.6, 9.1, 10.8 and 11.2
!            micron bands of CFC12
!
!
! Computes individual emissivities, accounting for band overlap, and
! sums to obtain the total.
!
! Author: W. Collins (H2O emissivity) and J. Kiehl
! 
!-----------------------------------------------------------------------
!bloss #include <crdcon.h>
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                    ! chunk identifier
   integer, intent(in) :: ncol                     ! number of atmospheric columns

   real(r4), intent(in) :: s2c(pcols,pverp)        ! H2o continuum path length
   real(r4), intent(in) :: tcg(pcols,pverp)        ! H2o-mass-wgted temp. (Curtis-Godson approx.)
   real(r4), intent(in) :: w(pcols,pverp)          ! H2o path length
   real(r4), intent(in) :: tplnke(pcols)           ! Layer planck temperature
   real(r4), intent(in) :: plh2o(pcols,pverp)      ! H2o prs wghted path length
   real(r4), intent(in) :: pnm(pcols,pverp)        ! Model interface pressure
   real(r4), intent(in) :: plco2(pcols,pverp)      ! Prs wghted path of co2
   real(r4), intent(in) :: tint(pcols,pverp)       ! Model interface temperatures
   real(r4), intent(in) :: tint4(pcols,pverp)      ! Tint to the 4th power
   real(r4), intent(in) :: tlayr(pcols,pverp)      ! K-1 model layer temperature
   real(r4), intent(in) :: tlayr4(pcols,pverp)     ! Tlayr to the 4th power
   real(r4), intent(in) :: plol(pcols,pverp)       ! Pressure wghtd ozone path
   real(r4), intent(in) :: plos(pcols,pverp)       ! Ozone path
   real(r4), intent(in) :: plh2ob(nbands,pcols,pverp) ! Pressure weighted h2o path with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 
   real(r4), intent(in) :: wb(nbands,pcols,pverp)     ! H2o path length with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 

   real(r4), intent(in) :: aer_trn_ttl(pcols,pverp,pverp,bnd_nbr_LW) 
!                               ! [fraction] Total strat. aerosol
!                               ! transmission between interfaces k1 and k2  

!
! Trace gas variables
!
   real(r4), intent(in) :: ucfc11(pcols,pverp)     ! CFC11 path length
   real(r4), intent(in) :: ucfc12(pcols,pverp)     ! CFC12 path length
   real(r4), intent(in) :: un2o0(pcols,pverp)      ! N2O path length
   real(r4), intent(in) :: un2o1(pcols,pverp)      ! N2O path length (hot band)
   real(r4), intent(in) :: uch4(pcols,pverp)       ! CH4 path length
   real(r4), intent(in) :: uco211(pcols,pverp)     ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco212(pcols,pverp)     ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco213(pcols,pverp)     ! CO2 9.4 micron band path length
   real(r4), intent(in) :: uco221(pcols,pverp)     ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco222(pcols,pverp)     ! CO2 10.4 micron band path length
   real(r4), intent(in) :: uco223(pcols,pverp)     ! CO2 10.4 micron band path length
   real(r4), intent(in) :: bn2o0(pcols,pverp)      ! pressure factor for n2o
   real(r4), intent(in) :: bn2o1(pcols,pverp)      ! pressure factor for n2o
   real(r4), intent(in) :: bch4(pcols,pverp)       ! pressure factor for ch4
   real(r4), intent(in) :: uptype(pcols,pverp)     ! p-type continuum path length
!
! Output arguments
!
   real(r4), intent(out) :: emstot(pcols,pverp)     ! Total emissivity
   real(r4), intent(out) :: co2em(pcols,pverp)      ! Layer co2 normalzd plnck funct drvtv
   real(r4), intent(out) :: co2eml(pcols,pver)      ! Intrfc co2 normalzd plnck func drvtv
   real(r4), intent(out) :: co2t(pcols,pverp)       ! Tmp and prs weighted path length
   real(r4), intent(out) :: h2otr(pcols,pverp)      ! H2o transmission over o3 band
   real(r4), intent(out) :: abplnk1(14,pcols,pverp) ! non-nearest layer Plack factor
   real(r4), intent(out) :: abplnk2(14,pcols,pverp) ! nearest layer factor

!
!---------------------------Local variables-----------------------------
!
   integer i                    ! Longitude index
   integer k                    ! Level index]
   integer k1                   ! Level index
!
! Local variables for H2O:
!
   real(r4) h2oems(pcols,pverp)     ! H2o emissivity
   real(r4) tpathe                  ! Used to compute h2o emissivity
   real(r4) dtx(pcols)              ! Planck temperature minus 250 K
   real(r4) dty(pcols)              ! Path temperature minus 250 K
!
! The 500-800 cm^-1 emission in emis(i,4) has been combined
!              into the 0-800 cm^-1 emission in emis(i,1)
!
   real(r4) emis(pcols,2)           ! H2O emissivity 
!
!
!
   real(r4) term7(pcols,2)          ! Kl_inf(i) in eq(r4) of table A3a of R&D
   real(r4) term8(pcols,2)          ! Delta kl_inf(i) in eq(r4)
   real(r4) tr1(pcols)              ! Equation(6) in table A2 for 650-800
   real(r4) tr2(pcols)              ! Equation(6) in table A2 for 500-650
   real(r4) tr3(pcols)              ! Equation(4) in table A2 for 650-800
   real(r4) tr4(pcols)              ! Equation(4),table A2 of R&D for 500-650
   real(r4) tr7(pcols)              ! Equation (6) times eq(4) in table A2
!                                      of R&D for 650-800 cm-1 region
   real(r4) tr8(pcols)              ! Equation (6) times eq(4) in table A2
!                                      of R&D for 500-650 cm-1 region
   real(r4) k21(pcols)              ! Exponential coefficient used to calc
!                                     rot band transmissivity in the 650-800
!                                     cm-1 region (tr1)
   real(r4) k22(pcols)              ! Exponential coefficient used to calc
!                                     rot band transmissivity in the 500-650
!                                     cm-1 region (tr2)
   real(r4) u(pcols)                ! Pressure weighted H2O path length
   real(r4) ub(nbands)              ! Pressure weighted H2O path length with
                                    !  Hulst-Curtis-Godson correction for
                                    !  each band
   real(r4) pnew                    ! Effective pressure for h2o linewidth
   real(r4) pnewb(nbands)           ! Effective pressure for h2o linewidth w/
                                    !  Hulst-Curtis-Godson correction for
                                    !  each band
   real(r4) uc1(pcols)              ! H2o continuum pathlength 500-800 cm-1
   real(r4) fwk                     ! Equation(33) in R&D far wing correction
   real(r4) troco2(pcols,pverp)     ! H2o overlap factor for co2 absorption
   real(r4) emplnk(14,pcols)        ! emissivity Planck factor
   real(r4) emstrc(pcols,pverp)     ! total trace gas emissivity
!
! Local variables for CO2:
!
   real(r4) co2ems(pcols,pverp)      ! Co2 emissivity
   real(r4) co2plk(pcols)            ! Used to compute co2 emissivity
   real(r4) sum(pcols)               ! Used to calculate path temperature
   real(r4) t1i                      ! Co2 hot band temperature factor
   real(r4) sqti                     ! Sqrt of temperature
   real(r4) pi                       ! Pressure used in co2 mean line width
   real(r4) et                       ! Co2 hot band factor
   real(r4) et2                      ! Co2 hot band factor
   real(r4) et4                      ! Co2 hot band factor
   real(r4) omet                     ! Co2 stimulated emission term
   real(r4) ex                       ! Part of co2 planck function
   real(r4) f1co2                    ! Co2 weak band factor
   real(r4) f2co2                    ! Co2 weak band factor
   real(r4) f3co2                    ! Co2 weak band factor
   real(r4) t1co2                    ! Overlap factor weak bands strong band
   real(r4) sqwp                     ! Sqrt of co2 pathlength
   real(r4) f1sqwp                   ! Main co2 band factor
   real(r4) oneme                    ! Co2 stimulated emission term
   real(r4) alphat                   ! Part of the co2 stimulated emiss term
   real(r4) wco2                     ! Consts used to define co2 pathlength
   real(r4) posqt                    ! Effective pressure for co2 line width
   real(r4) rbeta7                   ! Inverse of co2 hot band line width par
   real(r4) rbeta8                   ! Inverse of co2 hot band line width par
   real(r4) rbeta9                   ! Inverse of co2 hot band line width par
   real(r4) rbeta13                  ! Inverse of co2 hot band line width par
   real(r4) tpath                    ! Path temp used in co2 band model
   real(r4) tmp1                     ! Co2 band factor
   real(r4) tmp2                     ! Co2 band factor
   real(r4) tmp3                     ! Co2 band factor
   real(r4) tlayr5                   ! Temperature factor in co2 Planck func
   real(r4) rsqti                    ! Reciprocal of sqrt of temperature
   real(r4) exm1sq                   ! Part of co2 Planck function
   real(r4) u7                       ! Absorber amt for various co2 band systems
   real(r4) u8                       ! Absorber amt for various co2 band systems
   real(r4) u9                       ! Absorber amt for various co2 band systems
   real(r4) u13                      ! Absorber amt for various co2 band systems
   real(r4) r250                     ! Inverse 250K
   real(r4) r300                     ! Inverse 300K
   real(r4) rsslp                    ! Inverse standard sea-level pressure
!
! Local variables for O3:
!
   real(r4) o3ems(pcols,pverp)       ! Ozone emissivity
   real(r4) dbvtt(pcols)             ! Tmp drvtv of planck fctn for tplnke
   real(r4) dbvt,fo3,t,ux,vx
   real(r4) te                       ! Temperature factor
   real(r4) u1                       ! Path length factor
   real(r4) u2                       ! Path length factor
   real(r4) phat                     ! Effecitive path length pressure
   real(r4) tlocal                   ! Local planck function temperature
   real(r4) tcrfac                   ! Scaled temperature factor
   real(r4) beta                     ! Absorption funct factor voigt effect
   real(r4) realnu                   ! Absorption function factor
   real(r4) o3bndi                   ! Band absorption factor
!
! Transmission terms for various spectral intervals:
!
   real(r4) absbnd                   ! Proportional to co2 band absorptance
   real(r4) tco2(pcols)              ! co2 overlap factor
   real(r4) th2o(pcols)              ! h2o overlap factor
   real(r4) to3(pcols)               ! o3 overlap factor
!
! Variables for new H2O parameterization
!
! Notation:
! U   = integral (P/P_0 dW)  eq. 15 in Ramanathan/Downey 1986
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
   real(r4) fe               ! asymptotic value of emis. as U->infinity
   real(r4) e_star           ! normalized non-window emissivity
   real(r4) l_star           ! interpolated line transmission
   real(r4) c_star           ! interpolated continuum transmission

   real(r4) te1              ! emission temperature
   real(r4) te2              ! te^2
   real(r4) te3              ! te^3
   real(r4) te4              ! te^4
   real(r4) te5              ! te^5

   real(r4) log_u            ! log base 10 of U 
   real(r4) log_uc           ! log base 10 of H2O continuum path
   real(r4) log_p            ! log base 10 of P
   real(r4) t_p              ! T_p
   real(r4) t_e              ! T_e (offset by T_p)

   integer iu                ! index for log10(U)
   integer iu1               ! iu + 1
   integer iuc               ! index for log10(H2O continuum path)
   integer iuc1              ! iuc + 1
   integer ip                ! index for log10(P)
   integer ip1               ! ip + 1
   integer itp               ! index for T_p
   integer itp1              ! itp + 1
   integer ite               ! index for T_e
   integer ite1              ! ite + 1
   integer irh               ! index for RH
   integer irh1              ! irh + 1

   real(r4) dvar             ! normalized variation in T_p/T_e/P/U
   real(r4) uvar             ! U * diffusivity factor
   real(r4) uscl             ! factor for lineary scaling as U->0

   real(r4) wu               ! weight for U
   real(r4) wu1              ! 1 - wu
   real(r4) wuc              ! weight for H2O continuum path
   real(r4) wuc1             ! 1 - wuc
   real(r4) wp               ! weight for P
   real(r4) wp1              ! 1 - wp
   real(r4) wtp              ! weight for T_p
   real(r4) wtp1             ! 1 - wtp
   real(r4) wte              ! weight for T_e
   real(r4) wte1             ! 1 - wte
   real(r4) wrh              ! weight for RH
   real(r4) wrh1             ! 1 - wrh

   real(r4) w_0_0_           ! weight for Tp/Te combination
   real(r4) w_0_1_           ! weight for Tp/Te combination
   real(r4) w_1_0_           ! weight for Tp/Te combination
   real(r4) w_1_1_           ! weight for Tp/Te combination

   real(r4) w_0_00           ! weight for Tp/Te/RH combination
   real(r4) w_0_01           ! weight for Tp/Te/RH combination
   real(r4) w_0_10           ! weight for Tp/Te/RH combination
   real(r4) w_0_11           ! weight for Tp/Te/RH combination
   real(r4) w_1_00           ! weight for Tp/Te/RH combination
   real(r4) w_1_01           ! weight for Tp/Te/RH combination
   real(r4) w_1_10           ! weight for Tp/Te/RH combination
   real(r4) w_1_11           ! weight for Tp/Te/RH combination

   real(r4) w00_00           ! weight for P/Tp/Te/RH combination
   real(r4) w00_01           ! weight for P/Tp/Te/RH combination
   real(r4) w00_10           ! weight for P/Tp/Te/RH combination
   real(r4) w00_11           ! weight for P/Tp/Te/RH combination
   real(r4) w01_00           ! weight for P/Tp/Te/RH combination
   real(r4) w01_01           ! weight for P/Tp/Te/RH combination
   real(r4) w01_10           ! weight for P/Tp/Te/RH combination
   real(r4) w01_11           ! weight for P/Tp/Te/RH combination
   real(r4) w10_00           ! weight for P/Tp/Te/RH combination
   real(r4) w10_01           ! weight for P/Tp/Te/RH combination
   real(r4) w10_10           ! weight for P/Tp/Te/RH combination
   real(r4) w10_11           ! weight for P/Tp/Te/RH combination
   real(r4) w11_00           ! weight for P/Tp/Te/RH combination
   real(r4) w11_01           ! weight for P/Tp/Te/RH combination
   real(r4) w11_10           ! weight for P/Tp/Te/RH combination
   real(r4) w11_11           ! weight for P/Tp/Te/RH combination

   integer ib                ! spectral interval:
                             !   1 = 0-800 cm^-1 and 1200-2200 cm^-1
                             !   2 = 800-1200 cm^-1

   real(r4) pch2o            ! H2O continuum path
   real(r4) fch2o            ! temp. factor for continuum
   real(r4) uch2o            ! U corresponding to H2O cont. path (window)

   real(r4) fdif             ! secant(zenith angle) for diffusivity approx.

   real(r4) sslp_mks         ! Sea-level pressure in MKS units
   real(r4) esx              ! saturation vapor pressure returned by vqsatd
   real(r4) qsx              ! saturation mixing ratio returned by vqsatd
   real(r4) pnew_mks         ! pnew in MKS units
   real(r4) q_path           ! effective specific humidity along path
   real(r4) rh_path          ! effective relative humidity along path
   real(r4) omeps            ! 1 - epsilo

   integer  iest             ! index in estblh2o

!
!---------------------------Statement functions-------------------------
!
! Derivative of planck function at 9.6 micro-meter wavelength, and
! an absorption function factor:
!
!
   dbvt(t)=(-2.8911366682e-4+(2.3771251896e-6+1.1305188929e-10*t)*t)/ &
           (1.0+(-6.1364820707e-3+1.5550319767e-5*t)*t)
!
   fo3(ux,vx)=ux/sqrt(4.+ux*(1.+vx))
!
!
!
!-----------------------------------------------------------------------
!
! Initialize
!
   r250  = 1./250.
   r300  = 1./300.
   rsslp = 1./sslp
!
! Constants for computing U corresponding to H2O cont. path
!
   fdif       = 1.66
   sslp_mks   = sslp / 10.0
   omeps      = 1.0 - epsilo
!
! Planck function for co2
!
   do i=1,ncol
      ex             = exp(960./tplnke(i))
      co2plk(i)      = 5.e8/((tplnke(i)**4)*(ex - 1.))
      co2t(i,ntoplw) = tplnke(i)
      sum(i)         = co2t(i,ntoplw)*pnm(i,ntoplw)
   end do
   k = ntoplw
   do k1=pverp,ntoplw+1,-1
      k = k + 1
      do i=1,ncol
         sum(i)         = sum(i) + tlayr(i,k)*(pnm(i,k)-pnm(i,k-1))
         ex             = exp(960./tlayr(i,k1))
         tlayr5         = tlayr(i,k1)*tlayr4(i,k1)
         co2eml(i,k1-1) = 1.2e11*ex/(tlayr5*(ex - 1.)**2)
         co2t(i,k)      = sum(i)/pnm(i,k)
      end do
   end do
!
! Initialize planck function derivative for O3
!
   do i=1,ncol
      dbvtt(i) = dbvt(tplnke(i))
   end do
!
! Calculate trace gas Planck functions
!
   call trcplk(lchnk   ,ncol    ,                            &
               tint    ,tlayr   ,tplnke  ,emplnk  ,abplnk1 , &
               abplnk2 )
!
! Interface loop
!
   do k1=ntoplw,pverp
!
! H2O emissivity
!
! emis(i,1)     0 -  800 cm-1   h2o rotation band
! emis(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! emis(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O emissivity
!
!      emis(i,3)   = 0.0
!
! For the p type continuum
!
      do i=1,ncol
         u(i)        = plh2o(i,k1)
         pnew        = u(i)/w(i,k1)
         pnew_mks    = pnew * sslp_mks
!
! Apply scaling factor for 500-800 continuum
!
         uc1(i)      = (s2c(i,k1) + 1.7e-3*plh2o(i,k1))*(1. + 2.*s2c(i,k1))/ &
                       (1. + 15.*s2c(i,k1))
         pch2o       = s2c(i,k1)
!
! Changed effective path temperature to std. Curtis-Godson form
!
         tpathe   = tcg(i,k1)/w(i,k1)
         t_p = min(max(tpathe, min_tp_h2o), max_tp_h2o)
         iest = floor(t_p) - min_tp_h2o
         esx = estblh2o(iest) + (estblh2o(iest+1)-estblh2o(iest)) * &
               (t_p - min_tp_h2o - iest)
         qsx = epsilo * esx / (pnew_mks - omeps * esx)
!
! Compute effective RH along path
!
         q_path = w(i,k1) / pnm(i,k1) / rga
!
! Calculate effective u, pnew for each band using
!        Hulst-Curtis-Godson approximation:
! Formulae: Goody and Yung, Atmospheric Radiation: Theoretical Basis, 
!           2nd edition, Oxford University Press, 1989.
! Effective H2O path (w)
!      eq. 6.24, p. 228
! Effective H2O path pressure (pnew = u/w):
!      eq. 6.29, p. 228
!
         ub(1) = plh2ob(1,i,k1) / psi(t_p,1)
         ub(2) = plh2ob(2,i,k1) / psi(t_p,2)

         pnewb(1) = ub(1) / wb(1,i,k1) * phi(t_p,1)
         pnewb(2) = ub(2) / wb(2,i,k1) * phi(t_p,2)
!
!
!
         dtx(i) = tplnke(i) - 250.
         dty(i) = tpathe - 250.
!
! Define variables for C/H/E (now C/LT/E) fit
!
! emis(i,1)     0 -  800 cm-1   h2o rotation band
! emis(i,1)  1200 - 2200 cm-1   h2o vibration-rotation band
! emis(i,2)   800 - 1200 cm-1   h2o window
!
! Separation between rotation and vibration-rotation dropped, so
!                only 2 slots needed for H2O emissivity
!
! emis(i,3)   = 0.0
!
! Notation:
! U   = integral (P/P_0 dW)  
! P   = atmospheric pressure
! P_0 = reference atmospheric pressure
! W   = precipitable water path
! T_e = emission temperature
! T_p = path temperature
! RH  = path relative humidity
!
! Terms for asymptotic value of emissivity
!
         te1  = tplnke(i)
         te2  = te1 * te1
         te3  = te2 * te1
         te4  = te3 * te1
         te5  = te4 * te1
!
! Band-independent indices for lines and continuum tables
!
         dvar = (t_p - min_tp_h2o) / dtp_h2o
         itp = min(max(int(aint(dvar,r4)) + 1, 1), n_tp - 1)
         itp1 = itp + 1
         wtp = dvar - floor(dvar)
         wtp1 = 1.0 - wtp

         t_e = min(max(tplnke(i) - t_p, min_te_h2o), max_te_h2o)
         dvar = (t_e - min_te_h2o) / dte_h2o
         ite = min(max(int(aint(dvar,r4)) + 1, 1), n_te - 1)
         ite1 = ite + 1
         wte = dvar - floor(dvar)
         wte1 = 1.0 - wte

         rh_path = min(max(q_path / qsx, min_rh_h2o), max_rh_h2o)
         dvar = (rh_path - min_rh_h2o) / drh_h2o
         irh = min(max(int(aint(dvar,r4)) + 1, 1), n_rh - 1)
         irh1 = irh + 1
         wrh = dvar - floor(dvar)
         wrh1 = 1.0 - wrh

         w_0_0_ = wtp  * wte
         w_0_1_ = wtp  * wte1
         w_1_0_ = wtp1 * wte 
         w_1_1_ = wtp1 * wte1

         w_0_00 = w_0_0_ * wrh
         w_0_01 = w_0_0_ * wrh1
         w_0_10 = w_0_1_ * wrh
         w_0_11 = w_0_1_ * wrh1
         w_1_00 = w_1_0_ * wrh
         w_1_01 = w_1_0_ * wrh1
         w_1_10 = w_1_1_ * wrh
         w_1_11 = w_1_1_ * wrh1
!
! H2O Continuum path for 0-800 and 1200-2200 cm^-1
!
!    Assume foreign continuum dominates total H2O continuum in these bands
!    per Clough et al, JGR, v. 97, no. D14 (Oct 20, 1992), p. 15776
!    Then the effective H2O path is just 
!         U_c = integral[ f(P) dW ]
!    where 
!           W = water-vapor mass and 
!        f(P) = dependence of foreign continuum on pressure 
!             = P / sslp
!    Then 
!         U_c = U (the same effective H2O path as for lines)
!
!
! Continuum terms for 800-1200 cm^-1
!
!    Assume self continuum dominates total H2O continuum for this band
!    per Clough et al, JGR, v. 97, no. D14 (Oct 20, 1992), p. 15776
!    Then the effective H2O self-continuum path is 
!         U_c = integral[ h(e,T) dW ]                        (*eq. 1*)
!    where 
!           W = water-vapor mass and 
!           e = partial pressure of H2O along path
!           T = temperature along path
!      h(e,T) = dependence of foreign continuum on e,T
!             = e / sslp * f(T)
!
!    Replacing
!           e =~ q * P / epsilo
!           q = mixing ratio of H2O
!     epsilo = 0.622
!
!    and using the definition
!           U = integral [ (P / sslp) dW ]
!             = (P / sslp) W                                 (homogeneous path)
!
!    the effective path length for the self continuum is
!         U_c = (q / epsilo) f(T) U                         (*eq. 2*)
!
!    Once values of T, U, and q have been calculated for the inhomogeneous
!        path, this sets U_c for the corresponding
!        homogeneous atmosphere.  However, this need not equal the
!        value of U_c' defined by eq. 1 for the actual inhomogeneous atmosphere
!        under consideration.
!
!    Solution: hold T and q constant, solve for U' that gives U_c' by
!        inverting eq. (2):
!
!        U' = (U_c * epsilo) / (q * f(T))
!
         fch2o = fh2oself(t_p)
         uch2o = (pch2o * epsilo) / (q_path * fch2o)

!
! Band-dependent indices for non-window
!
         ib = 1

         uvar = ub(ib) * fdif
         log_u  = min(log10(max(uvar, min_u_h2o)), max_lu_h2o)
         dvar = (log_u - min_lu_h2o) / dlu_h2o
         iu = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
         iu1 = iu + 1
         wu = dvar - floor(dvar)
         wu1 = 1.0 - wu
         
         log_p  = min(log10(max(pnewb(ib), min_p_h2o)), max_lp_h2o)
         dvar = (log_p - min_lp_h2o) / dlp_h2o
         ip = min(max(int(aint(dvar,r4)) + 1, 1), n_p - 1)
         ip1 = ip + 1
         wp = dvar - floor(dvar)
         wp1 = 1.0 - wp

         w00_00 = wp  * w_0_00 
         w00_01 = wp  * w_0_01 
         w00_10 = wp  * w_0_10 
         w00_11 = wp  * w_0_11 
         w01_00 = wp  * w_1_00 
         w01_01 = wp  * w_1_01 
         w01_10 = wp  * w_1_10 
         w01_11 = wp  * w_1_11 
         w10_00 = wp1 * w_0_00 
         w10_01 = wp1 * w_0_01 
         w10_10 = wp1 * w_0_10 
         w10_11 = wp1 * w_0_11 
         w11_00 = wp1 * w_1_00 
         w11_01 = wp1 * w_1_01 
         w11_10 = wp1 * w_1_10 
         w11_11 = wp1 * w_1_11 

!
! Asymptotic value of emissivity as U->infinity
!
         fe = fet(1,ib) + &
              fet(2,ib) * te1 + &
              fet(3,ib) * te2 + &
              fet(4,ib) * te3 + &
              fet(5,ib) * te4 + &
              fet(6,ib) * te5

         e_star = &
              eh2onw(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
              eh2onw(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
              eh2onw(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
              eh2onw(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
              eh2onw(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
              eh2onw(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
              eh2onw(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
              eh2onw(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
              eh2onw(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
              eh2onw(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
              eh2onw(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
              eh2onw(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
              eh2onw(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
              eh2onw(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
              eh2onw(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
              eh2onw(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
              eh2onw(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
              eh2onw(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
              eh2onw(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
              eh2onw(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
              eh2onw(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
              eh2onw(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
              eh2onw(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
              eh2onw(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
              eh2onw(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
              eh2onw(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
              eh2onw(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
              eh2onw(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
              eh2onw(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
              eh2onw(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
              eh2onw(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
              eh2onw(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu 
         emis(i,ib) = min(max(fe * (1.0 - (1.0 - e_star) * &
                              aer_trn_ttl(i,k1,1,ib)), &
                          0.0), 1.0)
!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
         if (uvar < min_u_h2o) then
            uscl = uvar / min_u_h2o
            emis(i,ib) = emis(i,ib) * uscl
         endif

                      

!
! Band-dependent indices for window
!
         ib = 2

         uvar = ub(ib) * fdif
         log_u  = min(log10(max(uvar, min_u_h2o)), max_lu_h2o)
         dvar = (log_u - min_lu_h2o) / dlu_h2o
         iu = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
         iu1 = iu + 1
         wu = dvar - floor(dvar)
         wu1 = 1.0 - wu
         
         log_p  = min(log10(max(pnewb(ib), min_p_h2o)), max_lp_h2o)
         dvar = (log_p - min_lp_h2o) / dlp_h2o
         ip = min(max(int(aint(dvar,r4)) + 1, 1), n_p - 1)
         ip1 = ip + 1
         wp = dvar - floor(dvar)
         wp1 = 1.0 - wp

         w00_00 = wp  * w_0_00 
         w00_01 = wp  * w_0_01 
         w00_10 = wp  * w_0_10 
         w00_11 = wp  * w_0_11 
         w01_00 = wp  * w_1_00 
         w01_01 = wp  * w_1_01 
         w01_10 = wp  * w_1_10 
         w01_11 = wp  * w_1_11 
         w10_00 = wp1 * w_0_00 
         w10_01 = wp1 * w_0_01 
         w10_10 = wp1 * w_0_10 
         w10_11 = wp1 * w_0_11 
         w11_00 = wp1 * w_1_00 
         w11_01 = wp1 * w_1_01 
         w11_10 = wp1 * w_1_10 
         w11_11 = wp1 * w_1_11 

         log_uc  = min(log10(max(uch2o * fdif, min_u_h2o)), max_lu_h2o)
         dvar = (log_uc - min_lu_h2o) / dlu_h2o
         iuc = min(max(int(aint(dvar,r4)) + 1, 1), n_u - 1)
         iuc1 = iuc + 1
         wuc = dvar - floor(dvar)
         wuc1 = 1.0 - wuc
!
! Asymptotic value of emissivity as U->infinity
!
         fe = fet(1,ib) + &
              fet(2,ib) * te1 + &
              fet(3,ib) * te2 + &
              fet(4,ib) * te3 + &
              fet(5,ib) * te4 + &
              fet(6,ib) * te5

         l_star = &
              ln_eh2ow(ip , itp , iu , ite , irh ) * w11_11 * wu1 + &
              ln_eh2ow(ip , itp , iu , ite , irh1) * w11_10 * wu1 + &
              ln_eh2ow(ip , itp , iu , ite1, irh ) * w11_01 * wu1 + &
              ln_eh2ow(ip , itp , iu , ite1, irh1) * w11_00 * wu1 + &
              ln_eh2ow(ip , itp , iu1, ite , irh ) * w11_11 * wu  + &
              ln_eh2ow(ip , itp , iu1, ite , irh1) * w11_10 * wu  + &
              ln_eh2ow(ip , itp , iu1, ite1, irh ) * w11_01 * wu  + &
              ln_eh2ow(ip , itp , iu1, ite1, irh1) * w11_00 * wu  + &
              ln_eh2ow(ip , itp1, iu , ite , irh ) * w10_11 * wu1 + &
              ln_eh2ow(ip , itp1, iu , ite , irh1) * w10_10 * wu1 + &
              ln_eh2ow(ip , itp1, iu , ite1, irh ) * w10_01 * wu1 + &
              ln_eh2ow(ip , itp1, iu , ite1, irh1) * w10_00 * wu1 + &
              ln_eh2ow(ip , itp1, iu1, ite , irh ) * w10_11 * wu  + &
              ln_eh2ow(ip , itp1, iu1, ite , irh1) * w10_10 * wu  + &
              ln_eh2ow(ip , itp1, iu1, ite1, irh ) * w10_01 * wu  + &
              ln_eh2ow(ip , itp1, iu1, ite1, irh1) * w10_00 * wu  + &
              ln_eh2ow(ip1, itp , iu , ite , irh ) * w01_11 * wu1 + &
              ln_eh2ow(ip1, itp , iu , ite , irh1) * w01_10 * wu1 + &
              ln_eh2ow(ip1, itp , iu , ite1, irh ) * w01_01 * wu1 + &
              ln_eh2ow(ip1, itp , iu , ite1, irh1) * w01_00 * wu1 + &
              ln_eh2ow(ip1, itp , iu1, ite , irh ) * w01_11 * wu  + &
              ln_eh2ow(ip1, itp , iu1, ite , irh1) * w01_10 * wu  + &
              ln_eh2ow(ip1, itp , iu1, ite1, irh ) * w01_01 * wu  + &
              ln_eh2ow(ip1, itp , iu1, ite1, irh1) * w01_00 * wu  + &
              ln_eh2ow(ip1, itp1, iu , ite , irh ) * w00_11 * wu1 + &
              ln_eh2ow(ip1, itp1, iu , ite , irh1) * w00_10 * wu1 + &
              ln_eh2ow(ip1, itp1, iu , ite1, irh ) * w00_01 * wu1 + &
              ln_eh2ow(ip1, itp1, iu , ite1, irh1) * w00_00 * wu1 + &
              ln_eh2ow(ip1, itp1, iu1, ite , irh ) * w00_11 * wu  + &
              ln_eh2ow(ip1, itp1, iu1, ite , irh1) * w00_10 * wu  + &
              ln_eh2ow(ip1, itp1, iu1, ite1, irh ) * w00_01 * wu  + &
              ln_eh2ow(ip1, itp1, iu1, ite1, irh1) * w00_00 * wu 

         c_star = &
              cn_eh2ow(ip , itp , iuc , ite , irh ) * w11_11 * wuc1 + &
              cn_eh2ow(ip , itp , iuc , ite , irh1) * w11_10 * wuc1 + &
              cn_eh2ow(ip , itp , iuc , ite1, irh ) * w11_01 * wuc1 + &
              cn_eh2ow(ip , itp , iuc , ite1, irh1) * w11_00 * wuc1 + &
              cn_eh2ow(ip , itp , iuc1, ite , irh ) * w11_11 * wuc  + &
              cn_eh2ow(ip , itp , iuc1, ite , irh1) * w11_10 * wuc  + &
              cn_eh2ow(ip , itp , iuc1, ite1, irh ) * w11_01 * wuc  + &
              cn_eh2ow(ip , itp , iuc1, ite1, irh1) * w11_00 * wuc  + &
              cn_eh2ow(ip , itp1, iuc , ite , irh ) * w10_11 * wuc1 + &
              cn_eh2ow(ip , itp1, iuc , ite , irh1) * w10_10 * wuc1 + &
              cn_eh2ow(ip , itp1, iuc , ite1, irh ) * w10_01 * wuc1 + &
              cn_eh2ow(ip , itp1, iuc , ite1, irh1) * w10_00 * wuc1 + &
              cn_eh2ow(ip , itp1, iuc1, ite , irh ) * w10_11 * wuc  + &
              cn_eh2ow(ip , itp1, iuc1, ite , irh1) * w10_10 * wuc  + &
              cn_eh2ow(ip , itp1, iuc1, ite1, irh ) * w10_01 * wuc  + &
              cn_eh2ow(ip , itp1, iuc1, ite1, irh1) * w10_00 * wuc  + &
              cn_eh2ow(ip1, itp , iuc , ite , irh ) * w01_11 * wuc1 + &
              cn_eh2ow(ip1, itp , iuc , ite , irh1) * w01_10 * wuc1 + &
              cn_eh2ow(ip1, itp , iuc , ite1, irh ) * w01_01 * wuc1 + &
              cn_eh2ow(ip1, itp , iuc , ite1, irh1) * w01_00 * wuc1 + &
              cn_eh2ow(ip1, itp , iuc1, ite , irh ) * w01_11 * wuc  + &
              cn_eh2ow(ip1, itp , iuc1, ite , irh1) * w01_10 * wuc  + &
              cn_eh2ow(ip1, itp , iuc1, ite1, irh ) * w01_01 * wuc  + &
              cn_eh2ow(ip1, itp , iuc1, ite1, irh1) * w01_00 * wuc  + &
              cn_eh2ow(ip1, itp1, iuc , ite , irh ) * w00_11 * wuc1 + &
              cn_eh2ow(ip1, itp1, iuc , ite , irh1) * w00_10 * wuc1 + &
              cn_eh2ow(ip1, itp1, iuc , ite1, irh ) * w00_01 * wuc1 + &
              cn_eh2ow(ip1, itp1, iuc , ite1, irh1) * w00_00 * wuc1 + &
              cn_eh2ow(ip1, itp1, iuc1, ite , irh ) * w00_11 * wuc  + &
              cn_eh2ow(ip1, itp1, iuc1, ite , irh1) * w00_10 * wuc  + &
              cn_eh2ow(ip1, itp1, iuc1, ite1, irh ) * w00_01 * wuc  + &
              cn_eh2ow(ip1, itp1, iuc1, ite1, irh1) * w00_00 * wuc 
         emis(i,ib) = min(max(fe * (1.0 - l_star * c_star * &
                              aer_trn_ttl(i,k1,1,ib)), &
                          0.0), 1.0) 
!
! Invoke linear limit for scaling wrt u below min_u_h2o
!
         if (uvar < min_u_h2o) then
            uscl = uvar / min_u_h2o
            emis(i,ib) = emis(i,ib) * uscl
         endif

                      
!
! Compute total emissivity for H2O
!
         h2oems(i,k1) = emis(i,1)+emis(i,2)

      end do
!
!
!

      do i=1,ncol
         term7(i,1) = coefj(1,1) + coefj(2,1)*dty(i)*(1.+c16*dty(i))
         term8(i,1) = coefk(1,1) + coefk(2,1)*dty(i)*(1.+c17*dty(i))
         term7(i,2) = coefj(1,2) + coefj(2,2)*dty(i)*(1.+c26*dty(i))
         term8(i,2) = coefk(1,2) + coefk(2,2)*dty(i)*(1.+c27*dty(i))
      end do
      do i=1,ncol
!
! 500 -  800 cm-1   rotation band overlap with co2
!
         k21(i) = term7(i,1) + term8(i,1)/ &
                 (1. + (c30 + c31*(dty(i)-10.)*(dty(i)-10.))*sqrt(u(i)))
         k22(i) = term7(i,2) + term8(i,2)/ &
                 (1. + (c28 + c29*(dty(i)-10.))*sqrt(u(i)))
         fwk    = fwcoef + fwc1/(1.+fwc2*u(i))
         tr1(i) = exp(-(k21(i)*(sqrt(u(i)) + fc1*fwk*u(i))))
         tr2(i) = exp(-(k22(i)*(sqrt(u(i)) + fc1*fwk*u(i))))
         tr1(i)=tr1(i)*aer_trn_ttl(i,k1,1,idx_LW_0650_0800) 
!                                            ! H2O line+aer trn 650--800 cm-1
         tr2(i)=tr2(i)*aer_trn_ttl(i,k1,1,idx_LW_0500_0650) 
!                                            ! H2O line+aer trn 500--650 cm-1
         tr3(i) = exp(-((coefh(1,1) + coefh(2,1)*dtx(i))*uc1(i)))
         tr4(i) = exp(-((coefh(1,2) + coefh(2,2)*dtx(i))*uc1(i)))
         tr7(i) = tr1(i)*tr3(i)
         tr8(i) = tr2(i)*tr4(i)
         troco2(i,k1) = 0.65*tr7(i) + 0.35*tr8(i)
         th2o(i) = tr8(i)
      end do
!
! CO2 emissivity for 15 micron band system
!
      do i=1,ncol
         t1i    = exp(-480./co2t(i,k1))
         sqti   = sqrt(co2t(i,k1))
         rsqti  = 1./sqti
         et     = t1i
         et2    = et*et
         et4    = et2*et2
         omet   = 1. - 1.5*et2
         f1co2  = 899.70*omet*(1. + 1.94774*et + 4.73486*et2)*rsqti
         sqwp   = sqrt(plco2(i,k1))
         f1sqwp = f1co2*sqwp
         t1co2  = 1./(1. + 245.18*omet*sqwp*rsqti)
         oneme  = 1. - et2
         alphat = oneme**3*rsqti
         wco2   = 2.5221*co2vmr*pnm(i,k1)*rga
         u7     = 4.9411e4*alphat*et2*wco2
         u8     = 3.9744e4*alphat*et4*wco2
         u9     = 1.0447e5*alphat*et4*et2*wco2
         u13    = 2.8388e3*alphat*et4*wco2
!
         tpath  = co2t(i,k1)
         tlocal = tplnke(i)
         tcrfac = sqrt((tlocal*r250)*(tpath*r300))
         pi     = pnm(i,k1)*rsslp + 2.*dpfco2*tcrfac
         posqt  = pi/(2.*sqti)
         rbeta7 =  1./( 5.3288*posqt)
         rbeta8 = 1./ (10.6576*posqt)
         rbeta9 = rbeta7
         rbeta13= rbeta9
         f2co2  = (u7/sqrt(4. + u7*(1. + rbeta7))) + &
                  (u8/sqrt(4. + u8*(1. + rbeta8))) + &
                  (u9/sqrt(4. + u9*(1. + rbeta9)))
         f3co2  = u13/sqrt(4. + u13*(1. + rbeta13))
         tmp1   = log(1. + f1sqwp)
         tmp2   = log(1. +  f2co2)
         tmp3   = log(1. +  f3co2)
         absbnd = (tmp1 + 2.*t1co2*tmp2 + 2.*tmp3)*sqti
         tco2(i)=1.0/(1.0+10.0*(u7/sqrt(4. + u7*(1. + rbeta7))))
         co2ems(i,k1)  = troco2(i,k1)*absbnd*co2plk(i)
         ex     = exp(960./tint(i,k1))
         exm1sq = (ex - 1.)**2
         co2em(i,k1) = 1.2e11*ex/(tint(i,k1)*tint4(i,k1)*exm1sq)
      end do
!
! O3 emissivity
!
      do i=1,ncol
         h2otr(i,k1) = exp(-12.*s2c(i,k1))
          h2otr(i,k1)=h2otr(i,k1)*aer_trn_ttl(i,k1,1,idx_LW_1000_1200)
         te          = (co2t(i,k1)/293.)**.7
         u1          = 18.29*plos(i,k1)/te
         u2          = .5649*plos(i,k1)/te
         phat        = plos(i,k1)/plol(i,k1)
         tlocal      = tplnke(i)
         tcrfac      = sqrt(tlocal*r250)*te
         beta        = (1./.3205)*((1./phat) + (dpfo3*tcrfac))
         realnu      = (1./beta)*te
         o3bndi      = 74.*te*(tplnke(i)/375.)*log(1. + fo3(u1,realnu) + fo3(u2,realnu))
         o3ems(i,k1) = dbvtt(i)*h2otr(i,k1)*o3bndi
         to3(i)=1.0/(1. + 0.1*fo3(u1,realnu) + 0.1*fo3(u2,realnu))
      end do
!
!   Calculate trace gas emissivities
!
      call trcems(lchnk   ,ncol    ,                            &
                  k1      ,co2t    ,pnm     ,ucfc11  ,ucfc12  , &
                  un2o0   ,un2o1   ,bn2o0   ,bn2o1   ,uch4    , &
                  bch4    ,uco211  ,uco212  ,uco213  ,uco221  , &
                  uco222  ,uco223  ,uptype  ,w       ,s2c     , &
                  u       ,emplnk  ,th2o    ,tco2    ,to3     , &
                  emstrc  , &
                  aer_trn_ttl)
!
! Total emissivity:
!
      do i=1,ncol
         emstot(i,k1) = h2oems(i,k1) + co2ems(i,k1) + o3ems(i,k1)  &
                        + emstrc(i,k1)
      end do
   end do ! End of interface loop

   return
end subroutine radems

subroutine radtpl(lchnk   ,ncol    ,                            &
                  tnm     ,lwupcgs ,qnm     ,pnm     ,plco2   ,plh2o   , &
                  tplnka  ,s2c     ,tcg     ,w       ,tplnke  , &
                  tint    ,tint4   ,tlayr   ,tlayr4  ,pmln    , &
                  piln    ,plh2ob  ,wb      )
!--------------------------------------------------------------------
!
! Purpose:
! Compute temperatures and path lengths for longwave radiation
!
! Method:
! <Describe the algorithm(s) used in the routine.>
! <Also include any applicable external references.>
!
! Author: CCM1
!
!--------------------------------------------------------------------
!bloss #include <crdcon.h>

!------------------------------Arguments-----------------------------
!
! Input arguments
!
   integer, intent(in) :: lchnk                 ! chunk identifier
   integer, intent(in) :: ncol                  ! number of atmospheric columns

   real(r4), intent(in) :: tnm(pcols,pver)      ! Model level temperatures
   real(r4), intent(in) :: lwupcgs(pcols)       ! Surface longwave up flux
   real(r4), intent(in) :: qnm(pcols,pver)      ! Model level specific humidity
   real(r4), intent(in) :: pnm(pcols,pverp)     ! Pressure at model interfaces (dynes/cm2)
   real(r4), intent(in) :: pmln(pcols,pver)     ! Ln(pmidm1)
   real(r4), intent(in) :: piln(pcols,pverp)    ! Ln(pintm1)
!
! Output arguments
!
   real(r4), intent(out) :: plco2(pcols,pverp)   ! Pressure weighted co2 path
   real(r4), intent(out) :: plh2o(pcols,pverp)   ! Pressure weighted h2o path
   real(r4), intent(out) :: tplnka(pcols,pverp)  ! Level temperature from interface temperatures
   real(r4), intent(out) :: s2c(pcols,pverp)     ! H2o continuum path length
   real(r4), intent(out) :: tcg(pcols,pverp)     ! H2o-mass-wgted temp. (Curtis-Godson approx.)
   real(r4), intent(out) :: w(pcols,pverp)       ! H2o path length
   real(r4), intent(out) :: tplnke(pcols)        ! Equal to tplnka
   real(r4), intent(out) :: tint(pcols,pverp)    ! Layer interface temperature
   real(r4), intent(out) :: tint4(pcols,pverp)   ! Tint to the 4th power
   real(r4), intent(out) :: tlayr(pcols,pverp)   ! K-1 level temperature
   real(r4), intent(out) :: tlayr4(pcols,pverp)  ! Tlayr to the 4th power
   real(r4), intent(out) :: plh2ob(nbands,pcols,pverp)! Pressure weighted h2o path with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 
   real(r4), intent(out) :: wb(nbands,pcols,pverp)    ! H2o path length with 
                                                      !    Hulst-Curtis-Godson temp. factor 
                                                      !    for H2O bands 

!
!---------------------------Local variables--------------------------
!
   integer i                 ! Longitude index
   integer k                 ! Level index
   integer kp1               ! Level index + 1

   real(r4) repsil               ! Inver ratio mol weight h2o to dry air
   real(r4) dy                   ! Thickness of layer for tmp interp
   real(r4) dpnm                 ! Pressure thickness of layer
   real(r4) dpnmsq               ! Prs squared difference across layer
   real(r4) dw                   ! Increment in H2O path length
   real(r4) dplh2o               ! Increment in plh2o
   real(r4) cpwpl                ! Const in co2 mix ratio to path length conversn

!--------------------------------------------------------------------
!
   repsil = 1./epsilo
!
! Compute co2 and h2o paths
!
   cpwpl = amco2/amd * 0.5/(gravit*p0)
   do i=1,ncol
      plh2o(i,ntoplw)  = rgsslp*qnm(i,ntoplw)*pnm(i,ntoplw)*pnm(i,ntoplw)
      plco2(i,ntoplw)  = co2vmr*cpwpl*pnm(i,ntoplw)*pnm(i,ntoplw)
   end do
   do k=ntoplw,pver
      do i=1,ncol
         plh2o(i,k+1)  = plh2o(i,k) + rgsslp* &
                         (pnm(i,k+1)**2 - pnm(i,k)**2)*qnm(i,k)
         plco2(i,k+1)  = co2vmr*cpwpl*pnm(i,k+1)**2
      end do
   end do
!
! Set the top and bottom intermediate level temperatures,
! top level planck temperature and top layer temp**4.
!
! Tint is lower interface temperature
! (not available for bottom layer, so use ground temperature)
!
   do i=1,ncol
      tint4(i,pverp)   = lwupcgs(i)/stebol
      tint(i,pverp)    = sqrt(sqrt(tint4(i,pverp)))
      tplnka(i,ntoplw) = tnm(i,ntoplw)
      tint(i,ntoplw)   = tplnka(i,ntoplw)
      tlayr4(i,ntoplw) = tplnka(i,ntoplw)**4
      tint4(i,ntoplw)  = tlayr4(i,ntoplw)
   end do
!
! Intermediate level temperatures are computed using temperature
! at the full level below less dy*delta t,between the full level
!
   do k=ntoplw+1,pver
      do i=1,ncol
         dy = (piln(i,k) - pmln(i,k))/(pmln(i,k-1) - pmln(i,k))
         tint(i,k)  = tnm(i,k) - dy*(tnm(i,k)-tnm(i,k-1))
         tint4(i,k) = tint(i,k)**4
      end do
   end do
!
! Now set the layer temp=full level temperatures and establish a
! planck temperature for absorption (tplnka) which is the average
! the intermediate level temperatures.  Note that tplnka is not
! equal to the full level temperatures.
!
   do k=ntoplw+1,pverp
      do i=1,ncol
         tlayr(i,k)  = tnm(i,k-1)
         tlayr4(i,k) = tlayr(i,k)**4
         tplnka(i,k) = .5*(tint(i,k) + tint(i,k-1))
      end do
   end do
!
! Calculate tplank for emissivity calculation.
! Assume isothermal tplnke i.e. all levels=ttop.
!
   do i=1,ncol
      tplnke(i)       = tplnka(i,ntoplw)
      tlayr(i,ntoplw) = tint(i,ntoplw)
   end do
!
! Now compute h2o path fields:
!
   do i=1,ncol
!
! Changed effective path temperature to std. Curtis-Godson form
!
      tcg(i,ntoplw) = rga*qnm(i,ntoplw)*pnm(i,ntoplw)*tnm(i,ntoplw)
      w(i,ntoplw)   = sslp * (plh2o(i,ntoplw)*2.) / pnm(i,ntoplw)
!
! Hulst-Curtis-Godson scaling for H2O path
!
      wb(1,i,ntoplw) = w(i,ntoplw) * phi(tnm(i,ntoplw),1)
      wb(2,i,ntoplw) = w(i,ntoplw) * phi(tnm(i,ntoplw),2)
!
! Hulst-Curtis-Godson scaling for effective pressure along H2O path
!
      plh2ob(1,i,ntoplw) = plh2o(i,ntoplw) * psi(tnm(i,ntoplw),1)
      plh2ob(2,i,ntoplw) = plh2o(i,ntoplw) * psi(tnm(i,ntoplw),2)

      s2c(i,ntoplw) = plh2o(i,ntoplw)*fh2oself(tnm(i,ntoplw))*qnm(i,ntoplw)*repsil
   end do

   do k=ntoplw,pver
      do i=1,ncol
         dpnm       = pnm(i,k+1) - pnm(i,k)
         dpnmsq     = pnm(i,k+1)**2 - pnm(i,k)**2
         dw         = rga*qnm(i,k)*dpnm
         kp1        = k+1
         w(i,kp1)   = w(i,k) + dw
!
! Hulst-Curtis-Godson scaling for H2O path
!
         wb(1,i,kp1) = wb(1,i,k) + dw * phi(tnm(i,k),1)
         wb(2,i,kp1) = wb(2,i,k) + dw * phi(tnm(i,k),2)
!
! Hulst-Curtis-Godson scaling for effective pressure along H2O path
!
         dplh2o = plh2o(i,kp1) - plh2o(i,k)

         plh2ob(1,i,kp1) = plh2ob(1,i,k) + dplh2o * psi(tnm(i,k),1)
         plh2ob(2,i,kp1) = plh2ob(2,i,k) + dplh2o * psi(tnm(i,k),2)
!
! Changed effective path temperature to std. Curtis-Godson form
!
         tcg(i,kp1) = tcg(i,k) + dw*tnm(i,k)
         s2c(i,kp1) = s2c(i,k) + rgsslp*dpnmsq*qnm(i,k)* &
                      fh2oself(tnm(i,k))*qnm(i,k)*repsil
      end do
   end do
!
   return
end subroutine radtpl

subroutine radaeini( pstdx, mwdryx, mwco2x )
!
! Initialize radae module data
!
!bloss   use pmgrid,       only: masterproc
!bloss   use ioFileMod,    only: getfil
!bloss   use filenames,    only: absems_data
!bloss #ifdef SPMD
!bloss    use mpishorthand, only: mpir4, mpicom
!bloss #endif
   include 'netcdf.inc'
!
! Input variables
!
   real(r4), intent(in) :: pstdx   ! Standard pressure (dynes/cm^2)
   real(r4), intent(in) :: mwdryx  ! Molecular weight of dry air 
   real(r4), intent(in) :: mwco2x  ! Molecular weight of carbon dioxide
!
!      Variables for loading absorptivity/emissivity
!
   integer ncid_ae                ! NetCDF file id for abs/ems file

   integer pdimid                 ! pressure dimension id
   integer psize                  ! pressure dimension size

   integer tpdimid                ! path temperature dimension id
   integer tpsize                 ! path temperature size

   integer tedimid                ! emission temperature dimension id
   integer tesize                 ! emission temperature size

   integer udimid                 ! u (H2O path) dimension id
   integer usize                  ! u (H2O path) dimension size

   integer rhdimid                ! relative humidity dimension id
   integer rhsize                 ! relative humidity dimension size

   integer    ah2onwid            ! var. id for non-wndw abs.
   integer    eh2onwid            ! var. id for non-wndw ems.
   integer    ah2owid             ! var. id for wndw abs. (adjacent layers)
   integer cn_ah2owid             ! var. id for continuum trans. for wndw abs.
   integer cn_eh2owid             ! var. id for continuum trans. for wndw ems.
   integer ln_ah2owid             ! var. id for line trans. for wndw abs.
   integer ln_eh2owid             ! var. id for line trans. for wndw ems.
   
   character*(NF_MAX_NAME) tmpname! dummy variable for var/dim names
   character(len=256) locfn       ! local filename
   integer tmptype                ! dummy variable for variable type
   integer ndims                  ! number of dimensions
   integer dims(NF_MAX_VAR_DIMS)  ! vector of dimension ids
   integer natt                   ! number of attributes
!
! Variables for setting up H2O table
!
   integer t                     ! path temperature
   integer tmin                  ! mininum path temperature
   integer tmax                  ! maximum path temperature
   integer itype                 ! type of sat. pressure (=0 -> H2O only)
!
! Constants to set
!
   p0     = pstdx
   amd    = mwdryx
   amco2  = mwco2x
!
! Coefficients for h2o emissivity and absorptivity for overlap of H2O 
!    and trace gases.
!
   c16  = coefj(3,1)/coefj(2,1)
   c17  = coefk(3,1)/coefk(2,1)
   c26  = coefj(3,2)/coefj(2,2)
   c27  = coefk(3,2)/coefk(2,2)
   c28  = .5
   c29  = .002053
   c30  = .1
   c31  = 3.0e-5
!
! Initialize further longwave constants referring to far wing
! correction for overlap of H2O and trace gases; R&D refers to:
!
!            Ramanathan, V. and  P.Downey, 1986: A Nonisothermal
!            Emissivity and Absorptivity Formulation for Water Vapor
!            Journal of Geophysical Research, vol. 91., D8, pp 8649-8666
!
   fwcoef = .1           ! See eq(33) R&D
   fwc1   = .30          ! See eq(33) R&D
   fwc2   = 4.5          ! See eq(33) and eq(34) in R&D
   fc1    = 2.6          ! See eq(34) R&D

   if ( masterproc ) then
!bloss      call getfil(absems_data, locfn)
      locfn = trim(rundatadir)//trim(absems_data)
      call wrap_open(locfn, NF_NOWRITE, ncid_ae)
      
      call wrap_inq_dimid(ncid_ae, 'p', pdimid)
      call wrap_inq_dimlen(ncid_ae, pdimid, psize)
      
      call wrap_inq_dimid(ncid_ae, 'tp', tpdimid)
      call wrap_inq_dimlen(ncid_ae, tpdimid, tpsize)
      
      call wrap_inq_dimid(ncid_ae, 'te', tedimid)
      call wrap_inq_dimlen(ncid_ae, tedimid, tesize)
      
      call wrap_inq_dimid(ncid_ae, 'u', udimid)
      call wrap_inq_dimlen(ncid_ae, udimid, usize)
      
      call wrap_inq_dimid(ncid_ae, 'rh', rhdimid)
      call wrap_inq_dimlen(ncid_ae, rhdimid, rhsize)
      
      if (psize    /= n_p  .or. &
          tpsize   /= n_tp .or. &
          usize    /= n_u  .or. &
          tesize   /= n_te .or. &
          rhsize   /= n_rh) then
         call endrun ('RADAEINI: dimensions for abs/ems do not match internal def.')
      endif
      
      call wrap_inq_varid(ncid_ae, 'ah2onw',   ah2onwid)
      call wrap_inq_varid(ncid_ae, 'eh2onw',   eh2onwid)
      call wrap_inq_varid(ncid_ae, 'ah2ow',    ah2owid)
      call wrap_inq_varid(ncid_ae, 'cn_ah2ow', cn_ah2owid)
      call wrap_inq_varid(ncid_ae, 'cn_eh2ow', cn_eh2owid)
      call wrap_inq_varid(ncid_ae, 'ln_ah2ow', ln_ah2owid)
      call wrap_inq_varid(ncid_ae, 'ln_eh2ow', ln_eh2owid)
      
      call wrap_inq_var(ncid_ae, ah2onwid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: non-wndw abs. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, eh2onwid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: non-wndw ems. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, ah2owid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: window abs. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, cn_ah2owid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: cont. trans for abs. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, cn_eh2owid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: cont. trans. for ems. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, ln_ah2owid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: line trans for abs. in file /= internal def.')
      endif
      call wrap_inq_var(ncid_ae, ln_eh2owid, tmpname, tmptype, ndims, dims, natt)
      if (ndims /= 5 .or. &
           dims(1) /= pdimid    .or. &
           dims(2) /= tpdimid   .or. &
           dims(3) /= udimid    .or. &
           dims(4) /= tedimid   .or. &
           dims(5) /= rhdimid) then
         call endrun ('RADAEINI: line trans. for ems. in file /= internal def.')
      endif
      
      call wrap_get_var_realx (ncid_ae, ah2onwid,   ah2onw)
      call wrap_get_var_realx (ncid_ae, eh2onwid,   eh2onw)
      call wrap_get_var_realx (ncid_ae, ah2owid,    ah2ow)
      call wrap_get_var_realx (ncid_ae, cn_ah2owid, cn_ah2ow)
      call wrap_get_var_realx (ncid_ae, cn_eh2owid, cn_eh2ow)
      call wrap_get_var_realx (ncid_ae, ln_ah2owid, ln_ah2ow)
      call wrap_get_var_realx (ncid_ae, ln_eh2owid, ln_eh2ow)
      
      call wrap_close(ncid_ae)
!
! Set up table of H2O saturation vapor pressures for use in calculation
!     effective path RH.  Need separate table from table in wv_saturation 
!     because:
!     (1. Path temperatures can fall below minimum of that table; and
!     (2. Abs/Emissivity tables are derived with RH for water only.
!
      tmin = nint(min_tp_h2o)
      tmax = nint(max_tp_h2o)+1
      itype = 0
      do t = tmin, tmax
!bloss         call gffgch(dble(t),estblh2o(t-tmin),itype)
         call gffgch(float(t),estblh2o(t-tmin),itype)
      end do
    endif

    if (dompi) then
       if(sizeof(pstdx).ne.4) then
          if(masterproc) print*,'size of r4 type in RAD_CAM is not 4. Quitting...'
          call task_abort()
       end if
       call task_bcast_float4(0, ah2onw  , (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, eh2onw  , (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, ah2ow   , (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, cn_ah2ow, (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, cn_eh2ow, (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, ln_ah2ow, (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, ln_eh2ow, (n_p*n_tp*n_u*n_te*n_rh)) 
       call task_bcast_float4(0, estblh2o, (ntemp)                 ) 
    end if
!bloss #ifdef SPMD
!bloss     call mpibcast (ah2onw  , (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (eh2onw  , (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (ah2ow   , (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (cn_ah2ow, (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (cn_eh2ow, (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (ln_ah2ow, (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (ln_eh2ow, (n_p*n_tp*n_u*n_te*n_rh),mpir4,0,mpicom) 
!bloss     call mpibcast (estblh2o,(ntemp)                  ,mpir4,0,mpicom) 
!bloss #endif
    return
end subroutine radaeini

subroutine initialize_radbuffer
!
! Initialize radiation buffer data
!
  allocate (abstot_3d(pcols,pverp,pverp,begchunk:endchunk))
  allocate (absnxt_3d(pcols,pver,4,begchunk:endchunk))
  allocate (emstot_3d(pcols,pverp,begchunk:endchunk))

!  abstot_3d(:,:,:,:) = inf
!  absnxt_3d(:,:,:,:) = inf
!  emstot_3d(:,:,:) = inf
  abstot_3d(:,:,:,:) = 0.
  absnxt_3d(:,:,:,:) = 0.
  emstot_3d(:,:,:) = 0.

  return
end subroutine initialize_radbuffer

!-----------------------------------------------------------------------------
! Private Interfaces
!-----------------------------------------------------------------------------
function fh2oself( temp )
!
! Short function for H2O self-continuum temperature factor in
!   calculation of effective H2O self-continuum path length
!
! H2O Continuum: CKD 2.4
! Code for continuum: GENLN3
! Reference: Edwards, D.P., 1992: GENLN2, A General Line-by-Line Atmospheric
!                     Transmittance and Radiance Model, Version 3.0 Description
!                     and Users Guide, NCAR/TN-367+STR, 147 pp.
!
! In GENLN, the temperature scaling of the self-continuum is handled
!    by exponential interpolation/extrapolation from observations at
!    260K and 296K by:
!
!         TFAC =  (T(IPATH) - 296.0)/(260.0 - 296.0) 
!         CSFFT = CSFF296*(CSFF260/CSFF296)**TFAC
!
! For 800-1200 cm^-1, (CSFF260/CSFF296) ranges from ~2.1 to ~1.9
!     with increasing wavenumber.  The ratio <CSFF260>/<CSFF296>,
!     where <> indicates average over wavenumber, is ~2.07
! 
! fh2oself is (<CSFF260>/<CSFF296>)**TFAC
!
   real(r4),intent(in) :: temp     ! path temperature
   real(r4) fh2oself               ! mean ratio of self-continuum at temp and 296K

   fh2oself = 2.0727484**((296.0 - temp) / 36.0)
end function fh2oself


function phi(tpx,iband)
!
! History: First version for Hitran 1996 (C/H/E)
!          Current version for Hitran 2000 (C/LT/E)
! Short function for Hulst-Curtis-Godson temperature factors for
!   computing effective H2O path
! Line data for H2O: Hitran 2000, plus H2O patches v11.0 for 1341 missing
!                    lines between 500 and 2820 cm^-1.
!                    See cfa-www.harvard.edu/HITRAN
! Isotopes of H2O: all
! Line widths: air-broadened only (self set to 0)
! Code for line strengths and widths: GENLN3
! Reference: Edwards, D.P., 1992: GENLN2, A General Line-by-Line Atmospheric
!                     Transmittance and Radiance Model, Version 3.0 Description
!                     and Users Guide, NCAR/TN-367+STR, 147 pp.
!
! Note: functions have been normalized by dividing by their values at
!       a path temperature of 160K
!
! spectral intervals:
!   1 = 0-800 cm^-1 and 1200-2200 cm^-1
!   2 = 800-1200 cm^-1      
!
! Formulae: Goody and Yung, Atmospheric Radiation: Theoretical Basis, 
!           2nd edition, Oxford University Press, 1989.
! Phi: function for H2O path
!      eq. 6.25, p. 228
!
   real(r4),intent(in):: tpx      ! path temperature
   integer, intent(in):: iband    ! band to process
   real(r4) phi                   ! phi for given band
   real(r4),parameter ::  phi_r0(nbands) = (/ 9.60917711E-01, -2.21031342E+01/)
   real(r4),parameter ::  phi_r1(nbands) = (/ 4.86076751E-04,  4.24062610E-01/)
   real(r4),parameter ::  phi_r2(nbands) = (/-1.84806265E-06, -2.95543415E-03/)
   real(r4),parameter ::  phi_r3(nbands) = (/ 2.11239959E-09,  7.52470896E-06/)

   phi = (((phi_r3(iband) * tpx) + phi_r2(iband)) * tpx + phi_r1(iband)) &
          * tpx + phi_r0(iband)
end function phi

function psi(tpx,iband)
!
! History: First version for Hitran 1996 (C/H/E)
!          Current version for Hitran 2000 (C/LT/E)
! Short function for Hulst-Curtis-Godson temperature factors for
!   computing effective H2O path
! Line data for H2O: Hitran 2000, plus H2O patches v11.0 for 1341 missing
!                    lines between 500 and 2820 cm^-1. 
!                    See cfa-www.harvard.edu/HITRAN
! Isotopes of H2O: all
! Line widths: air-broadened only (self set to 0)
! Code for line strengths and widths: GENLN3
! Reference: Edwards, D.P., 1992: GENLN2, A General Line-by-Line Atmospheric
!                     Transmittance and Radiance Model, Version 3.0 Description
!                     and Users Guide, NCAR/TN-367+STR, 147 pp.
!
! Note: functions have been normalized by dividing by their values at
!       a path temperature of 160K
!
! spectral intervals:
!   1 = 0-800 cm^-1 and 1200-2200 cm^-1
!   2 = 800-1200 cm^-1      
!
! Formulae: Goody and Yung, Atmospheric Radiation: Theoretical Basis, 
!           2nd edition, Oxford University Press, 1989.
! Psi: function for pressure along path
!      eq. 6.30, p. 228
!
   real(r4),intent(in):: tpx      ! path temperature
   integer, intent(in):: iband    ! band to process
   real(r4) psi                   ! psi for given band
   real(r4),parameter ::  psi_r0(nbands) = (/ 5.65308452E-01, -7.30087891E+01/)
   real(r4),parameter ::  psi_r1(nbands) = (/ 4.07519005E-03,  1.22199547E+00/)
   real(r4),parameter ::  psi_r2(nbands) = (/-1.04347237E-05, -7.12256227E-03/)
   real(r4),parameter ::  psi_r3(nbands) = (/ 1.23765354E-08,  1.47852825E-05/)

   psi = (((psi_r3(iband) * tpx) + psi_r2(iband)) * tpx + psi_r1(iband)) * tpx + psi_r0(iband)
end function psi

end module radae
