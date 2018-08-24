MODULE module_mp_water_isotopologues
  use params, only: ggr, rgas, Cp, fac_cond
  use grid, only: case, caseid, masterproc

  implicit none
  private

  public :: Drat_light_over_heavy, Mrat_light_over_heavy, iso_index_ref, iso_string, &
       alfaW_equilibrium, alfaI_equilibrium, alfaK, water_isotopologue_setparm, niso_max, &
       initprofile_isotope, isotopic_surface_flux, doAlphaAtTEnv, disable_rain_fractionation
  !bloss: Constants for computing isotope fractionation
  ! NOTE iso_index=18, H2O16
  ! NOTE iso_index=19, HDO16
  ! NOTE iso_index=20, H2O18
  ! NOTE iso_index=17, H2O17 (!!NOT IMPLEMENTED YET!!)
  !
  !                 DEUTERIUM         O18
  REAL, DIMENSION(18:20), PARAMETER, PRIVATE :: &
       calI1 = (/  0.,   16288.,      0.       /), & ! coefs in \alpha_I computation
       calI2 = (/  0.,   0.,          11.839    /), &
       calI3=  (/  0.,  -0.0934,     -0.028224 /), &
       calW1 = (/  0.,   24884.,      1137.       /), & ! coefs in \alpha_W computation
       calW2=  (/  0.,  -76.248,     -0.4156   /), &
       calW3=  (/  0.,   0.052612,   -0.0020667/)
  REAL, DIMENSION(18:20), PARAMETER :: & 
       Drat_Merlivat =  (/     1.0000,  1./0.9755,     1./0.9723 /), &  ! Merlivat (1978)
       Drat_Cappa    =  (/     1.0000,  1./0.9839,     1./0.9691 /), &  ! Cappa et al (2003)
       Mrat_light_over_heavy =  (/     18./18.,    18./19.,      18./20. /) ! M/M' where ' is the heavy isotope

  ! These settings can be changed using the WATER_ISOTOPOLOGUES namelist
  REAL, DIMENSION(18:20) :: &
       Drat_light_over_heavy = Drat_Cappa, & ! ratio of molecular diffusivities, D_v / D_v'
       R_surface = (/ 1.0000, 1.0000, 1.0000 /) ! isotopic composition of surface waters

  INTEGER, DIMENSION(1:3), PARAMETER :: &
       iso_index_ref = (/  18,  19,  20 /)         ! indices needed for alpha_* computations

  CHARACTER(LEN=3), DIMENSION(18:20), PARAMETER :: &
       iso_string = (/ 'H2O', 'HDO', 'O18' /)

  !iso_factor is a small number to make the contribution of isotopes to
  !  the mass negligible It is an inconvenience due to putting isotopes in
  !  microphysics.
  REAL     , PARAMETER ::            iso_factor=1.e-10
  !
  integer, parameter :: niso_max = 3

  ! This flag enables the fractionation coefficient for rain evaporation/equilibration
  !   and deposition onto snow/cloud ice to be computed at the environmental temperature
  !   (if true) or the surface temperature of the hydrometeor as predicted by the 
  !   microphysics (if false).
  logical :: doAlphaAtTEnv = .false. ! For sensitivity studies

  logical :: disable_rain_fractionation = .false. ! For sensitivity studies

  ! switches for Surface fluxes
  logical :: doInteractive_surface_fluxes = .true., & ! false implies surface fluxes in equilibrium w/surface
       doWindSpeedIndependent_surface_fractionation = .false. ! true uses Pfahl & Wernli (2009) formulation.

  ! option for using Rayleigh curve to define initial fractionation
  logical :: doRayleigh_InitialFractionation = .true. ! if false, fractionation is fixed to value below
  real :: FixedInitialFractionation = 1. ! only important if Rayleigh is not used.
  logical, save :: isInitialized_Rayleigh = .false.
  integer, parameter :: nRayleigh = 100
  real, dimension(nRayleigh) :: qRayleigh
  real, dimension(nRayleigh,18:20) :: RisoRayleigh

  logical, parameter :: debug = .false.

CONTAINS 

  !****************************************************************
  !****************************************************************
  !bloss 061209
  !Add function that only computes equilibrium alpha for liquid.
  !  Coefficients are from Majoube (1971, Journal
  !  de Chimie Physique et Physico-Chimie Biologique, vol. 68,
  !  no. 10, pp. 1423-1436).
  REAL FUNCTION alfaW_equilibrium(Tk,iso_index)
    !----------------------------------------------------------------
    IMPLICIT NONE
    !----------------------------------------------------------------
    REAL, INTENT(IN   ) :: Tk
    INTEGER, INTENT(IN) :: iso_index !==19 for HDO, ==20 for H2O18

    alfaW_equilibrium = &
         exp( (calW1(iso_index) + Tk*(calW2(iso_index) + Tk*calW3(iso_index))) &
              /(Tk*Tk) )

  END FUNCTION alfaW_equilibrium

  !****************************************************************
  !****************************************************************
  !bloss 061209
  !Add function that only computes equilibrium alpha for ice.
  !  Coefficients for O18 are from Majoube (1970, Nature, vol. 226,
  !  p. 1242, 27 June 1970, doi: 10.1038/2261242a0).  Those for
  !  deuterium are from Merlivat & Nief (1967, Tellus, doi:
  !  10.1111/j.2153-3490.1967.tb01465.x) as presented in Jouzel (1986,
  !  Handbook of Environmental Isotope Geochemistry) with a couple
  !  of differences in the last decimal places.
  !   Merlivat & Nief: alfaI = exp(-0.0945 + 16289/T^2)
  !   Jouzel 1986: alfaI = exp(-0.0934 + 16288/T^2)

  REAL FUNCTION alfaI_equilibrium(Tk,iso_index)
    !----------------------------------------------------------------
    IMPLICIT NONE
    !----------------------------------------------------------------
    REAL, INTENT(IN   ) :: Tk
    INTEGER, INTENT(IN) :: iso_index !==19 for HDO, ==20 for H2O18

    alfaI_equilibrium = &
         exp( (calI1(iso_index) + Tk*(calI2(iso_index) + Tk*calI3(iso_index))) &
              /(Tk*Tk) )

  END FUNCTION alfaI_equilibrium

  !****************************************************************
  !****************************************************************
  REAL FUNCTION alfaK(alpha_equil, S_tilde, Vrat_light_over_heavy, iso_index)
    !bloss (061509, modified 2014-03-31): Note that alpha_k formula 
    !  is identical for liquid and ice.  Use a single function for both.
    !  The difference is in the equilibrium fractionation coefficient
    !  and the value of the saturation ratio S_tilde that are input.
    !  These will reflect whether the exchange occurs between vapor 
    !  and ice or vapor and liquid.
    !  Modification (in 2014): change formulation to remove dependence
    !  on material properties through the constant b in Gedzelman
    !  & Arnold (1994, doi: 10.1029/93JD03518) and the appendix of 
    !  Blossey et al (2010, doi: 10.1029/2010JD014554, eqn B26).  
    !  The current form is that of equation 11 in Jouzel & Merlivat 
    !  (1984, doi: 10.1029/JD089iD07p11749) where our S_tilde 
    !  corresponds to their Si, which is defined in their equation 13.
    !  In addition, we have included the ratio of the ventilation
    !  factors.
    !----------------------------------------------------------------
    IMPLICIT NONE
    !----------------------------------------------------------------
    REAL, INTENT(IN   ) :: alpha_equil  ! equilibrium fractionation
    REAL, INTENT(IN   ) :: S_tilde  ! saturation ratio = rv*^drop/rv_ambient
                                    !  equivalent to Si (Jouzel & Merlivat, 1984, eqn 13).
    REAL, INTENT(IN   ) :: Vrat_light_over_heavy ! = V/V' where V and V' are
                                ! ventilation coefficients for the 
                                ! standard and heavy isotopes, respectively. 
    INTEGER, INTENT(IN) :: iso_index !==19 for HDO, ==20 for H2O18

    alfaK = S_tilde / (1. + alpha_equil*(S_tilde - 1.) &
                             *Drat_light_over_heavy(iso_index) &
                             *Vrat_light_over_heavy)

  END FUNCTION alfaK
  
  subroutine water_isotopologue_setparm(niso,isoname,iso_index)
    implicit none

    ! outputs
    integer, intent(out) :: niso, iso_index(niso_max)
    character(LEN=3), DIMENSION(niso_max), intent(out) :: isoname

    !local variables
    logical :: dohdo = .false., doh2o18 = .false., doh2o17 = .false., doh2o16 = .false., &
         doMerlivat_diffusivities = .false.
    integer :: n, place_holder, ios, ios_missing_namelist
    real :: R_surface_hdo = -9999., R_surface_h2o18 = -9999., R_surface_h2o17 = -9999.

    NAMELIST /WATER_ISOTOPOLOGUES/ dohdo, doh2o18, doh2o17, doh2o16, &
         doMerlivat_diffusivities, &
         doInteractive_surface_fluxes, & ! 
         doWindSpeedIndependent_surface_fractionation, & ! true uses Pfahl & Wernli (2009) formulation.
         R_surface_hdo, R_surface_h2o18, R_surface_h2o17, &
         doRayleigh_InitialFractionation, &
         FixedInitialFractionation, &
         doAlphaAtTEnv, &
         disable_rain_fractionation

   !bloss: Create dummy namelist, so that we can figure out error code
   !       for a mising namelist.  This lets us differentiate between
   !       missing namelists and those with an error within the namelist.
   NAMELIST /BNCUIODSBJCB/ place_holder

     !----------------------------------
   !  Read namelist for microphysics options from prm file:
   !------------
   open(55,file='./'//trim(case)//'/prm', status='old',form='formatted') 

   !bloss: get error code for missing namelist (by giving the name for
   !       a namelist that doesn't exist in the prm file).
   read (UNIT=55,NML=BNCUIODSBJCB,IOSTAT=ios_missing_namelist)
   rewind(55) !note that one must rewind before searching for new namelists

   !bloss: read in WATER_ISOTOPOLOGUES namelist
   read (55,WATER_ISOTOPOLOGUES,IOSTAT=ios)

   if (ios.ne.0) then
     !namelist error checking
     if(ios.ne.ios_missing_namelist) then
       write(*,*) '****** ERROR: bad specification in WATER_ISOTOPOLOGUES namelist'
       rewind(55)
       read (55,WATER_ISOTOPOLOGUES) ! read it again to get a useful error message
       call task_abort()
     elseif(masterproc) then
       write(*,*) '****************************************************'
       write(*,*) '*** No WATER_ISOTOPOLOGUES namelist in prm file ****'
       write(*,*) '****************************************************'
     end if
   end if
   close(55)

   ! write namelist values out to file for documentation
   if(masterproc) then
      open(unit=55,file='./'//trim(case)//'/'//trim(case)//'_'//trim(caseid)//'.nml', form='formatted', position='append')    
      write (unit=55,nml=WATER_ISOTOPOLOGUES,IOSTAT=ios)
      write(55,*) ' '
      close(unit=55)
   end if

    niso = 0
    if(doh2o16) then
      ! have an isotopologue tracer that should track standard water h2O16
      niso = niso + 1
      iso_index(niso) = iso_index_ref(1)
      if(masterproc) write(*,997) 'H2O16', niso, iso_index(niso)
      997 format('Isotopologue implementation of ',A5,' enabled, iso_index(',I2,') = ',I4)

    end if

    if(dohdo) then
      ! have an isotopologue tracer for deuterated water HDO
      niso = niso + 1
      iso_index(niso) = iso_index_ref(2)
      if(masterproc) write(*,997) 'HDO  ', niso, iso_index(niso)
    end if

    if(doh2o18) then
      ! have an isotopologue tracer for heavy water H2O18
      niso = niso + 1
      iso_index(niso) = iso_index_ref(3)
      if(masterproc) write(*,997) 'H2O18', niso, iso_index(niso)
    end if

    if(doh2o17) then
      if(masterproc) write(*,*) 'H2O17 not yet implemented, stopping ...'
      call task_abort()
    end if

    do n = 1,niso
      isoname(n) = iso_string(iso_index(n))
    end do

    if(doMerlivat_diffusivities) then
      Drat_light_over_heavy = Drat_Merlivat
      if(masterproc) write(*,*) 'Using Merlivat (1978) isotopic diffusivities in place of Cappa et al (2003)'
    end if

    if(doInteractive_surface_fluxes) then
      if(masterproc) write(*,*) 'Isotopic surface fluxes computed interactively (roughly) following Bony et al (2008)'
      if(doWindSpeedIndependent_surface_fractionation) then
        if(masterproc) write(*,*) 'Surface fractionation is wind-speed-independent following Pfahl & Wernli (2009)'
      else        
        if(masterproc) write(*,*) 'Surface fractionation depends on wind speed following Merlivat & Jouzel (1979)'
      end if
    else
      if(masterproc) write(*,*) 'Isotopic surface fluxes in equilibrium with surface waters'
    end if

    if( (.NOT.doInteractive_surface_fluxes) .AND. doWindSpeedIndependent_surface_fractionation) then
      if(masterproc) write(*,*) 'Wind-speed-independent surface fractionation requires interactive surface fluxes'
      if(masterproc) write(*,*) '**** Fix doInteractive_surface_fluxes==.true. in WATER_ISOTOPOLOGUES namelist'
      call task_abort()
    end if

    if(doRayleigh_InitialFractionation.AND.(ABS(FixedInitialFractionation-1.).gt.EPSILON(1.))) then
      if(masterproc) write(*,*) '***********************************************************************'
      if(masterproc) write(*,*) '**** Warning: isotopic initial conditions will come from Rayleigh curve'
      if(masterproc) write(*,*) '****   The namelist value for FixedInitialFractionation will be ignored'
      if(masterproc) write(*,*) '***********************************************************************'
    end if

    ! option for user-specified isotopic content for surface waters
    if(R_surface_hdo.gt.0.) R_surface(19) = R_surface_hdo
    if(R_surface_h2o18.gt.0.) R_surface(20) = R_surface_h2o18
!bloss    if(R_surface_h2o17.gt.0.) R_surface(21) = R_surface_h2o17

  end subroutine water_isotopologue_setparm

  subroutine initprofile_isotope(N,tsurf,qsat_surf,tabs,qv,qc,iso_qv,iso_qc,pres,index)
    implicit none

    integer, intent(in) :: N, index
    real, intent(in) :: tsurf, qsat_surf
    real, dimension(N), intent(in) :: tabs, pres, qv, qc ! pressure in hPa
    real, dimension(N), intent(out) :: iso_qv, iso_qc

    real :: psurf ! in hPa
    real :: frac_m, tmpRiso
    integer :: k, m, m1

    if(doRayleigh_InitialFractionation) then
      if(.NOT.isInitialized_Rayleigh) then
        if(tsurf.lt.150.) then
          write(*,*) '** Cannot compute Rayleigh profile without a valid surface temperature'
          write(*,*) '** Set tabs_s in PARAMETERS namelist to an appropriate value'
          write(*,*) '** I realize that this is redunadant to the surface temperature '
          write(*,*) '** in the sfc file, but it is necessary here for arcane reasons.'
          STOP 'in initiprofile_isotope in module_mp_water_isotopologues.f90'
        end if

        psurf = pres(1)
        call compute_Rayleigh_Curve(tsurf, qsat_surf, psurf)
        isInitialized_Rayleigh = .true.
      end if

      do k = 1,N
        if(qv(k).gt.qRayleigh(1)) then
          m1 = 2
        elseif(qv(k).lt.qRayleigh(nRayleigh)) then
          m1 = nRayleigh
        else
          do m = 2,nRayleigh
            if((qv(k)-qRayleigh(m))*(qv(k)-qRayleigh(m-1)).lt.0.) then
              m1 = m
              EXIT
            end if
          end do
        end if
        ! interpolate to find iso_qv
        frac_m = (qv(k) - qRayleigh(m1-1)) / (qRayleigh(m1) - qRayleigh(m1-1))
        tmpRiso = RisoRayleigh(m1-1,index) &
             + frac_m*( RisoRayleigh(m1,index) - RisoRayleigh(m1-1,index) )
        iso_qv(k) = qv(k) * tmpRiso
        ! initialize cloud liquid
        iso_qc(k) = qc(k) * alfaw_equilibrium( tabs(k), index ) * tmpRiso

        if(debug) then
          write(*,992) k, m1, qRayleigh(m1-1), qv(k), qRayleigh(m1), frac_m, RisoRayleigh(m1,index), tmpRiso, iso_qv(k)
992       format(2i4,3e12.4,3f10.4,e12.4)
        end if
      end do
      if(debug) write(*,*)

    else
      if(masterproc) write(*,*) 'Fixed initial fractionation with isotope ratio = ', FixedInitialFractionation
      iso_qv(:) = FixedInitialFractionation*qv(:)
      iso_qc(:) = FixedInitialFractionation*qc(:)
    end if

  end subroutine initprofile_isotope

  subroutine compute_Rayleigh_Curve(tsurf,qsat_surf,psurf)
    implicit none

    real, intent(in) :: tsurf, qsat_surf, psurf
    ! outputs qRayleigh, RisoRayleigh are defined at top of module.

    real, dimension(nRayleigh) :: tabsRayleigh, logqv
    real, dimension(nRayleigh,18:20) :: logRiso
    real :: q0, qf, fac, theta0, theta_e0, tabs_InitialGuess
    integer :: k, index

    ! parameters used to start parcel model at the surface.
    real, parameter :: RHsurf = 0.8, dTsurf = 1. ! surface RH, temperature jump

    real, external :: qsatw, dtqsatw

    q0 = RHsurf*qsatw(tsurf-dTsurf,psurf)
    qf = qsatw(190., 100.) ! Rough tropopause temperature/pressure
    fac = 1./real(nRayleigh-1)
    do k = 1,nRayleigh
      ! log(water vapor) is uniformly spaced, from moist to dry.
      qRayleigh(k) = exp( log(q0) + fac*real(k-1)*(log(qf) - log(q0)) )
    end do

    ! Surface isotopic composition in equilibrium with surface water
    do index = 18,20
      RisoRayleigh(1,index) = R_surface(index) / alfaW_equilibrium(tsurf,index)
    end do

    ! surface moist static energy (use theta_e here if model is based on theta).
    !   Slight inconsistency here since model is based on static energy, rather than theta
    !   but we're going to ignore that here since this is only used for defining the isotopes
    !   and the likely changes from using MSE would be small, I think.  The real reason
    !   is that it's more convenient to perform saturation adjustment for theta_e since
    !   pressure enters both the qsat formula and theta_e.  MSE also involves height, so that
    !   we would need both pressure and height and a sounding to relate them.  Do-able but 
    !   more trouble to implement.
    theta0 = (tsurf - dTsurf)*(1000./psurf)**(rgas/Cp)
    theta_e0 = theta0*( 1. + fac_cond*q0/(tsurf - dTsurf) )

    tabs_InitialGuess = tsurf-dTsurf

    ! find temperature corresponding to each water vapor amount
    !   liquid-only Rayleigh line at first
    do k = 1,nRayleigh
      ! assume conservation of theta_e
      tabsRayleigh(k) = tabs_satadj_fixedqv(theta_e0,qRayleigh(k),tabs_InitialGuess)

      tabs_InitialGuess = tabsRayleigh(k)
    end do

    if(debug) then
      write(*,*)
      do k = 1,nRayleigh
        write(*,997) theta_e0, tabsRayleigh(k), qRayleigh(k)
      end do
      write(*,*)
!bloss      STOP 'in compute_Rayleigh_Curve'
    end if

    logRiso(1,:) = log(RisoRayleigh(1,:))
    logqv(:) = log(qRayleigh(:))

    do index = 18,20
      do k = 2,nRayleigh
        logRiso(k,index) = logRiso(k-1,index) + ( logqv(k) - logqv(k-1) ) &
             *( alfaW_equilibrium( 0.5*(tabsRayleigh(k-1)+tabsRayleigh(k)), index ) - 1. )
      end do
    end do

    RisoRayleigh(:,:) = exp( logRiso(:,:) )

    if(debug) then
      write(*,*)
      do k = 1,nRayleigh
        write(*,997) qRayleigh(k), (RisoRayleigh(k,index),index=18,20)
      end do
      write(*,*)
      STOP 'in compute_Rayleigh_Curve'
      997 format(4f16.8)
    end if



  end subroutine compute_Rayleigh_Curve

  real function tabs_satadj_fixedqv(thetae,qv,tabs_InitialGuess)
    real, intent(in) :: thetae, qv, tabs_InitialGuess

    real :: t1, p1, resid1, t2, p2, resid2, dt
    integer :: k

    integer, parameter :: niter = 10
    real, parameter :: relh_tol = 1.e-4, dt_tol = 1.e-3

    real, external :: qsatw, dtqsatw

    t1 = tabs_InitialGuess
    p1 = 1000.*( (t1 + fac_cond*qv) / thetae )**(Cp/rgas) ! compute p from theta_e, temperature, qv

    resid1 = qsatw( t1, p1 ) - qv
    dt = - resid1 / dtqsatw( t1, p1 ) ! first step newton

    if(debug) write(*,999) t1, p1, resid1, resid1/qv, dt
    999 format('Tguess = ',f10.4,' pguess = ',f10.4,' resid = ',e10.2,' rh resid = ',e10.2,' dt iter = ',f16.8)

    do k = 1,niter
      t2 = t1 + dt
      p2 = 1000.*( (t2 + fac_cond*qv) / thetae )**(Cp/rgas)

      resid2 = qsatw( t2, p2 ) - qv
      if((ABS(resid2).lt.relh_tol*qv).AND.(ABS(dt).lt.dt_tol)) EXIT

      ! secant method here.  dt = -resid2 * (t2 - t1) / (resid2 - resid1)
      !   To avoid a zero in the denominator, we take the absolute value of numerator 
      !   denominator and multiply by the sign of (t2-t1)*(resid2-resid1).
      dt = - resid2 * ABS( t2 - t1 ) / MAX( TINY(1.), ABS(resid2 - resid1) ) &
           * SIGN( 1., (resid2 - resid1)*(t2 - t1) )

      if(debug) write(*,999) t2, p2, resid2, resid2/qv, dt

      t1 = t2
      resid1 = resid2
    end do

    if((ABS(resid2).gt.relh_tol*qv).OR.(ABS(dt).gt.dt_tol)) then
      write(*,*) 'tabs_satadj_fixedqv did not converge'
      STOP
    else
      if(debug) write(*,*) 'converged...'
      tabs_satadj_fixedqv = t2
    end if
  end function tabs_satadj_fixedqv

  subroutine isotopic_surface_flux(n1,n2,ts,iso_qv,iso_flux,index)
    use vars, only: sstxy, dtfactor, rho, tabs, z, pres, &
         fluxbq, fluxbq_coef, ustar, qsat_surf
    implicit none

    ! in/outputs
    integer, intent(in) :: n1, n2, index
    real, dimension(n1,n2), intent(in) :: iso_qv, ts
    real, dimension(n1,n2), intent(out) :: iso_flux

    ! local variables
    integer :: tmp_iso_index, i, j, n

    real :: zrough ! roughness length [m]
    real :: mu ! dynamic viscosity of air [kg/m/s]
    real :: nu ! kinematic viscosity of air [m2/s]
    real :: diffwv ! molecular diffusivity of air [m2/s]
    real :: res ! surface reynolds number based on zrough and ustar [1]
    real :: kcoef ! k from equation 11 of Merlivat and Jouzel 1979 (MJ79)
    real :: rhoTorhoM ! ratio of turbulent to molecular "resistances", also from MJ79
    real :: nexp ! exponent from MJ79
    real :: epsD ! diffusivity ratio (standard to heavy isotope) minus one

    real :: alpha_W

    real, dimension(n1,n2) :: kfrac ! Fractionation coefficient from Pfahl & Wernli (2009)

    REAL     , PARAMETER ::                              &
         xk = 0.4,     & ! von Karman constant
         avisc = 1.49628e-6, &!const. in dynamic viscosity formula for air
         adiffwv = 8.7602e-5,&!const in diffusivity formula for water vapor
         axka = 1.4132e3 ! const in thermal conductiviy formula for air

    if(doInteractive_surface_fluxes) then
      ! compute surface fluxes over water as for the standard isotope, but with equilibrium
      !   and non-equilibrium fractionation included.
      if(doWindSpeedIndependent_surface_fractionation) then
        ! following section 5 of Pfahl & Wernli (2009, doi: 10.1029/2008JD009839)
        select case(index)
        case(18)
          kfrac(:,:) = 1. ! standard isotope
        case(19)
          kfrac(:,:) = 0.9961 ! HDO (section 5 of Pfahl & Wernli, 2009)
        case(20)
          kfrac(:,:) = 0.9925 ! H2O18 (section 5 of Pfahl & Wernli, 2009)
        end select

      else
        ! follows Merlivat & Jouzel (1979), Bony et al (2008)

        do j = 1,n2
          do i = 1,n1
            ! compute rhoT/rhoM from just after equation 11 in Merlivat and Jouzel 1979
            !   JGR, vol. 84, no. C8, p. 5030.  Hereafter, MJ79.
            mu = avisc*tabs(i,j,1)**1.5/(tabs(i,j,1)+120.) ! from wrf lin scheme.
            nu = mu/rho(1)
            diffwv = adiffwv*tabs(i,j,1)**1.81/(100.*pres(1))

            zrough = ustar(i,j)**2/81.1/ggr
            res = ustar(i,j)*zrough/nu
            if(res.ge.1) then
              ! rough regime
              rhoTorhoM = (1./xk)*(log(z(1)/zrough) - 5.) &
                   /( 7.3 * res**0.25 * sqrt(nu/diffwv) )
              nexp = 0.5
            else
              ! smooth regime
              rhoTorhoM = (1./xk)*log(ustar(i,j)*z(1)/30./nu) &
                   /( 13.6 * (nu/diffwv)**0.67 )
              nexp = 0.67
            end if

            ! compute k and eps_D from equation 11 in MJ79
            epsD = Drat_light_over_heavy(index) - 1.
            kcoef = ( (1. + epsD)**nexp - 1. ) / ( (1. + epsD)**nexp + rhoTorhoM )

            kfrac(i,j) = 1. - kcoef
          end do
        end do

      end if

      do j = 1,n2
        do i = 1,n1
          alpha_W = alfaW_equilibrium(ts(i,j),index)

          ! follow Bony et al (2008), doi:10.1029/2008JD009942
          !   sec 2.3.3, except for new value of kcoef.
          iso_flux(i,j) = - fluxbq_coef(i,j) * kfrac(i,j) &
               *( R_surface(index)*qsat_surf(i,j) / alpha_W - iso_qv(i,j) )

        end do
      end do

    else ! equilibrium surface fluxes

      ! assume surface fluxes of heavy isotopes relate to standard isotope
      !   fluxes according to equilibrium fractionation.
      do j = 1,n2
        do i = 1,n1
          ! surface fluxes follow equilibrium fractionation at Sw=1, T=SST
          iso_flux(i,j) = R_surface(index)*fluxbq(i,j)/alfaW_equilibrium(ts(i,j),index)
        end do
      end do

    end if

  end subroutine isotopic_surface_flux

END MODULE module_mp_water_isotopologues
