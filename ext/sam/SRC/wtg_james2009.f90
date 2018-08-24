! This subroutine solves for perturbation omega based on WTG approximation as
! described in the appendix of Blossey, Bretherton & Wyant (2009):
!
!   http://dx.doi.org/10.3894/JAMES.2009.1.8
!
!  - original version: Peter Blossey (pblossey at gmail dot com), 
!                      September 2009 based on earlier implementaion in SAM.
!
! Note 1: this version handles the situation where the model top is
!   below the tropopause gracefully.  The boundary condition at the
!   tropopause (omega'=0) is enforced by adding additional levels
!   between the model top and the tropopause (whose height is assumed
!   to be 100 hPa if it is above the model top).
!
! Note 2: Below the tropopause, the temperature will be modified
!   by the WTG method through large-scale vertical motion.
!   Above the tropopause, the temperature should be nudged 
!   to the observed profile on a longish (~2 day?) timescale.
!
! Note 3: The wrapper routine allows the code to handle models
!   that index their pressure, temperature and moisture soundings
!   from either the surface-upwards or the top-downwards.  The driver
!   routine assumes that these soundings are indexed from the surface
!   upwards as in SAM, the model for which this routine was first
!   written.

subroutine wtg_james2009(nzm, pres, tabs_ref, qv_ref, tabs_model, &
     qv_model, qcond_model, f_coriolis, lambda_wtg, am_wtg, am_wtg_exp, &
     omega_wtg, ktrop)
  implicit none

  ! ======= inputs =======
  integer, intent(in) :: nzm ! number of model levels
  real, intent(in) :: pres(nzm) ! pressure of model levels in Pa (domain-mean for LES)
  real, intent(in) :: tabs_ref(nzm) ! reference temperature sounding in K
  real, intent(in) :: qv_ref(nzm)   ! reference water vapor sounding in kg/kg
  real, intent(in) :: tabs_model(nzm) ! model temperature sounding in K (domain-mean for LES)
  real, intent(in) :: qv_model(nzm)   ! model water vapor sounding in kg/kg (domain-mean for LES)
  real, intent(in) :: qcond_model(nzm) ! model condensate (liq+ice) sounding in kg/kg (domain-mean for LES)
  
  real, intent(in) :: f_coriolis ! coriolis parameter in 1/s.

  real, intent(in) :: lambda_wtg ! WTG length scale in m (JAMES2009 value = 650.e6 m)

  ! WTG momentum damping rate am(p) = am_wtg * (p/pref)**(am_wtg_exp)
  !   where pref = 1000 hPa.  
  !   JAMES2009 default was am_wtg = 1/day and am_wtg_exp = 1.  
  !   Also in paper: am_wtg = 0.5/day and am_wtg_exp=0. (vertically-uniform)
  real, intent(in) :: am_wtg     ! WTG momentum damping rate in 1/s at p/pref (default = 1./86400. /s)
  real, intent(in) :: am_wtg_exp ! exponenent in WTG momentum damping rate, dimensionless (default = 1.)

  ! ======= output =======
  integer, intent(out) :: ktrop ! index of interface just above the cold point.
  real, intent(out) :: omega_wtg(nzm) ! WTG large-scale pressure velocity in Pa/s on model levels.
  
  ! ======= local variables ======= 
  ! copies of sounding that are indexed from the surface upwards.
  real :: pres_local(nzm) ! pressure of model levels in Pa (domain-mean for LES)
  real :: tabs_ref_local(nzm) ! reference temperature sounding in K
  real :: qv_ref_local(nzm)   ! reference water vapor sounding in kg/kg
  real :: tabs_model_local(nzm) ! model temperature sounding in K (domain-mean for LES)
  real :: qv_model_local(nzm)   ! model water vapor sounding in kg/kg (domain-mean for LES)
  real :: qcond_model_local(nzm) ! model condensate (liq+ice) sounding in kg/kg (domain-mean for LES)

  real :: tmp(nzm)

  if(pres(2).gt.pres(1)) then
    ! flip pressure et al, so that they are indexed from the surface upwards.
    !   Driver routine only works with variables indexed from the surface up.
    pres_local(1:nzm) = pres(nzm:1:-1)
    tabs_ref_local(1:nzm) = tabs_ref(nzm:1:-1)
    qv_ref_local(1:nzm) = qv_ref(nzm:1:-1)
    tabs_model_local(1:nzm) = tabs_model(nzm:1:-1)
    qv_model_local(1:nzm) = qv_model(nzm:1:-1)
    qcond_model_local(1:nzm) = qcond_model(nzm:1:-1)

    ! call driver routine using local variables.
    call wtg_james2009_driver(nzm, pres_local, tabs_ref_local, qv_ref_local, tabs_model_local, &
         qv_model_local, qcond_model_local, f_coriolis, lambda_wtg, am_wtg, am_wtg_exp, &
         omega_wtg, ktrop)

    ! re-order omega to be indexed from top down.
    tmp(1:nzm) = omega_wtg(1:nzm)
    omega_wtg(1:nzm) = tmp(nzm:1:-1)

    ! re-compute ktrop
    ktrop = nzm+1-ktrop
  else
    ! call driver routine using input variables, since reordering is un-necessary.
    call wtg_james2009_driver(nzm, pres, tabs_ref, qv_ref, tabs_model, &
         qv_model, qcond_model, f_coriolis, lambda_wtg, am_wtg, am_wtg_exp, &
         omega_wtg, ktrop)
  end if

contains

  subroutine wtg_james2009_driver(nzm, pres, tabs_ref, qv_ref, tabs_model, &
       qv_model, qcond_model, f_coriolis, lambda_wtg, am_wtg, am_wtg_exp, &
       omega_wtg, ktrop)
    implicit none

    ! ======= inputs =======
    integer, intent(in) :: nzm ! number of model levels
    real, intent(in) :: pres(nzm) ! pressure of model levels in Pa (domain-mean for LES)
    real, intent(in) :: tabs_ref(nzm) ! reference temperature sounding in K
    real, intent(in) :: qv_ref(nzm)   ! reference water vapor sounding in kg/kg
    real, intent(in) :: tabs_model(nzm) ! model temperature sounding in K (domain-mean for LES)
    real, intent(in) :: qv_model(nzm)   ! model water vapor sounding in kg/kg (domain-mean for LES)
    real, intent(in) :: qcond_model(nzm) ! model condensate (liq+ice) sounding in kg/kg (domain-mean for LES)

    real, intent(in) :: f_coriolis ! coriolis parameter in 1/s.

    real, intent(in) :: lambda_wtg ! WTG length scale in m (JAMES2009 value = 650.e6 m)

    ! WTG momentum damping rate am(p) = am_wtg * (p/pref)**(am_wtg_exp)
    !   where pref = 1000 hPa.  
    !   JAMES2009 default was am_wtg = 1/day and am_wtg_exp = 1.  
    !   Also in paper: am_wtg = 0.5/day and am_wtg_exp=0. (vertically-uniform)
    real, intent(in) :: am_wtg     ! WTG momentum damping rate in 1/s at p/pref (default = 1./86400. /s)
    real, intent(in) :: am_wtg_exp ! exponenent in WTG momentum damping rate, dimensionless (default = 1.)

    ! ======= output =======
    integer, intent(out) :: ktrop ! index of interface just above the cold point.
    real, intent(out) :: omega_wtg(nzm) ! WTG large-scale pressure velocity in Pa/s on model levels.

    ! ======= local variables =======
    integer :: k

    ! flag to handle top-down indexing of pressure, temperature
    !   and moisture soundings.
    logical :: topdown_indexing = .false.

    ! NOTE: omega will be computed at interfaces in vertical.
    !   This lets the boundary conditions be cleanly applied at the surface
    !   and at the interface just above the cold point tropopause.
    real :: min_temp ! temporary variable used to find cold point of model sounding.

    ! extra levels in case model domain stops below tropopause.
    integer, parameter :: nzextra = 21

    ! A tridiagonal matrix is constructed to solve the elliptic equation for omega
    real :: tmpa(nzm+nzextra) ! lower diagonal of tridiagonal matrix
    real :: tmpb(nzm+nzextra) ! main diagonal of tridiagonal matrix
    real :: tmpc(nzm+nzextra) ! upper diagonal of tridiagonal matrix
    real :: tmp_rhs(nzm+nzextra) ! right hand side of system for omega, then the solution.

    real :: presc(nzm+nzextra-1) ! pressure at model levels in Pa.
    real :: presi(nzm+nzextra) ! pressure at model interfaces in Pa.
    real :: tv_model(nzm+nzextra) ! virtual temperature of model sounding in K
    real :: tv_ref(nzm+nzextra) !  virtual temperature of reference sounding in K

    real :: tmp_am(nzm+nzextra) ! momentum damping rate on model grid, 1/s
    real :: tmp_coef(nzm+nzextra) ! coefficient on LHS of equation for omega
    real :: coef_wtg ! coefficient on RHS of omega equation.

    logical :: short_domain ! true if domain top is below tropopause.
    real :: fac, dp_top

    real, parameter :: pres_ref = 1.e5 ! reference pressure in Pa.
    real, parameter :: pi = 3.141592653589793 ! from MATLAB, format long.
    real, parameter :: rgas = 287. ! Gas constant for dry air, J/kg/K

    ! default tropopause pressure in Pa.  
    !   ONLY used if model domain stops below tropopause.
    real, parameter :: pres_trop = 1.e4

    ! ===== find index of cold point tropopause in vertical. =====
    ! reverse pressure coordinate, and find index
    !   of cold point tropopause in the vertical.
    ktrop = nzm+1 ! default is top of model/atmosphere (counting from surface)
    min_temp = tabs_model(nzm) 
    do k = 1,nzm
      presc(k) = pres(k)
      if(tabs_model(k).lt.min_temp) then
        ktrop = k
        min_temp = tabs_model(k)
      end if
    end do

    ! check whether the model top is much below the likely location of
    ! the tropopause (~100-150 hPa).
    if((ktrop.gt.nzm-5) & ! min temperature is close to model top
         .OR.(presc(nzm).gt.2.e4) & ! model top has p>200hPa
         ) then
      ! Add extra levels (default=20) to solve for omega between
      !   the top of the model and the tropopause.
      short_domain = .true.
      ktrop = nzm+nzextra 
    else
      ! apply omega=0 boundary condition at the tropopause.
      short_domain = .false.
    end if

    ! compute pressure at interfaces up to model top
    presi(1) = 1.5*presc(1) - 0.5*presc(2)
    presi(2:nzm) = 0.5*presc(1:nzm-1) + 0.5*presc(2:nzm)
    presi(nzm+1) = 1.5*presc(nzm) - 0.5*presc(nzm-1)

    if(short_domain) then
      ! extend pressure sounding to tropopause (default=100hPa).
      !   make pressure grid spacing continuous at model top.
      dp_top = presi(nzm+1) - presi(nzm)

      if(presi(nzm+1)+float(nzextra-1)*dp_top.LT.pres_trop) then
        ! if uniformly-spaced pressure grid will reach tropopause,
        !   use  p(k) = ptop + k*dp_top 
        !   where k is the number of levels above the model top.
        fac = 0.
      else
        ! use  p(k) = ptop + k*dp_top + fac*k^2 
        !   where k is the number of levels above the model top.
        fac = (pres_trop - presi(nzm+1) - float(nzextra-1)*dp_top) &
             /float(nzextra-1)**2
      end if

      do k = 2,nzextra-1
        presi(nzm+k) = presi(nzm+1) + dp_top*float(k-1) &
             + fac*float(k-1)**2
        if(presi(nzm+k).lt.pres_trop) then
          ! truncate pressure sounding at this level, 
          !   since tropopause has been reached.
          presi(nzm+k) = pres_trop
          ktrop = nzm+k
          EXIT
        end if
      end do
      presi(nzm+nzextra) = pres_trop

      ! compute pressure at cell centers
      presc(nzm+1:ktrop-1) = &
           0.5*(presi(nzm+1:ktrop-1)+presi(nzm+2:ktrop))
    end if

    !bloss: WTG based on Appendix of Blossey, Bretherton & Wyant, JAMES 2009.

    ! with this method, a second-order differential equation is
    ! solved for omega:
    !
    !  d/dp (am^2 + f^2)/am d(omega')/dp = (Rd*k^2/p) Tv'
    !
    ! where omega' is the omega perturbation from the base state
    ! omega and Tv' is the virtual temperature pertubation from
    ! the base state.  This equation results from assuming small
    ! perturbations about a reference state that are governed by
    ! the linear, hydrostatic, steady-state, damped momentum and
    ! mass conservation equations for a single horizontal
    ! wavenumber k.  Here, am and f are the momentum damping
    ! rate and coriolis frequency, respectively.  Boundary
    ! conditions of omega'=0 at the surface and tropopause are applied.

    ! NOTE: since the pressure isn't non-dimensionalized in this
    !       implementation, we should be consistent in our units
    !       as omega' has units of pressure over time.  The approach
    !       taken here is to use the model's default units of mb
    !       and then to rescale omega into Pa/s at the end.

    ! NOTE: Tv' is computed at cell centers.  Will need to
    !    be interpolated to cell faces where we're solving for omega.
    do k = 1,MIN(ktrop-1,nzm)
      ! virtual temperature of model sounding
      tv_model(k) = tabs_model(k)*( 1. + 0.61*qv_model(k) - qcond_model(k) )

      ! virtual temperature of reference sounding (assume unsaturated)
      tv_ref(k) = tabs_ref(k)*( 1. + 0.61*qv_ref(k) )
    end do


    if(ktrop.gt.nzm+1) then
      ! extend temperature soundings above model top.
      !   assume no temperature anomalies above model.
      tv_model(nzm+1:ktrop-1) = 0.
      tv_ref(nzm+1:ktrop-1) = 0.
    end if

    do k = 1,ktrop-1
      ! momentum damping rate
      tmp_am(k) = am_wtg*(presc(k)/pres_ref)**am_wtg_exp

      ! coefficient on LHS
      tmp_coef(k) = (f_coriolis**2 + tmp_am(k)**2) / tmp_am(k)
    end do

    ! ========= Set up RHS ==========
    tmp_rhs(:) = 0.
    coef_wtg = rgas *(0.5*pi/lambda_wtg)**2 ! useful coefficient
    do k = 2,ktrop-1
      ! RHS = - Rd*k^2*Tv' / p
      tmp_rhs(k) = coef_wtg &
           * 0.5*( tv_model(k)+tv_model(k-1) - (tv_ref(k)+tv_ref(k-1)) ) &
           / presi(k) ! NOTE: normalize pressure by surface value
    end do

    ! ========= set up elliptic operator  ===========

    ! set up tridiagonal, finite difference approximation to 
    ! the elliptical operator to solve:
    !
    !         d/dp ( (f^2 + am^2) / am d/dp ) omega'  = RHS
    ! 
    ! tmpa, tmpb, tmpc are lower, main and upper diagonal,
    ! respectively.  Here, am (the momentum damping rate is
    !  taken as proportional to pressure).
    tmpa(2:ktrop-1) =  tmp_coef(1:ktrop-2) &
         /(presi(2:ktrop-1)-presi(1:ktrop-2)) &
         /(presc(2:ktrop-1) - presc(1:ktrop-2))

    tmpc(2:ktrop-1) =  tmp_coef(2:ktrop-1) &
         /(presi(3:ktrop) -presi(2:ktrop-1) ) &
         /(presc(2:ktrop-1) - presc(1:ktrop-2))

    tmpb(2:ktrop-1) = - tmpa(2:ktrop-1) - tmpc(2:ktrop-1)

    ! ========= set up boundary conditions  ===========

    ! boundary condition at surface, omega'=0 (homogeneous Dirichlet BC).
    tmpa(1) = 0.
    tmpb(1) = 1.
    tmpc(1) = 0.
    tmp_rhs(1) = 0. ! omega==0 at surface

    ! apply omega'=0 at tropopause (homogeneous Dirichlet BC).
    tmpa(ktrop) = 0.
    tmpb(ktrop) = 1.
    tmpc(ktrop) = 0.
    tmp_rhs(ktrop) = 0. ! omega==0 at tropopause

    ! ========= solve for omega ===========

    ! invert tridiagonal system to find omega: no pivoting.
    do k = 1,ktrop-1
      ! forward sweep: normalize by diagonal element.
      ! note that diagonal element is one after this step
      tmp_rhs(k) = tmp_rhs(k)/tmpb(k)
      tmpc(k)   = tmpc(k)/tmpb(k)

      ! forward sweep: eliminate lower diagonal element from next eqn.
      tmpb(k+1)   = tmpb(k+1)   - tmpa(k+1)*tmpc(k)
      tmp_rhs(k+1) = tmp_rhs(k+1) - tmpa(k+1)*tmp_rhs(k)
    end do

    tmp_rhs(ktrop) = tmp_rhs(ktrop)/tmpb(ktrop)

    do k = ktrop-1,1,-1
      ! backward sweep
      tmp_rhs(k) = tmp_rhs(k) - tmpc(k)*tmp_rhs(k+1)
    end do

    ! Interpolate omega from the interfaces onto the model levels
    !   from the surface up to the tropopause or top of model, 
    !   whichever is lower.
    omega_wtg(1:nzm) = &
         0.5*(tmp_rhs(1:nzm)+tmp_rhs(2:nzm+1))

    if(ktrop.lt.nzm+1) then
      omega_wtg(ktrop-1:nzm) = 0.  ! set omega to zero above tropopause.
    end if

  end subroutine wtg_james2009_driver

end subroutine wtg_james2009
