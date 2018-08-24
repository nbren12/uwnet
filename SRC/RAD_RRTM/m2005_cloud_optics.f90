module m2005_cloud_optics
  ! Compute cloud optical properties (optical thickness in LW and SW, w0 and g in SW) 
  !   from particle size distributions coming from Morrison 2005 microphysics scheme 
  !   This is a repackaging of routines from CAM5 originally written by Andrew Conley at NCAR. 
  !   Based on ~cesm_1.1.1/models/atm/cam/physics/rrtmg/cloud_rad_props.F90
  !
  ! Note!! The original CAM optics files have spectral band as the slowest-varying dimension
  !   of the lookup tables. To increase efficiency we reordered the arrays: 
  !  ncpdq -a lambda_scale,mu,sw_band,lw_band F_nwvl200_mu20_lam50_res64_t298_c080428.nc cam5_liq_optics.nc
  !  ncpdq -a d_eff,sw_band,lw_band iceoptics_c080917.nc cam5_ice_optics.nc
  !
  use netcdf
  use grid, only: rundatadir, masterproc, dompi
  use parkind, only : r8=>kind_rb ! Eight byte reals
  use parrrtm,      only : nbndlw ! Number of LW bands
  use parrrsw,      only : nbndsw ! Number of SW bands
  implicit none

  private
  public :: m2005_cloud_optics_init, compute_m2005_cloud_optics
  
  logical, save :: initialized = .false. 
  character(len=64) :: liqCldOpticsFile = "cam5_liq_optics.nc", & 
                       iceCldOpticsFile = "cam5_ice_optics.nc"
  
  ! Tables for liquid clouds
  integer :: nmu, nlambda
  real(r8), allocatable :: g_mu(:)           ! mu samples on grid
  real(r8), allocatable :: g_lambda(:,:)     ! lambda scale samples on grid
  real(r8), allocatable :: ext_sw_liq(:,:,:)
  real(r8), allocatable :: ssa_sw_liq(:,:,:)
  real(r8), allocatable :: asm_sw_liq(:,:,:)
  real(r8), allocatable :: abs_lw_liq(:,:,:)

  ! Tables for ice clouds and snow
  integer :: n_g_d
  real(r8), allocatable :: g_d_eff(:)        ! radiative effective diameter samples on grid
  real(r8), allocatable :: ext_sw_ice(:,:)
  real(r8), allocatable :: ssa_sw_ice(:,:)
  real(r8), allocatable :: asm_sw_ice(:,:)
  real(r8), allocatable :: abs_lw_ice(:,:)

contains
  ! --------------------------------------------------------------------------------------  
  subroutine endrun(message)
    character(len=*), intent(in) :: message
    
    print *, message
    call task_abort()  
  end subroutine endrun
  ! --------------------------------------------------------------------------------------  
  subroutine m2005_cloud_optics_init
    !
    ! Initialize lookup tables from CAM
    ! 
    integer :: status(16) = nf90_NoErr
    integer :: ncid, ncdimid, ncvarid, dimlengths(4)  
    integer :: ii, itmp(2)
    
    if(.not. initialized) then 
      if(masterproc) write(*,*) 'Initializing M2005 Cloud Optics'
      !-------------
      ! Liquid cloud lookup tables 
      !-------------
      !
      ! Determine sizes, allocate tables 
      !
      if(masterproc) then ! (sw_band, lambda_scale, mu)
        status( 1) = nf90_open(trim(rundatadir) // '/' // &
             trim(liqCldOpticsFile),nf90_nowrite,ncid)
        status( 2) = nf90_inq_dimid(ncid, 'mu', ncdimid) 
        status( 3) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(1)) 
        status( 4) = nf90_inq_dimid(ncid, 'lambda_scale', ncdimid) 
        status( 5) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(2)) 
        status( 6) = nf90_inq_dimid(ncid, 'sw_band', ncdimid) 
        status( 7) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(3)) 
        status( 8) = nf90_inq_dimid(ncid, 'lw_band', ncdimid) 
        status( 9) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(4)) 
        if(any(status(:9) /= nf90_NoErr)) then
          do ii=1,9
            if(status(ii) /= nf90_NoErr) write(*,*) 'status ', ii, ': ', nf90_strerror(status(ii))
          end do
          call endrun('Error reading sizes from liquid cloud optics file') 
        end if

        nmu = dimlengths(1)
        nlambda = dimlengths(2)
        if(dimlengths(3) /= nbndsw) call endrun("Number of RRTMG SW bands does't match liquid optics file")
        if(dimlengths(4) /= nbndlw) call endrun("Number of RRTMG LW bands does't match liquid optics file")
      end if

      if(dompi) then
        if(masterproc) then
          itmp(1) = nmu
          itmp(2) = nlambda
        end if
        call task_bcast_integer(0,itmp,2)
        nmu = itmp(1)
        nlambda = itmp(2)
      end if
      
      allocate(g_mu(nmu))
      allocate(g_lambda(nmu,nlambda))
      allocate(ext_sw_liq(nbndsw,nmu,nlambda) )
      allocate(ssa_sw_liq(nbndsw,nmu,nlambda))
      allocate(asm_sw_liq(nbndsw,nmu,nlambda))
      allocate(abs_lw_liq(nbndlw,nmu,nlambda))

      !
      ! Read tables and broadcast 
      !
      if(masterproc) then 
        status( 1) = nf90_inq_varid(ncid, 'mu', ncvarid) 
        status( 2) = nf90_get_var  (ncid, ncvarid, g_mu) 
        status( 3) = nf90_inq_varid(ncid, 'lambda', ncvarid) 
        status( 4) = nf90_get_var  (ncid, ncvarid, g_lambda) 

        status( 5) = nf90_inq_varid(ncid, 'k_abs_lw', ncvarid) 
        status( 6) = nf90_get_var  (ncid, ncvarid, abs_lw_liq) 

        status( 7) = nf90_inq_varid(ncid, 'k_ext_sw', ncvarid) 
        status( 8) = nf90_get_var  (ncid, ncvarid, ext_sw_liq) 
        status( 9) = nf90_inq_varid(ncid, 'ssa_sw', ncvarid) 
        status(10) = nf90_get_var  (ncid, ncvarid, ssa_sw_liq) 
        status(11) = nf90_inq_varid(ncid, 'asm_sw', ncvarid) 
        status(12) = nf90_get_var  (ncid, ncvarid, asm_sw_liq) 
        if(any(status(:12) /= nf90_NoErr)) then
          do ii=1,12
            if(status(ii) /= nf90_NoErr) write(*,*) 'status ', ii, ': ', nf90_strerror(status(ii))
          end do
          call endrun('Error reading coefficients from liquid cloud optics file') 
        end if
        status(1) = nf90_close(ncid) 
      end if

      if(dompi) then
        call task_bcast_real8(0, g_mu      , size(g_mu      ))
        call task_bcast_real8(0, g_lambda  , size(g_lambda  ))
        call task_bcast_real8(0, abs_lw_liq, size(abs_lw_liq))
        call task_bcast_real8(0, ext_sw_liq, size(ext_sw_liq))
        call task_bcast_real8(0, ssa_sw_liq, size(ssa_sw_liq))
        call task_bcast_real8(0, asm_sw_liq, size(asm_sw_liq))
      end if
      
      ! Following note is from CESM 1.1.1 code, presumably Andrew Conley
      ! I forgot to convert kext from m^2/Volume to m^2/Kg
      ext_sw_liq(:,:,:) = ext_sw_liq(:,:,:) / 0.9970449e3_r8 
      abs_lw_liq(:,:,:) = abs_lw_liq(:,:,:) / 0.9970449e3_r8 

      !-------------
      ! Ice lookup tables 
      !-------------
      if(masterproc) then ! (sw_band, lambda_scale, mu)
        status( 1) = nf90_open(trim(rundatadir)// '/' // &
             trim(iceCldOpticsFile),nf90_nowrite,ncid)
        status( 2) = nf90_inq_dimid(ncid, 'd_eff', ncdimid) 
        status( 3) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(1)) 
        status( 4) = nf90_inq_dimid(ncid, 'sw_band', ncdimid) 
        status( 5) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(2)) 
        status( 6) = nf90_inq_dimid(ncid, 'lw_band', ncdimid) 
        status( 7) = nf90_inquire_dimension(ncid, ncdimid, len = dimlengths(3)) 
        if(any(status(:7) /= nf90_NoErr)) then
          do ii=1,7
            if(status(ii) /= nf90_NoErr) write(*,*) 'status ', ii, ': ', nf90_strerror(status(ii))
          end do
          call endrun('Error reading sizes from ice cloud optics file') 
        end if

        n_g_d = dimlengths(1)
        if(dimlengths(2) /= nbndsw) call endrun("Number of RRTMG SW bands does't match liquid optics file")
        if(dimlengths(3) /= nbndlw) call endrun("Number of RRTMG SW bands does't match liquid optics file")
      end if

      if(dompi) then
        if(masterproc) itmp(1) = n_g_d
        call task_bcast_integer(0,itmp,1)
        n_g_d = itmp(1)
      end if
      
      allocate(g_d_eff(n_g_d))
      allocate(ext_sw_ice(nbndsw,n_g_d))
      allocate(ssa_sw_ice(nbndsw,n_g_d))
      allocate(asm_sw_ice(nbndsw,n_g_d))
      allocate(abs_lw_ice(nbndlw,n_g_d))

      if(masterproc) then 
        status( 1) = nf90_inq_varid(ncid, 'd_eff', ncvarid) 
        status( 2) = nf90_get_var  (ncid, ncvarid, g_d_eff) 

        status( 3) = nf90_inq_varid(ncid, 'lw_abs', ncvarid) 
        status( 4) = nf90_get_var  (ncid, ncvarid, abs_lw_ice) 

        status( 5) = nf90_inq_varid(ncid, 'sw_ext', ncvarid) 
        status( 6) = nf90_get_var  (ncid, ncvarid, ext_sw_ice) 
        status( 7) = nf90_inq_varid(ncid, 'sw_ssa', ncvarid) 
        status( 8) = nf90_get_var  (ncid, ncvarid, ssa_sw_ice) 
        status( 9) = nf90_inq_varid(ncid, 'sw_asm', ncvarid) 
        status(10) = nf90_get_var  (ncid, ncvarid, asm_sw_ice) 
        if(any(status(:10) /= nf90_NoErr)) then
          do ii=1,10
            if(status(ii) /= nf90_NoErr) write(*,*) nf90_strerror(status(ii))
          end do
          call endrun('Error reading coefficients from ice cloud optics file') 
        end if
        status(1) = nf90_close(ncid) 
      end if

      if(dompi) then
        call task_bcast_real8(0, g_d_eff   , size(g_d_eff   ))
        call task_bcast_real8(0, abs_lw_ice, size(abs_lw_ice))
        call task_bcast_real8(0, ext_sw_ice, size(ext_sw_ice))
        call task_bcast_real8(0, ssa_sw_ice, size(ssa_sw_ice))
        call task_bcast_real8(0, asm_sw_ice, size(asm_sw_ice))
      end if

    end if 
    initialized = .true. 
  end subroutine m2005_cloud_optics_init
  ! --------------------------------------------------------------------------------------  
  subroutine compute_m2005_cloud_optics(nx, nz, ilat, layerMass, cloudFrac, &
       tauLW, tauSW, ssaSW, asmSW, forSW, tauSW_cldliq, tauSW_cldice, tauSW_snow) 
    !
    ! Provide total optical properties from all radiatively-active species: liquid, ice, and snow
    !   Liquid cloud lookup tables are expressed in terms of two parameters of the drop size 
    !   distribution; ice and snow properties use the same table stored as functions of 
    !   generalized effective diameter
    !
    use micro_params, only: rho_cloud_ice, rho_snow
    use microphysics, only: &
         CloudLiquidMassMixingRatio, CloudLiquidGammaExponent, CloudLiquidLambda, &
         CloudIceMassMixingRatio, reffi, &
         SnowMassMixingRatio, reffs, dosnow_radiatively_active

    integer,  intent(in ) :: nx, nz, ilat ! number of columns and levels
    real(r8), intent(in ) :: layerMass(nx,nz+1)
    real(r8), intent(inout)::cloudFrac(nx,nz+1)
    real(r8), intent(out) :: tauLW(nbndlw,nx,nz+1), &
                             tauSW(nbndsw,nx,nz+1), ssaSW(nbndsw,nx,nz+1), asmSW(nbndsw,nx,nz+1), forSW(nbndsw,nx,nz+1), &
                             tauSW_cldliq(nbndsw,nx,nz+1), tauSW_cldice(nbndsw,nx,nz+1), tauSW_snow(nbndsw,nx,nz+1)

    ! inputs to cloud_optics routines
    real(r8) :: lwp(nx,nz),  pgam(nx,nz), lamc(nx,nz) ! liquid water path, gamma and lambda from Morrison 2005    
    real(r8) :: iwp(nx,nz),  dgi (nx,nz)              ! cloud ice water path, ice  generalized effective diameter
    real(r8) :: swp(nx,nz),  dgs (nx,nz)              ! snow water path,      snow generalized effective diameter

    real(r8) :: tmpTauLW(nbndlw,nx,nz), &
                tmpTauSW(nbndsw,nx,nz), tauSsa(nbndsw,nx,nz), tauSsaG(nbndsw,nx,nz), tauSsaF(nbndsw,nx,nz)

    cloudFrac(:,:) = 0.

    lwp(1:nx,1:nz) = CloudLiquidMassMixingRatio(1:nx,ilat,1:nz)*LayerMass(1:nx,1:nz)
    iwp(1:nx,1:nz) = CloudIceMassMixingRatio(1:nx,ilat,1:nz)*LayerMass(1:nx,1:nz)

    pgam(1:nx,1:nz) = CloudLiquidGammaExponent(1:nx,ilat,1:nz)
    lamc(1:nx,1:nz) = CloudLiquidLambda(1:nx,ilat,1:nz)

    ! compute generalized effective diameter for cloud ice 
    !   using code drawn from micro_mg.f90 in CESM1.
    ! Here, we use the snow effective radius output by the M2005 microphysics
    !   rather than computing it here and then deriving the generalized 
    !   effective diameter from it, as is done in micro_mg.f90.
    dgi(1:nx,1:nz) = reffi(1:nx,ilat,1:nz)*rho_cloud_ice/917._r8*2._r8

    ! 
    ! Individual routines report tau, tau*ssa, tau*ssa*asm, and tau*ssa*for; these are summed 
    !   and normalized when all contributions have been added  
    !
    ! Zero out optics arrays
    tauLW(:,:,:) = 0._r8
    tauSW(:,:,:) = 0._r8
    ssaSW(:,:,:) = 0._r8
    asmSW(:,:,:) = 0._r8
    forSW(:,:,:) = 0._r8

    tauSW_cldliq(:,:,:) = 0._r8
    tauSW_cldice(:,:,:) = 0._r8
    tauSW_snow(:,:,:) = 0._r8
    
    ! Liquid cloud optics routine puts zeros in non-cloudy cells 
    call compute_liquid_cloud_optics(nx, nz, lwp, pgam, lamc, tauLW, tauSW, ssaSW, asmSW, forSW)
    tauSW_cldliq(:,:,:) = tauSW(:,:,:)

    ! Ice clouds - routine adds to arrays
    call add_ice_cloud_optics(nx, nz, iwp, dgi, tauLW, tauSW, ssaSW, asmSW, forSW)
    tauSW_cldice(:,:,:) = tauSW(:,:,:) - tauSW_cldliq(:,:,:)

    if(dosnow_radiatively_active) then
      ! get snow water path and generalized effective diameter, as for cloud ice
      swp(1:nx,1:nz) = SnowMassMixingRatio(1:nx,ilat,1:nz)*LayerMass(1:nx,1:nz)
      dgs(1:nx,1:nz) = reffs(1:nx,ilat,1:nz)*rho_snow/917._r8*2._r8
      call add_ice_cloud_optics(nx, nz, swp, dgs, tauLW, tauSW, ssaSW, asmSW, forSW)
      tauSW_snow(:,:,:) = tauSW(:,:,:) - tauSW_cldliq(:,:,:) - tauSW_cldice(:,:,:)
    end if
    !
    ! Total cloud optical properties  
    !
    where(ssaSW(:,:,:) > 0._r8) 
      asmSW(:,:,:) = asmSW(:,:,:)/ssaSW(:,:,:)
      forSW(:,:,:) = forSW(:,:,:)/ssaSW(:,:,:)
    end where             
    where(tauSW(:,:,:) > 0._r8) 
      ssaSW(:,:,:) = ssaSW(:,:,:)/tauSW(:,:,:)
    end where             

    !bloss: Re-define cloud fraction here.
    !  Could be particularly important if snow is radiatively active.
    if(dosnow_radiatively_active) then
      cloudFrac(1:nx,1:nz) = MERGE(1., 0., lwp>0. .OR. iwp>0. .OR. swp>0.)
    else
      cloudFrac(1:nx,1:nz) = MERGE(1., 0., lwp>0. .OR. iwp>0.)
    end if

!bloss    write(*,999) MAXVAL(SUM(tauSW,DIM=1)), MAXVAL(SUM(tauLW,DIM=1)), MAXVAL(lwp), MAXVAL(dgi), MAXVAL(iwp)
!bloss    999 format('Max tauSW = ',F10.4,' max tauLW = ',F10.4,' max lwp = ',F10.4,' max rei = ',F10.4' max iwp = ',F10.4)
  end subroutine compute_m2005_cloud_optics
  ! --------------------------------------------------------------------------------------  
  subroutine compute_liquid_cloud_optics(nx, nz, lwp, pgam, lamc, tauLW, tauSW, taussaSW, taussagSW, taussafSW) 
    integer,  intent(in) :: nx, nz ! number of columns and levels
    real(r8), intent(in) :: lwp(nx, nz),  pgam(nx, nz), lamc(nx, nz) ! liquid water path, gamma and lambda from Morrison 2005    
    real(r8), intent(inout) :: tauLW(nbndlw,nx,nz+1), &
                             tauSW(nbndsw,nx,nz+1), taussaSW(nbndsw,nx,nz+1), taussagSW(nbndsw,nx,nz+1), taussafSW(nbndsw,nx,nz+1)
                             ! Provide tau, tau*ssa, tau*ssa*g, tau*ssa*f = tau*ssa*g*g to make 
                             !   summing over liquid/ice/snow in calling routines more efficient.  
                             
    ! Interpolation variables
    integer  :: nUse, i, j, bnd
    integer  :: iUse(nx*nz), jUse(nx*nz) 
    integer  :: kmu(nx*nz), klambda(nx*nz)
    real(r8) :: wmu(nx*nz), onemwmu(nx*nz)
    real(r8) :: wlambda, onemwlambda, lambdaplus, lambdaminus, f1, f2, f3, f4
    real(r8) :: ext(nbndsw), ssa(nbndsw), asm(nbndsw) 
    
    nUse = 0
    do j = 1, nz
      do i = 1, nx
        if(lamc(i,j) > 0. .and. lwp(i,j) > 0.) then 
          nUse = nUse + 1
          iUse(nUse) = i
          jUse(nUse) = j
        else
          tauLW    (1:nbndlw,i,j) = 0._r8
          tauSW    (1:nbndsw,i,j) = 0._r8
          taussaSW (1:nbndsw,i,j) = 0._r8
          taussagSW(1:nbndsw,i,j) = 0._r8
          taussafSW(1:nbndsw,i,j) = 0._r8
        end if 
      end do 
    end do 

    if(nUse.eq.0) return

    do i = 1, nUse
      do j = 1, nmu
        if (g_mu(j) > pgam(iUse(i),jUse(i))) exit
      enddo
      kmu(i) = j
      wmu(i) = (g_mu(kmu(i)) - pgam(iUse(i),jUse(i)))/(g_mu(kmu(i)) - g_mu(kmu(i)-1))
      wmu(i) = MAX(0._r8,MIN(1._r8,wmu(i))) !bloss: Bound between zero and one (to be sure)
      onemwmu(i) = 1._r8 - wmu(i)
      
      klambda(i) = -1 !bloss: Initialize to invalid value as check
      if (wmu(i)*g_lambda(kmu(i)-1,2) + onemwmu(i)*g_lambda(kmu(i),2) < lamc(iUse(i),jUse(i))) then
        klambda(i) = 2
      elseif (wmu(i)*g_lambda(kmu(i)-1,nlambda-1) + onemwmu(i)*g_lambda(kmu(i),nlambda-1) >= lamc(iUse(i),jUse(i))) then
        klambda(i) = nlambda
      else
        do j = 3, nlambda-1
          if (wmu(i)*g_lambda(kmu(i)-1,j) + onemwmu(i)*g_lambda(kmu(i),j) < lamc(iUse(i),jUse(i))) exit
        enddo
        klambda(i) = j
      end if
    end do 
    if (any(klambda(:nUse) < 1 .or. klambda(:nUse) > nlambda)) then
      call endrun('compute_cloud_optics: lamc exceeds limits')
    end if

    do i = 1, nUse
      !
      ! Interpolation weights
      !
      lambdaplus  = wmu(i)*g_lambda(kmu(i)-1,klambda(i)  ) + onemwmu(i)*g_lambda(kmu(i),klambda(i)  )
      lambdaminus = wmu(i)*g_lambda(kmu(i)-1,klambda(i)-1) + onemwmu(i)*g_lambda(kmu(i),klambda(i)-1)
      wlambda = (lambdaplus - lamc(iUse(i),jUse(i))) / (lambdaplus - lambdaminus)
      wlambda = MAX(0._r8,MIN(1._r8,wlambda)) !bloss: This limit may be used if lambda is larger than bounds.
      onemwlambda = 1._r8 - wlambda
      f1 =     wlambda*    wmu(i)
      f2 = onemwlambda*    wmu(i) 
      f3 =     wlambda*onemwmu(i)
      f4 = onemwlambda*onemwmu(i)
      
      !
      ! Longwave cloud properties
      !
      do bnd = 1, nbndlw
         tauLW(bnd, iUse(i), jUse(i)) = lwp(iUse(i),jUse(i)) * ( & 
                                        f1*abs_lw_liq(bnd,kmu(i)-1,klambda(i)-1) + &
                                        f2*abs_lw_liq(bnd,kmu(i)-1,klambda(i)  ) + &
                                        f3*abs_lw_liq(bnd,kmu(i)  ,klambda(i)-1) + &
                                        f4*abs_lw_liq(bnd,kmu(i)  ,klambda(i)  ) )
      end do

      !
      ! Shortwave cloud properties
      !
      do bnd = 1, nbndsw
        ext(bnd) = lwp(iUse(i),jUse(i)) * (             &
                   f1*ext_sw_liq(bnd,kmu(i)-1,klambda(i)-1) + &
                   f2*ext_sw_liq(bnd,kmu(i)-1,klambda(i)  ) + &
                   f3*ext_sw_liq(bnd,kmu(i)  ,klambda(i)-1) + &
                   f4*ext_sw_liq(bnd,kmu(i)  ,klambda(i)  ) )
        ssa(bnd) = f1*ssa_sw_liq(bnd,kmu(i)-1,klambda(i)-1) + &
                   f2*ssa_sw_liq(bnd,kmu(i)-1,klambda(i)  ) + &
                   f3*ssa_sw_liq(bnd,kmu(i)  ,klambda(i)-1) + &
                   f4*ssa_sw_liq(bnd,kmu(i)  ,klambda(i)  )
        asm(bnd) = f1*asm_sw_liq(bnd,kmu(i)-1,klambda(i)-1) + &
                   f2*asm_sw_liq(bnd,kmu(i)-1,klambda(i)  ) + &
                   f3*asm_sw_liq(bnd,kmu(i)  ,klambda(i)-1) + &
                   f4*asm_sw_liq(bnd,kmu(i)  ,klambda(i)  )
      end do
      do bnd = 1, nbndsw
        tauSW    (bnd,iUse(i),jUse(i)) = ext(bnd) 
        taussaSW (bnd,iUse(i),jUse(i)) = ext(bnd) * ssa(bnd)
        taussagSW(bnd,iUse(i),jUse(i)) = ext(bnd) * ssa(bnd) * asm(bnd)
        taussafSW(bnd,iUse(i),jUse(i)) = ext(bnd) * ssa(bnd) * asm(bnd) * asm(bnd) ! f = g**2
      end do
    end do
  end subroutine compute_liquid_cloud_optics
  ! --------------------------------------------------------------------------------------  
  subroutine add_ice_cloud_optics(nx, nz, wp, dg, tauLW, tauSW, tauSsaSW, tauSsaGSW, tauSsaFSW) 
    !
    ! Optical properties for ice or snow
    ! 
    integer,  intent(in) :: nx, nz ! number of columns and levels
    real(r8), intent(in) :: wp(nx, nz),  dg(nx, nz) ! cloud ice water path, ice  generalized effective diameter
    real(r8), intent(inout) :: tauLW(nbndlw,nx,nz+1), &
                               tauSW(nbndsw,nx,nz+1), tauSsaSW(nbndsw,nx,nz+1), tauSsaGSW(nbndsw,nx,nz+1), tauSsaFSW(nbndsw,nx,nz+1)
                             ! Provide tau, tau*ssa, tau*ssa*g, tau*ssa*f = tau*ssa*g*g to make 
                             !   summing over liquid/ice/snow in calling routines more efficient.  

    integer  :: nuse, i, j, bnd
    integer  :: iUse(nx*nz), jUse(nx*nz) 
    integer  :: k_d_eff(nx*nz)
    real(r8) :: wd(nx*nz), onemwd(nx*nz) 
    real(r8) :: ext(nbndsw), ssa(nbndsw), asm(nbndsw) 

    nUse = 0
    do j = 1, nz
      do i = 1, nx
        if(wp(i,j) > 0. .and. dg(i,j) > 0.) then 
          nUse = nUse + 1
          iUse(nUse) = i
          jUse(nUse) = j
        end if
      end do 
    end do 
    
    if (nUse.eq.0) return
    !
    ! Following CAM optics code we put a floor and ceiling on the value of effective diameter
    !
    do i = 1, nUse
      if (dg(iUse(i),jUse(i)) <= g_d_eff(1)) then
        k_d_eff(i) = 2
        wd(i) = 1._r8
      elseif (dg(iUse(i),jUse(i)) >= g_d_eff(n_g_d)) then
        k_d_eff(i) = n_g_d
        wd(i) = 0._r8
      else
        do j = 1, n_g_d
           if(g_d_eff(j) > dg(iUse(i),jUse(i))) exit
        enddo
        k_d_eff(i) = j 
        wd(i) = (g_d_eff(k_d_eff(i)) - dg(iUse(i),jUse(i)))/(g_d_eff(k_d_eff(i)) - g_d_eff(k_d_eff(i)-1))
      end if
      onemwd(i) = 1._r8 - wd(i)
    end do
    
    do i = 1,nUse
      do bnd = 1,nbndlw
         tauLW(bnd, iUse(i), jUse(i)) = tauLW(bnd, iUse(i), jUse(i)) +        & 
                                        wp(iUse(i),jUse(i)) * (               &
                                            wd(i)*abs_lw_ice(bnd,k_d_eff(i)-1) + &
                                        onemwd(i)*abs_lw_ice(bnd,k_d_eff(i)  ) )
      end do
      
      do bnd = 1,nbndsw
        ext(bnd) = wp(iUse(i),jUse(i)) * (              &
                       wd(i)*ext_sw_ice(bnd,k_d_eff(i)-1) + &
                   onemwd(i)*ext_sw_ice(bnd,k_d_eff(i)  ) )
        ssa(bnd) =     wd(i)*ssa_sw_ice(bnd,k_d_eff(i)-1) + &
                   onemwd(i)*ssa_sw_ice(bnd,k_d_eff(i)  )
        asm(bnd) =     wd(i)*asm_sw_ice(bnd,k_d_eff(i)-1) + &
                   onemwd(i)*asm_sw_ice(bnd,k_d_eff(i)  )
      end do  
      do bnd = 1, nbndsw
        tauSW    (bnd,iUse(i),jUse(i)) = tauSW    (bnd,iUse(i),jUse(i)) + ext(bnd) 
        taussaSW (bnd,iUse(i),jUse(i)) = taussaSW (bnd,iUse(i),jUse(i)) + ext(bnd) * ssa(bnd)
        taussagSW(bnd,iUse(i),jUse(i)) = taussagSW(bnd,iUse(i),jUse(i)) + ext(bnd) * ssa(bnd) * asm(bnd)
        taussafSW(bnd,iUse(i),jUse(i)) = taussafSW(bnd,iUse(i),jUse(i)) + ext(bnd) * ssa(bnd) * asm(bnd) * asm(bnd) ! f = g**2
      end do
   enddo

  end subroutine add_ice_cloud_optics

  ! --------------------------------------------------------------------------------------  
end module m2005_cloud_optics 
