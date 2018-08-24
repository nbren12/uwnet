module rad
    ! -------------------------------------------------------------------------- 
    !
    ! Interface to RRTM longwave and shortwave radiation code.
    !   Robert Pincus, November 2007
    !  
    ! Modified by Peter Blossey, July 2009.
    !   - interfaced to RRTMG LW v4.8 and SW v3.8.
    !   - reversed indices in tracesini to match bottom-up RRTM indexing.
    !   - fixed issue w/doperpetual caused by eccf=0.
    !   - added extra layer in calls to RRTM to improve heating rates at
    !        model top.  Required changes to inatm in rrtmg_lw_rad.f90
    !        and inatm_sw in rrtmg_sw_rad.f90.
    !   - fixed bug that would zero out cloudFrac if LWP>0 and IWP==0.
    !   - changed definition of interfaceT(:,1) to SST.
    !   - added o3, etc. profiles to restart file.  Only call tracesini if nrestart==0.
    !
    ! Modified by Peter Blossey, July 2009.
    !   - update to RRTMG LW v4.84
    !   - add extra layer to model top within rad_driver, rather than inatm_*w.
    !   - use instantaneous fields for radiation computation, rather than time-averaged.
    !
    ! Modified by Peter Blossey, May 2014.
    !   - update to RRTMG SW v3.9 which include fix in lookup tables for cloud
    !      liquid optical properties.  See AER's description of the update here:
    !        http://rtweb.aer.com/rrtmg_sw_whats_new.html
    !
    ! Modified by Peter Blossey (with help from Robert Pincus), Sept 2015.
    !   - Moved all "use" statements to top of module.  This seems to be
    !      better fortran90 coding practice.
    !   - Coupled radiation more tightly with instrument simulators in 
    !      SRC/SIMULATORS/ by providing tau_067 and emis_105 with values from
    !      closest RRTMG bands.
    !   - Added tighter microphysics-coupling for M2005 and Thompson microphysics.
    !      MICRO_M2005 uses CAM5 cloud optics lookup tables for cloud liquid, cloud ice
    !      and snow.  MICRO_WRF (Thompson microphysics) uses the RRTMG lookup tables
    !      for cloud liquid, cloud ice and snow.  Both routines use mass and other cloud
    !      properties (mostly effective radius or generalized effective diameter for ice)
    !      to determine the optical properties.  See the new modules for computing
    !      cloud properties in m2005_cloud_optics.f90 and thompson_cloud_optics.f90.
    !      These new cloud optics routines are enabled by default.  They can be disabled
    !      in the microphysics namelists with 
    !          dorrtm_cloud_optics_from_effrad_LegacyOption = .true.
    !
    ! Modified by Peter Blossey (with help from Robert Pincus), Feb 2016.
    !   - providing separate optical depths for liquid, cloud ice and
    !      snow to better couple with the MODIS simulator in SRC/SIMULATORS/.
    !
    ! -------------------------------------------------------------------------- 

    ! -------------------------------------------------------------------------- 
  use parkind, only : kind_rb, kind_im 
  use params, only : ggr, coszrs, cp, ocean,  &
       doshortwave, dolongwave, doradhomo,      &
       doseasons, doperpetual, dosolarconstant, &
       solar_constant, zenith_angle, nxco2, notracegases
  use grid, only : nx, ny, nz, nzm, compute_reffc, compute_reffi, compute_reffl, &
       icycle, dtn, dt, nstop, nstep, nstat, nrestart, nrad, day, day0, &
       pres, presi, z, dz, adz, dompi, masterproc, nsubdomains,          &
       dostatis, dostatisrad, nelapse, nrestart_skip, case, rundatadir, &
       doisccp, domodis, domisr, &
       restart_sep, caseid, case_restart, caseid_restart, rank
  use vars, only : t, tabs, qv, qcl, qci, qpl, sstxy, rho, t00, &
       latitude, longitude,                         &
                                ! Domain-average diagnostic fields
       radlwup, radlwdn, radswup, radswdn, radqrlw, radqrsw, &
                                ! 2D diagnostics
       lwns_xy, lwnt_xy, swns_xy, swnt_xy, solin_xy, &      
       lwnsc_xy, lwntc_xy, swnsc_xy, swntc_xy, &
                                ! 1D diagnostics
       s_flns, s_fsns, s_flnt, s_flntoa, s_fsnt, s_fsntoa, &
       s_flnsc, s_fsnsc, s_flntoac, s_fsntoac, s_solin, &
       s_fsds, s_flds

  !
  ! Radiation solvers
  !
  use rrlw_ncpar, only: cpdair, maxAbsorberNameLength, &
       status, getAbsorberIndex
  use rrtmg_sw_init, only: rrtmg_sw_ini
  use rrtmg_lw_init, only: rrtmg_lw_ini
  use rrtmg_sw_rad, only : rrtmg_sw
  use rrtmg_lw_rad, only : rrtmg_lw
  use rrtmg_lw_cldprop, &
                     only: cldprop
  use rrtmg_sw_cldprop, & 
                    only : cldprop_sw
  use parrrtm,      only : nbndlw ! Number of LW bands
  use parrrsw,      only : nbndsw, naerec, jpband ! Number of SW bands
  use cam_rad_parameterizations, only : &
       computeRe_Liquid, computeRe_Ice, albedo
  use microphysics, only : micro_scheme_name, reffc, reffi, reffr, &
                           dorrtm_cloud_optics_from_effrad_LegacyOption
  use m2005_cloud_optics, &
                    only : m2005_cloud_optics_init, compute_m2005_cloud_optics
  use thompson_cloud_optics, &
                    only : thompson_cloud_optics_init, compute_thompson_cloud_optics

  use shr_orb_mod, only: shr_orb_params, shr_orb_decl, shr_orb_cosz

  implicit none
  private

  ! Public procedures
  public :: rad_driver, write_rad
  
  ! Public data
  public :: qrad, lwnsxy, swnsxy, radqrcsw, radqrclw, &
       do_output_clearsky_heating_profiles, &
       tau_067, emis_105, rad_reffc, rad_reffi, &  ! For instrument simulators
       tau_067_cldliq, tau_067_cldice, tau_067_snow ! for MODIS simulator: wants individual phases


  real, dimension(nx, ny, nzm) :: qrad ! Radiative heating rate (K/s) 
  real, dimension(nx, ny)      :: lwnsxy, swnsxy ! Long- and short-wave radiative heating (W/m2)  
  real, dimension(nx, ny, nzm) :: tau_067, emis_105, &  ! Optical thickness at 0.67 microns, emissivity at 10.5 microns, for instrument simulators
                                  tau_067_cldliq, tau_067_cldice, tau_067_snow, & ! for MODIS simulator: wants individual phases
                                  rad_reffc, rad_reffi  ! Particle sizes assumed for radiation calculation when microphysics doesn't provide them

  logical, parameter :: do_output_clearsky_heating_profiles = .true.
  real, dimension(nzm) :: radqrclw, radqrcsw ! Domain-average diagnostic fields for clearsky radiation
  !
  ! Constants
  !
  real, parameter :: Pi = 3.14159265358979312
  real(kind = kind_rb), parameter :: scon = 1367. ! solar constant 
  integer, parameter :: iyear = 1999
  !
  ! Molecular weights (taken from CAM shrc_const_mod.F90 and physconst.F90)
  !
  real, parameter :: mwdry =  28.966, & ! molecular weight dry air
                     mwco2 =  44.,    & ! molecular weight co2
                     mwh2o =  18.016, & ! molecular weight h2o
                     mwn2o =  44.,    & ! molecular weight n2o
                     mwch4 =  16.,    & ! molecular weight ch4
                     mwf11 = 136.,    & ! molecular weight cfc11
                     mwf12 = 120.,    & ! molecular weight cfc12
                     mwo3  =  48.       ! ozone, strangely missing
  ! mixingRatioMass = mol_weight/mol_weight_air * mixingRatioVolume
  
  !
  ! Global storage
  !
  logical :: initialized = .false., use_m2005_cloud_optics = .false., &
       use_thompson_cloud_optics = .false., have_cloud_optics = .false.
  real(KIND=kind_rb) :: land_frac = 1.
  
  real, dimension(nx, ny) :: &
     lwDownSurface, lwDownSurfaceClearSky, lwUpSurface, lwUpSurfaceClearSky, & 
                                           lwUpToa,     lwUpToaClearSky,     &
     lwDownTom,                            lwUpTom,  &
     swDownSurface, swDownSurfaceClearSky, swUpSurface, swUpSurfaceClearSky, & 
     swDownToa,                            swUpToa,     swUpToaClearSky,     &
     swDownTom,                            swUpTom,  &
     insolation_TOA

  !bloss(072009): changed from mass mixing ratios to volume mixing ratios
  !                 because we're now using rrtmg_lw.nc sounding for trace gases.
  ! Profiles of gas volume mixing ratios 
  !bloss(120409): add level to account for trace gases above model top.
  real(kind_rb), dimension(nzm+1) :: o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
  
  integer :: nradsteps ! current number of steps done before
				       !   calling radiation routines

  real, dimension(nx, ny) ::  p_factor ! perpetual-sun factor
   
  !
  ! Earth's orbital characteristics
  !   Calculated in shr_orb_mod which is called by rad_driver
  !
  real(kind_rb), save ::  eccf,  & ! eccentricity factor (1./earth-sun dist^2)
                 eccen, & ! Earth's eccentricity factor (unitless) (typically 0 to 0.1)
                 obliq, & ! Earth's obliquity angle (deg) (-90 to +90) (typically 22-26)
                 mvelp, & ! Earth's moving vernal equinox at perhelion (deg)(0 to 360.0)
                 !
                 ! Orbital information after processed by orbit_params
                 !
                 obliqr, &  ! Earth's obliquity in radians
                 lambm0, &  ! Mean longitude of perihelion at the
                            ! vernal equinox (radians)
                 mvelpp     ! Earth's moving vernal equinox longitude
                            ! of perihelion plus pi (radians)
!
contains 
  ! ----------------------------------------------------------------------------
  subroutine rad_driver 

    implicit none
    
    ! -------------------------------------------------------------------------- 

    ! Local variables 
    !
    ! Input and output variables for RRTMG SW and LW
    !   RRTM specifies the kind of real variables in 
    !   Only one column dimension is allowed parkind
    !   RRTM is indexed from bottom to top (like the CRM) 
    !
    !bloss: add layer to top to improve top-of-model heating rates.
    real(kind = kind_rb), dimension(nx, nzm+1) ::     &
        layerP,     layerT, layerMass,         & ! layer mass is for convenience
        h2ovmr,   o3vmr,    co2vmr,   ch4vmr, n2ovmr,  & ! Volume mixing ratios for H2O, O3, CH4, N20, CFCs
        o2vmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, &
        swHeatingRate, swHeatingRateClearSky,  &
        lwHeatingRate, lwHeatingRateClearSky, &
        duflx_dt, duflxc_dt
        
    ! Arrays for cloud optical properties or the physical properties needed by the RRTM internal parmaertizations
    real(kind = kind_rb), dimension(nx, nzm+1) ::     &
        LWP, IWP, liqRe, iceRe, cloudFrac             ! liquid/ice water path (g/m2) and size (microns)
    real(kind = kind_rb), dimension(nbndlw, nx, nz+1) :: cloudTauLW = 0. 
    real(kind = kind_rb), dimension(nbndsw, nx, nz+1) :: cloudTauSW = 0., cloudSsaSW = 0., &
                                                         cloudAsmSW = 0., cloudForSW = 0., &
                                                         cloudTauSW_cldliq = 0., &
                                                         cloudTauSW_cldice = 0., &
                                                         cloudTauSW_snow = 0.
    real(kind = kind_rb), dimension(nx, nzm+1, nbndlw) :: dummyTauAerosolLW = 0. 
    real(kind = kind_rb), dimension(nx, nzm+1, nbndsw) :: dummyAerosolProps = 0. 
    real(kind = kind_rb), dimension(nx, nzm+1, naerec) :: dummyAerosolProps2 = 0. 
    ! Arguments to RRTMG cloud optical depth routines
    real(kind = kind_rb), dimension(nbndlw, nzm+1) :: prpLWIn
    real(kind = kind_rb), dimension(nzm+1, nbndlw) :: tauLWOut
    real(kind = kind_rb), dimension(nbndsw, nzm+1) :: prpSWIn
    real(kind = kind_rb), dimension(nzm+1, jpband) :: tauSWOut, scaled1, scaled2, scaled3 
    integer                                        :: ncbands  
                                
    !bloss: add layer to top to improve top-of-model heating rates.
    real(kind = kind_rb), dimension(nx, nz+1)  :: interfaceP, interfaceT,        &
                                swUp, swDown, swUpClearSky, swDownClearSky, &
                                lwUp, lwDown, lwUpClearSky, lwDownClearSky
                                
    real(kind = kind_rb), dimension(nx) :: surfaceT, solarZenithAngleCos
                                      ! Surface direct/diffuse albedos for 0.2-0.7 (s) 
                                      !   and 0.7-5.0 (l) microns
    real(kind = kind_rb), dimension(nx) ::  asdir, asdif, aldir, aldif 

                                
    real(kind = kind_rb), dimension(nx, nbndlw) :: surfaceEmissivity = 0.95
    integer :: overlap = 1
    integer :: idrv ! option to have rrtm compute d(OLR)/dTABS for both full and clear sky
    
    integer :: lat, i, j, k
    real(kind = kind_rb) :: dayForSW, delta
    !
    ! 8 byte reals, I guess, used by MPI
    !
    !bloss: add layer to top to improve top-of-model heating rates.
    real(kind = 8),    dimension(nzm+1) :: radHeatingProfile, tempProfile
    
    real, external :: qsatw, qsati

    !bloss: extra arrays for handling liquid-only and ice-only cloud optical depth
    !   computation for MODIS simulator
    real(kind = kind_rb), dimension(nx, nzm+1) ::     &
        dummyWP, dummyRe, cloudFrac_liq, cloudFrac_ice ! liquid/ice water path (g/m2) and size (microns)

    ! ----------------------------------------------------------------------------
    if(icycle == 1) then   ! Skip subcycles (i.e. when icycle /= 1) 
    
      !------------------------------------------------------
      ! Initialize if necessary 
      !
      if(.not. initialized)  call initialize_radiation

      nradsteps = nradsteps+1
    
      !----------------------------------------------------
      ! Update radiation variables if the time is due
      !
    
      !kzm Oct.14, 03 changed == nrad to >= nrad to handle the
      ! case when a smaller nrad is used in restart  
      if(nstep == 1 .or. nradsteps >= nrad) then 
        !
        ! Initialize 1D diagnostics
        !
        radlwup(:) = 0.; radlwdn(:) = 0.
        radswup(:) = 0.; radswdn(:) = 0.
        radqrlw(:) = 0.; radqrsw(:) = 0.
        radqrclw(:) = 0.; radqrcsw(:) = 0.
        qrad(:, :, :) = 0. 

	! set trace gas concentrations.  Assumed to be uniform in the horizontal.
        o3vmr   (:, 1:nzm+1) = spread(o3   (:), dim = 1, ncopies = nx) 
        co2vmr  (:, 1:nzm+1) = spread(co2  (:), dim = 1, ncopies = nx) 
        ch4vmr  (:, 1:nzm+1) = spread(ch4  (:), dim = 1, ncopies = nx) 
        n2ovmr  (:, 1:nzm+1) = spread(n2o  (:), dim = 1, ncopies = nx) 
        o2vmr   (:, 1:nzm+1) = spread(o2   (:), dim = 1, ncopies = nx) 
        cfc11vmr(:, 1:nzm+1) = spread(cfc11(:), dim = 1, ncopies = nx) 
        cfc12vmr(:, 1:nzm+1) = spread(cfc12(:), dim = 1, ncopies = nx) 
        cfc22vmr(:, 1:nzm+1) = spread(cfc22(:), dim = 1, ncopies = nx) 
        ccl4vmr (:, 1:nzm+1) = spread(ccl4 (:), dim = 1, ncopies = nx) 
        !
        ! No concentrations for these gases
        !
        cfc22vmr(:, :) = 0. 
        ccl4vmr(:, :)  = 0.
      
        !
        ! Fill out 2D arrays needed by RRTMG 
        !
        layerP(:, 1:nzm) = spread(pres (:), dim = 1, ncopies = nx) 

        interfaceP(:, 1:nz) = spread(presi(:), dim = 1, ncopies = nx) 
        
        ! add layer to top, top pressure <= 0.01 Pa.
        layerP(:, nzm+1) = 0.5*interfaceP(:, nz) 
        interfaceP(:, nz+1) = MIN( 1.e-4_kind_rb, 0.25*MINVAL(layerP(:,nzm+1)) )

        ! Convert hPa to Pa in layer mass calculation (kg/m2) 
        layerMass(:, 1:nzm+1) = &
             100.*(interfaceP(:,1:nz) - interfaceP(:,2:nz+1))/ ggr
             
        ! Compute assumed cloud particle sizes if microphysics doesn't provide them 
        if(compute_reffc) then
          rad_reffc(1:nx,1:ny,1:nzm) = reffc(1:nx,1:ny,1:nzm)
        else
          rad_reffc(1:nx,1:ny,1:nzm) = &
                          MERGE(computeRe_Liquid(REAL(tabs(1:nx,1:ny,1:nzm),KIND=kind_rb), land_frac), &
                                0._kind_rb,                                         &
                                qcl(1:nx,1:ny,1:nzm) > 0._kind_rb )
        end if 
        if(compute_reffi) then
          rad_reffi(1:nx,1:ny,1:nzm) = reffi(1:nx,1:ny,1:nzm)
        else
          rad_reffi(1:nx,1:ny,1:nzm) = &
                          MERGE(computeRe_Ice(REAL(tabs(1:nx,1:ny,1:nzm),KIND=kind_rb)), &
                                0._kind_rb,                       &
                                qci(1:nx,1:ny,1:nzm) > 0._kind_rb )
        end if 
        
        ! option for drizzle/rain being radiatively active
        if(compute_reffl.AND.dorrtm_cloud_optics_from_effrad_LegacyOption) then
          do k = 1,nzm
            do j = 1,ny
              do i = 1,nx
                if(qpl(i,j,k)+qcl(i,j,k).gt.0._kind_rb) then
                  rad_reffc(i,j,k) = ( qcl(i,j,k) + qpl(i,j,k) ) &
                       / (qcl(i,j,k)/reffc(i,j,k) + qpl(i,j,k)/reffr(i,j,k) )
                end if
              end do
            end do
          end do
        end if

        ! The radiation code takes a 1D vector of columns, so we loop 
        !   over the y direction
        !
        do lat = 1, ny 
          lwHeatingRate(:, :) = 0.; swHeatingRate(:, :) = 0. 

          layerT(:, 1:nzm) = tabs(:, lat, 1:nzm) 
          layerT(:, nzm+1) = 2.*tabs(:, lat, nzm) - tabs(:, lat, nzm-1) ! add a layer at top.

          interfaceT(:, 2:nzm+1) = (layerT(:, 1:nzm) + layerT(:, 2:nzm+1)) / 2. 
          !
          ! Extrapolate temperature at top and bottom interfaces
          !   from lapse rate within the layer
          !
          interfaceT(:, 1)  = sstxy(1:nx, lat) + t00 
          !bloss(120709): second option for interfaceT(:,1):
          !bloss  interfaceT(:, 1)  = layerT(:, 1)   + (layerT(:, 1)   - interfaceT(:, 2))   

          interfaceT(:, nzm+2) = 2.*layerT(:, nzm+1) - interfaceT(:, nzm+1)
          ! ---------------------------------------------------
          !
          ! Compute cloud IWP/LWP and particle sizes - convert from kg to g
          !
          LWP(:, 1:nzm) = qcl(:, lat, 1:nzm) * 1.e3 * layerMass(:, 1:nzm) 
          LWP(:, nzm+1) = 0. ! zero out extra layer

          ! option for drizzle/rain being radiatively active
          if(compute_reffl.AND.dorrtm_cloud_optics_from_effrad_LegacyOption) then
            LWP(:, 1:nzm) = ( qcl(:, lat, 1:nzm) + qpl(:, lat, 1:nzm) )* 1.e3 * layerMass(:, 1:nzm) 
            LWP(:, nzm+1) = 0. ! zero out extra layer
          end if

          IWP(:, 1:nzm) = qci(:, lat, 1:nzm) * 1.e3 * layerMass(:, 1:nzm) 
          IWP(:, nzm+1) = 0. ! zero out extra layer
          cloudFrac(:,:) = MERGE(1., 0., LWP(:,:)>0. .or. IWP(:,:)>0.)

          if(have_cloud_optics) then
            if(use_m2005_cloud_optics) then
              call compute_m2005_cloud_optics(nx, nzm, lat, layerMass, cloudFrac, &
                  cloudTauLW, cloudTauSW, cloudSsaSW, cloudAsmSW, cloudForSW, &
                  cloudTauSW_cldliq, cloudTauSW_cldice, cloudTauSW_snow )
            elseif(use_thompson_cloud_optics) then
              call compute_thompson_cloud_optics(nx, nzm, lat, layerMass, cloudFrac, &
                  cloudTauLW, cloudTauSW, cloudSsaSW, cloudAsmSW, cloudForSW, &
                  cloudTauSW_cldliq, cloudTauSW_cldice, cloudTauSW_snow )
            end if
            !
            ! Normally simulators are run only when the sun is up,
            !    but in case someone decides to use nighttime values...
            !
            if(doisccp .or. domodis .or. domisr) then
              ! band 9 is 625 - 778 nm, needed is 670 nm
              tau_067 (1:nx,lat,1:nzm) = cloudTauSW(9,1:nx,1:nzm)
              tau_067_cldliq (1:nx,lat,1:nzm) = cloudTauSW_cldliq(9,1:nx,1:nzm)
              tau_067_cldice (1:nx,lat,1:nzm) = cloudTauSW_cldice(9,1:nx,1:nzm)
              tau_067_snow (1:nx,lat,1:nzm) = cloudTauSW_snow(9,1:nx,1:nzm)
              ! band 6 is 820 - 980 cm-1, we need 10.5 micron
              emis_105(1:nx,lat,1:nzm) = 1. - exp(-cloudTauLW(6,1:nx,1:nzm))
            end if
          else
            liqRe(:,1:nzm) =   rad_reffc(:, lat, 1:nzm)
            iceRe(:,1:nzm) =   rad_reffi(:, lat, 1:nzm)
            !
            ! Limit particle sizes to range allowed by RRTMG parameterizations, add top layer
            !
            where(LWP(:,1:nzm) > 0.) & 
              liqRe(:,1:nzm) = max(2.5_kind_rb, min( 60._kind_rb,liqRe(:,1:nzm)))
            where(IWP(:,1:nzm) > 0.) & 
                 iceRe(:,1:nzm) = max(5.0_kind_rb, min(140._kind_rb,iceRe(:,1:nzm)))
                 
            liqRe(:,nzm+1) = 0._kind_rb
            iceRe(:,nzm+1) = 0._kind_rb
            
            if(doisccp .or. domodis .or. domisr) then 
              !
              ! Compute cloud optical depths directly so we can provide to instrument simulators
              !   Ice particle size should be "generalized effective size" from Fu et al. 1998
              !   doi:10.1175/1520-0442(1998)011<2223:AAPOTI>2.0.CO;2
              !   This would normally require some conversion, I guess
              !
              prpLWIn = 0.; prpSWIn = 0. 
              do i = 1, nx
                call cldprop   (nzm+1, 2, 3, 1, cloudFrac(i,:), prpLWIn, &
                                IWP(i,:), LWP(i,:), iceRe(i,:), liqRe(i,:), ncbands, tauLWOut)
                ! Last three output arguments from cldprop_sw are *delta-scaled* optical properties - 
                !   RRTM needs unscaled variables, so we need to provide physical quantities to RRTMG,
                ! which will call cldprop_sw again
                call cldprop_sw(nzm+1, 2, 3, 1, cloudFrac(i,:), &
                                prpSWIn, prpSWIn, prpSWIn, prpSWIn, IWP(i,:), LWP(i,:), iceRe(i,:), liqRe(i,:), &
                                tauSWOut, scaled1, scaled2, scaled3)
                tau_067 (i,lat,1:nzm) =           tauSWOut(1:nzm,24) ! RRTMG SW bands run from 16 to 29 (parrrsw.f90); we want 9th of these
                                                                     ! band 9 is 625 - 778 nm, needed is 670 nm
                emis_105(i,lat,1:nzm) = 1. - exp(-tauLWOut(1:nzm,6)) ! band 6 is 820 - 980 cm-1, we need 10.5 micron 
              end do
            end if

            if(domodis) then 
              !
              ! Compute separate cloud optical depths for liquid and ice clouds for input to 
              !   MODIS simulator, which wants these things separately.
              !
              cloudFrac_liq(:,:) = MERGE(1., 0., LWP(:,:)>0.)
              cloudFrac_ice(:,:) = MERGE(1., 0., IWP(:,:)>0.)
              prpLWIn = 0.; prpSWIn = 0.; dummyRe = 0.; dummyWP = 0.
              do i = 1, nx
                ! See above comment.  We want unscaled optical depth from cloud liquid at 670nm
                call cldprop_sw(nzm+1, 2, 3, 1, cloudFrac_liq(i,:), &
                                prpSWIn, prpSWIn, prpSWIn, prpSWIn, dummyWP(i,:), LWP(i,:), dummyRe(i,:), liqRe(i,:), &
                                tauSWOut, scaled1, scaled2, scaled3)
                tau_067_cldliq (i,lat,1:nzm) = tauSWOut(1:nzm,24) ! RRTMG SW band number 9 (625 - 778 nm), needed is 670 nm

                ! Same for cloud ice
                call cldprop_sw(nzm+1, 2, 3, 1, cloudFrac_ice(i,:), &
                                prpSWIn, prpSWIn, prpSWIn, prpSWIn, IWP(i,:), dummyWP(i,:), iceRe(i,:), dummyRe(i,:), &
                                tauSWOut, scaled1, scaled2, scaled3)
                tau_067_cldice (i,lat,1:nzm) = tauSWOut(1:nzm,24) ! RRTMG SW band number 9 (625 - 778 nm), needed is 670 nm

                tau_067_snow (i,lat,1:nzm) = 0.! snow is not radiatively active here.
              end do
            end if
          end if
          ! ---------------------------------------------------
          
          !
          ! Volume mixing fractions for gases.
          !bloss(072009): Note that o3, etc. are now in ppmv and don't need conversions.
          !
          h2ovmr(1:nx, 1:nzm)   = mwdry/mwh2o * qv(1:nx, lat, 1:nzm) 
          h2ovmr(1:nx, nzm+1)   = h2ovmr(1:nx, nzm) ! extrapolate above model top

          ! ---------------------------------------------------------------------------------
          if (dolongwave) then
            surfaceT(:) = sstxy(1:nx, lat) + t00
            
            idrv = 0
            duflx_dt(:,:) = 0.
            duflxc_dt(:,:) = 0.

            call t_startf ('radiation-lw')
            if(have_cloud_optics) then
              call rrtmg_lw (nx, nzm+1, overlap, idrv,            &
                layerP, interfaceP, layerT, interfaceT, surfaceT, &
                h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
                cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, surfaceEmissivity,  &
                0, 0, 0, cloudFrac, &
                CloudTauLW, IWP, LWP, iceRe, liqRe, &
                dummyTauAerosolLW, &
                lwUp,lwDown, lwHeatingRate, lwUpClearSky, lwDownClearSky, lwHeatingRateClearSky, &
                duflx_dt, duflxc_dt)
            else
              call rrtmg_lw (nx, nzm+1, overlap, idrv,            & 
                layerP, interfaceP, layerT, interfaceT, surfaceT, &
                h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
                cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, surfaceEmissivity,  &
                2, 3, 1, cloudFrac, &
                CloudTauLW, IWP, LWP, iceRe, liqRe, &
                dummyTauAerosolLW, &
                lwUp,lwDown, lwHeatingRate, lwUpClearSky, lwDownClearSky, lwHeatingRateClearSky, &
                duflx_dt, duflxc_dt)
            end if 
            
            !bloss: Recompute heating rate using layer density.
            !       This will provide better energy conservation since other
            !       flux difference terms in energy budget are computed this way.
            lwHeatingRate(:,1:nzm) = &
                 (lwUp(:,1:nzm) - lwUp(:,2:nz) + lwDown(:,2:nz) - lwDown(:,1:nzm)) &
                 /spread(cp*rho(1:nzm)*dz*adz(1:nzm), dim=1, ncopies=nx )
            lwHeatingRateClearSky(:,1:nzm) = &
                 (lwUpClearSky(:,1:nzm) - lwUpClearSky(:,2:nz) &
                  + lwDownClearSky(:,2:nz) - lwDownClearSky(:,1:nzm)) &
                 /spread(cp*rho(1:nzm)*dz*adz(1:nzm), dim=1, ncopies=nx )
            call t_stopf ('radiation-lw')
            !
            ! Add fluxes to average-average diagnostics
            !
            radlwup(:) = radlwup(:) + sum(lwUp(:, 1:nz),   dim = 1)
            radlwdn(:) = radlwdn(:) + sum(lwDown(:, 1:nz), dim = 1)
            radqrlw(1:nzm) = radqrlw(1:nzm) + sum(lwHeatingRate(:, 1:nzm), dim = 1)
            radqrclw(1:nzm) = radqrclw(1:nzm) + sum(lwHeatingRateClearSky(:, 1:nzm), dim = 1)
            qrad(:, lat, :) = lwHeatingRate(:, 1:nzm)
            !
            ! 2D diagnostic fields
            !
            lwDownSurface        (:, lat) = lwDown(:, 1)
            lwDownSurfaceClearSky(:, lat) = lwDownClearSky(:, 1)
            lwUpSurface          (:, lat) = lwUp(:, 1)
            lwUpSurfaceClearSky  (:, lat) = lwUpClearSky(:, 1)
            lwUpToa              (:, lat) = lwUp(:, nz+1) !bloss: nz+1 --> TOA
            lwUpTom              (:, lat) = lwUp(:, nz) 
            lwDownTom            (:, lat) = lwDown(:, nz) 
            lwUpToaClearSky      (:, lat) = lwUpClearSky(:, nz+1)
          end if 
         ! ---------------------------------------------------------------------------------
         
         ! ---------------------------------------------------------------------------------
         if(doshortwave) then
            !
            ! Solar insolation depends on several choices
            !
            !---------------
            if(doseasons) then 
              ! The diurnal cycle of insolation will vary
              ! according to time of year of the current day.
              dayForSW = day 
            else
              ! The diurnal cycle of insolation from the calendar
              ! day on which the simulation starts (day0) will be
              ! repeated throughout the simulation.
              dayForSW = int(day0) + day - int(day) 
            end if 
            !---------------
            if(doperpetual) then
               if (dosolarconstant) then
                  ! fix solar constant and zenith angle as specified
                  ! in prm file.
                  solarZenithAngleCos(:) = cos(zenith_angle * pi/180.)
                  eccf = solar_constant/(1367.)
               else
                  ! perpetual sun (no diurnal cycle) - Modeled after Tompkins
                  solarZenithAngleCos(:) = 0.637 ! equivalent to zenith angle of 50.5 deg
                   ! Adjst solar constant by mean value 
                  eccf = sum(p_factor(:, lat)/solarZenithAngleCos(:)) / real(nx) 
               end if
            else
               call shr_orb_decl (dayForSW, eccen, mvelpp, lambm0, obliqr, delta, eccf)
               solarZenithAngleCos(:) =  &
                 zenith(dayForSW, real(pi * latitude(:, lat)/180., kind_rb), &
                                  real(pi * longitude(:, lat)/180., kind_rb) )
            end if
            !---------------
            ! coszrs is found in params.f90 and used in the isccp simulator
            coszrs = max(0._kind_rb, solarZenithAngleCos(1))
            
            !
            ! We only call the shortwave if the sun is above the horizon. 
            !   We assume that the domain is small enough that the entire 
            !   thing is either lit or shaded
            !
            if(all(solarZenithAngleCos(:) >= tiny(solarZenithAngleCos))) then 

              if(lat.eq.1.AND.masterproc) print *, "Let's do some shortwave" 
              call albedo(ocean, solarZenithAngleCos(:), surfaceT, &
                          asdir(:), aldir(:), asdif(:), aldif(:))
              if(lat.eq.1.AND.masterproc) then
                print *, "Range of zenith angles", minval(solarZenithAngleCos), maxval(solarZenithAngleCos)
                print *, "Range of surface albedo (asdir)", minval(asdir), maxval(asdir)
                print *, "Range of surface albedo (aldir)", minval(aldir), maxval(aldir)
                print *, "Range of surface albedo (asdif)", minval(asdif), maxval(asdif)
                print *, "Range of surface albedo (aldif)", minval(aldif), maxval(aldif)
              end if

              call t_startf ('radiation-sw')
              if(have_cloud_optics) then
                call rrtmg_sw(nx, nzm+1, overlap,                     &
                  layerP, interfaceP, layerT, interfaceT, surfaceT, &
                  h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,     &
                  asdir, asdif, aldir, aldif, &
                  solarZenithAngleCos, eccf, 0, scon,   &
                  0, 0, 0, cloudFrac, &
                  cloudTauSW, cloudSsaSW, cloudAsmSW, cloudForSW, &
                  IWP, LWP, iceRe, liqRe,  &
                  dummyAerosolProps, dummyAerosolProps, dummyAerosolProps, dummyAerosolProps2, &
                  swUp, swDown, swHeatingRate, swUpClearSky, swDownClearSky, swHeatingRateClearSky)
              else 
                call rrtmg_sw(nx, nzm+1, overlap,                     & 
                  layerP, interfaceP, layerT, interfaceT, surfaceT, &
                  h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,     &
                  asdir, asdif, aldir, aldif, &
                  solarZenithAngleCos, eccf, 0, scon,   &
                  2, 3, 1, cloudFrac, &
                  cloudTauSW, cloudSsaSW, cloudAsmSW, cloudForSW, &
                  IWP, LWP, iceRe, liqRe,  &
                  dummyAerosolProps, dummyAerosolProps, dummyAerosolProps, dummyAerosolProps2, &
                  swUp, swDown, swHeatingRate, swUpClearSky, swDownClearSky, swHeatingRateClearSky)
              end if 
              !bloss: Recompute heating rate using layer density.
              !       This will provide better energy conservation since other
              !       flux difference terms in energy budget are computed this way.
              swHeatingRate(:,1:nzm) = &
                   (swUp(:,1:nzm) - swUp(:,2:nz) + swDown(:,2:nz) - swDown(:,1:nzm)) &
                   /spread(cp*rho(1:nzm)*dz*adz(1:nzm), dim=1, ncopies=nx )
              swHeatingRateClearSky(:,1:nzm) = &
                   (swUpClearSky(:,1:nzm) - swUpClearSky(:,2:nz) &
                   + swDownClearSky(:,2:nz) - swDownClearSky(:,1:nzm)) &
                   /spread(cp*rho(1:nzm)*dz*adz(1:nzm), dim=1, ncopies=nx )
              call t_stopf ('radiation-sw')
  
              !
              ! Add fluxes to average-average diagnostics
              !
              radswup(:) = radswup(:) + sum(swUp(:, 1:nz),   dim = 1)
              radswdn(:) = radswdn(:) + sum(swDown(:, 1:nz), dim = 1)
              radqrsw(:nzm) = radqrsw(:nzm) + sum(swHeatingRate(:, 1:nzm), dim = 1)
              radqrcsw(:nzm) = radqrcsw(:nzm) + sum(swHeatingRateClearSky(:, 1:nzm), dim = 1)
              qrad(:, lat, :) = qrad(:, lat, :) + swHeatingRate(:, 1:nzm)
              !
              ! 2D diagnostic fields
              !
              swDownSurface        (:, lat) = swDown(:, 1)
              swDownSurfaceClearSky(:, lat) = swDownClearSky(:, 1)
              swUpSurface          (:, lat) = swUp(:, 1)
              swUpSurfaceClearSky  (:, lat) = swUpClearSky(:, 1)
              swDownToa            (:, lat) = swDown(:, nz+1) !bloss: nz+1 --> TOA
              swDownTom            (:, lat) = swDown(:, nz) 
              swUpToa              (:, lat) = swUp(:, nz+1)
              swUpTom              (:, lat) = swUp(:, nz)
              swUpToaClearSky      (:, lat) = swUpClearSky(:, nz+1) 
              insolation_TOA       (:, lat) = swDown(:, nz+1)
            else
              !
              ! 2D diagnostic fields - nighttime values
              !
              swDownSurface        (:, lat) = 0.0
              swDownSurfaceClearSky(:, lat) = 0.0
              swUpSurface          (:, lat) = 0.0
              swUpSurfaceClearSky  (:, lat) = 0.0
              swDownToa            (:, lat) = 0.0
              swDownTom            (:, lat) = 0.0
              swUpToa              (:, lat) = 0.0
              swUpTom              (:, lat) = 0.0
              swUpToaClearSky      (:, lat) = 0.0
              insolation_TOA       (:, lat) = 0.0
            end if 
          end if 
          ! ---------------------------------------------------------------------------------
        end do ! Loop over y dimension
                
        !
        ! 2D diagnostics
        !
        nradsteps = 0 ! re-initialize nradsteps
        
        if(masterproc) then 
          if(doshortwave) then 
            if(doperpetual) then
              print *,'radiation: perpetual sun, solin=', sum(swDownToa(:, :)) / float(nx*ny)
            else
              print *,'radiation: coszrs=', coszrs,&
                      ' solin=', sum(swDownToa(:, :)) / float(nx*ny)
            end if
          end if
          if(dolongwave) print *,'longwave radiation is called'
        end if
        
        
        if(doradhomo) then    
          !
          ! Homogenize radiation if desired
          !
          radHeatingProfile(1:nzm) = sum(sum(qrad(:, :, :), dim = 1), dim = 1) / (nx * ny) 
          
          !
          ! Homogenize across the entire domain
          !
          if(dompi) then
            tempProfile(1:nzm) = radHeatingProfile(1:nzm)
            call task_sum_real8(radHeatingProfile, tempProfile, nzm)
            radHeatingProfile(1:nzm) = tempProfile(1:nzm) / real(nsubdomains)
          end if 
          
          qrad(:, 1, :) = spread(radHeatingProfile(1:nzm), dim = 1, ncopies = nx)
          if(ny > 1) &
            qrad(:, :, :) = spread(qrad(:, 1, :), dim = 2, ncopies = ny)
        end if
      end if ! nradsteps >= nrad
      
      !------------------------------------------------------------------------
      !
      ! Update 2d diagnostic fields 
      !
      ! Net surface and toa fluxes
      !
      ! First two for ocean evolution
      lwnsxy(:, :) = lwUpSurface(:, :) - lwDownSurface(:, :)  ! Net LW upwards
      swnsxy(:, :) = swDownSurface(:, :) - swUpSurface(:, :)  ! Net SW downwards
      
      lwns_xy(:, :) = lwns_xy(:, :) + &
                          lwUpSurface(:, :) - lwDownSurface(:, :)  ! Net LW upwards
      swns_xy(:, :) = swns_xy(:, :) + &
                          swDownSurface(:, :) - swUpSurface(:, :)  ! New SW downwards
      ! Net LW at Toa is upwards 
      lwnt_xy(:, :) = lwnt_xy(:, :) + lwUpToa(:, :) 
      ! Net SW at Toa  
      swnt_xy(:, :) = swnt_xy(:, :) + swDownToa(:, :) - swUpToa(:, :) 
     
      !
      ! Net surface and toa clear sky fluxes
      ! 
      lwnsc_xy(:, :) = lwnsc_xy(:, :) + &
                           lwUpSurfaceClearSky(:, :) - lwDownSurfaceClearSky(:, :) 
      swnsc_xy(:, :) = swnsc_xy(:, :) + &
                           swDownSurfaceClearSky(:, :) - swUpSurfaceClearSky(:, :)
      lwntc_xy(:, :) = lwntc_xy(:, :) +                           lwUpToaClearSky(:, :) 
      swntc_xy(:, :) = swntc_xy(:, :) + swDownToa(:, :) - swUpToaClearSky(:, :)
      
      ! TOA Insolation
      solin_xy(:, :) = solin_xy(:, :) + swDownToa(:, :) 
      
      
      !------------------------------------------------------------------------
      !
      ! Update 1D diagnostics
      ! 
    
      if(dostatisrad) then
        s_flns = s_flns + sum(lwUpSurface(:, :) - lwDownSurface(:, :)) ! lwnsxy
        s_fsns = s_fsns + sum(swDownSurface(:, :) - swUpSurface(:, :)) ! swnsxy
        s_flntoa = s_flntoa + sum(lwUpToa(:, :))                       ! lwntxy
        s_flnt = s_flnt + sum(lwUpTom(:, :) - lwDownTom(:, :))         ! lwntmxy
        s_fsntoa = s_fsntoa + sum(swDownToa(:, :) - swUpToa(:, :))         ! swntxy
        s_fsnt = s_fsnt + sum(swDownTom(:, :) - swUpTom(:, :))         ! swntxy
        s_flnsc = s_flnsc + &
          sum(lwUpSurfaceClearSky(:, :) - lwDownSurfaceClearSky(:, :)) ! lwnscxy
        s_fsnsc = s_fsnsc + &
          sum(swDownSurfaceClearSky(:, :) - swUpSurfaceClearSky(:, :)) ! swnscxy 
        s_flntoac = s_flntoac + &
          sum(lwUpToaClearSky(:, :))                                   ! lwntcxy 
        s_fsntoac = s_fsntoac + &
          sum(swDownToa(:, :) - swUpToaClearSky(:, :))                 ! swntcxy
        s_solin = s_solin + sum(swDownToa(:, :))                       ! solinxy 
        ! 
        ! I think the next two are supposed to be downwelling fluxes at the surface
        !
        s_fsds = s_fsds + sum(swDownSurface(:, :)) 
        s_flds = s_flds + sum(lwDownSurface(:, :)) 
      end if ! if(dostatis)
       
      if(mod(nstep,nstat*(1+nrestart_skip)).eq.0.or.nstep.eq.nstop.or.nelapse.eq.0) &
                 call write_rad() ! write radiation restart file
  
    end if ! if icycle == 1

    !
    ! Add radiative heating to liquid ice static energy variable
    !
    t(1:nx, 1:ny, 1:nzm) = t(1:nx, 1:ny, 1:nzm) + qrad(:, :, :) * dtn
  end subroutine rad_driver
  ! ----------------------------------------------------------------------------
  subroutine initialize_radiation

    implicit none

    real(KIND=kind_rb) :: cpdair

    !bloss  subroutine shr_orb_params
    !bloss  inputs:  iyear, log_print
    !bloss  ouptuts: eccen, obliq, mvelp, obliqr, lambm0, mvelpp
    
    call shr_orb_params(iyear, eccen, obliq, mvelp, obliqr, lambm0, mvelpp, .false.)
 
    ! sets up initial mixing ratios of trace gases.
    if(nrestart.eq.0) call tracesini()
 
    if(nrestart == 0) then
      qrad    (:, :, :) = 0.
      radlwup(:) = 0.
      radlwdn(:) = 0.
      radswup(:) = 0.
      radswdn(:) = 0.
      radqrlw(:) = 0.
      radqrsw(:) = 0.
      nradsteps = 0
    else
       call read_rad()
    endif
 
    if(doperpetual) then
      ! perpetual sun (no diurnal cycle)
      p_factor(:, :) = perpetual_factor(real(day0, kind_rb), &
                                        real(latitude(:, :), kind_rb), &
                                        real(longitude(:, :), kind_rb) )
    end if
 
    cpdair = cp
    call rrtmg_sw_ini(cpdair)
    call rrtmg_lw_ini(cpdair)
    
    if(trim(micro_scheme_name()) == 'm2005' .and. & 
       (compute_reffc .or. compute_reffi) .and. &
       (.NOT.dorrtm_cloud_optics_from_effrad_LegacyOption)) then
       call m2005_cloud_optics_init
       use_m2005_cloud_optics = .true.
       have_cloud_optics = .true.
    end if

    if(trim(micro_scheme_name()) == 'thompson' .and. &
       (compute_reffc .or. compute_reffi) .and. &
       (.NOT.dorrtm_cloud_optics_from_effrad_LegacyOption)) then
       call thompson_cloud_optics_init
       use_thompson_cloud_optics = .true.
       have_cloud_optics = .true.
    end if

    land_frac = MERGE(0., 1., ocean)
    initialized = .true.
  end subroutine initialize_radiation
  ! ----------------------------------------------------------------------------
  !
  ! Trace gas profiles
  !
  ! ----------------------------------------------------------------------------
  subroutine tracesini()
    use netcdf

    implicit none
    !
    ! Initialize trace gaz vertical profiles
    !   The files read from the top down 
    !
    !bloss(072009): Get trace gas profiles from rrtmg_lw.nc, the data
    !                 file provided with RRTMG.  These are indexed from
    !                 bottom to top and are in ppmv, so that no conversion
    !                 is needed for use with RRTMG.
    !
    integer k, m, ierr
    real :: godp ! gravity over delta pressure
    real :: plow, pupp
    real(kind=kind_rb), dimension(nzm+1) :: tmp_pres ! level pressure with extra level at TOA
    real(kind=kind_rb), dimension(nz+1) :: tmp_presi ! interface pressure with extra level at TOA
    integer(kind=kind_im) :: ncid, varID, dimIDab, dimIDp

    integer(kind=kind_im) :: Nab, nPress, ab
    real(kind=kind_rb) :: wgtlow, wgtupp, pmid
    real(kind=kind_rb), allocatable, dimension(:) :: pMLS
    real(kind=kind_rb), allocatable, dimension(:,:) :: trace, trace_in
    character(LEN=nf90_max_name) :: tmpName

    integer, parameter :: nTraceGases = 9
    real(kind=kind_rb), dimension(nzm+1) :: tmpTrace
    real(kind=kind_rb), dimension(nz+1,nTraceGases) :: trpath
    character(len = maxAbsorberNameLength), dimension(nTraceGases), parameter :: &
         TraceGasNameOrder = (/        &
     				'O3   ',  &
     				'CO2  ',  &
     				'CH4  ',  &
     				'N2O  ',  & 
     				'O2   ',  &
     				'CFC11',  &
     				'CFC12',  &
     				'CFC22',  &
     				'CCL4 '  /)
	
    real factor
    ! ---------------------------------

    !bloss: RRTMG radiation orders levels from surface to top of model.
    !       This routine was originally written for CCM/CAM radiation
    !       which orders levels from top down.  As a result, we need to
    !       reverse the ordering here to make things work for RRTMG.

!DD add code to read from master only and bcast trace profiles
  if(masterproc) then

    ! Read profiles from rrtmg data file.
    status(:)   = nf90_NoErr
    status(1)   = nf90_open(trim(rundatadir)//'/rrtmg_lw.nc',nf90_nowrite,ncid)
	
    status(2)   = nf90_inq_dimid(ncid,"Pressure",dimIDp)
    status(3)   = nf90_inquire_dimension(ncid, dimIDp, tmpName, nPress)

    status(4)   = nf90_inq_dimid(ncid,"Absorber",dimIDab)
    status(5)   = nf90_inquire_dimension(ncid, dimIDab, tmpName, Nab)

    allocate(pMLS(nPress), trace(nTraceGases,nPress), trace_in(Nab,nPress), STAT=ierr)
    pMLS = 0.
    trace = 0.
    if(ierr.ne.0) then
      write(*,*) 'ERROR: could not declare arrays in tracesini'
      call task_abort()
    end if

    status(6)   = nf90_inq_varid(ncid,"Pressure",varID)
    status(7)   = nf90_get_var(ncid, varID, pMLS)

    status(8)   = nf90_inq_varid(ncid,"AbsorberAmountMLS",varID)
    status(9) = nf90_get_var(ncid, varID, trace_in)

    do m = 1,nTraceGases
      call getAbsorberIndex(TRIM(tracegasNameOrder(m)),ab)
      trace(m,1:nPress) = trace_in(ab,1:nPress)
      where (trace(m,:)>2.)
        trace(m,:) = 0.
      end where
    end do

    if(MAXVAL(ABS(status(1:8+nTraceGases))).ne.0) then
      write(*,*) 'Error in reading trace gas sounding from'//trim(rundatadir)//'/rrtmg_lw.nc'
      call task_abort()
    end if

    !bloss(120409): copy level, interface pressures into local variable.
    tmp_pres(1:nzm) = pres(1:nzm)
    tmp_presi(1:nz) = presi(1:nz)

    ! Add level at top of atmosphere (top interface <= 0.01 Pa)
    tmp_pres(nz) = 0.5*presi(nz)
    tmp_presi(nz+1) = MIN(1.e-4_kind_rb, 0.25*tmp_pres(nz))

    !bloss: modify routine to compute trace gas paths from surface to
    ! top of atmosphere.  Then, interpolate these paths onto the
    !  interface pressure levels of the model grid, with an extra level
    !  between the model top and the top of atmosphere.  Differencing these
    !   paths and dividing by dp/g will give the mean mass concentration
    !    in that level.
    !
    !  This procedure has the advantage that the total trace gas path
    !   will be invariant to changes in the vertical grid.

    ! trace gas paths at surface are zero.
    trpath(1,:) = 0.

    do k = 2,nz+1
      ! start with trace path at interface below.
      trpath(k,:) = trpath(k-1,:)

      ! if pressure greater than sounding, assume concentration at bottom.
      if (tmp_presi(k-1).gt.pMLS(1)) then
        trpath(k,:) = trpath(k,:) &
             + (tmp_presi(k-1) - MAX(tmp_presi(k),pMLS(1)))/ggr & ! dp/g
             *trace(:,1)                                 ! *tr
      end if

      do m = 2,nPress
        ! limit pMLS(m:m-1) so that they are within the model level
        !  tmp_presi(k-1:k).
        plow = MIN(tmp_presi(k-1),MAX(tmp_presi(k),pMLS(m-1)))
        pupp = MIN(tmp_presi(k-1),MAX(tmp_presi(k),pMLS(m)))

        if(plow.gt.pupp) then
          pmid = 0.5*(plow+pupp)

          wgtlow = (pmid-pMLS(m))/(pMLS(m-1)-pMLS(m))
          wgtupp = (pMLS(m-1)-pmid)/(pMLS(m-1)-pMLS(m))
!!$          write(*,*) pMLS(m-1),pmid,pMLS(m),wgtlow,wgtupp
          ! include this level of the sounding in the trace gas path
          trpath(k,:) = trpath(k,:) &
               + (plow - pupp)/ggr*(wgtlow*trace(:,m-1)+wgtupp*trace(:,m)) ! dp/g*tr
        end if
      end do

      ! if pressure is off top of trace gas sounding, assume
      !  concentration at top
      if (tmp_presi(k).lt.pMLS(nPress)) then
        trpath(k,:) = trpath(k,:) &
             + (MIN(tmp_presi(k-1),pMLS(nPress)) - tmp_presi(k))/ggr & ! dp/g
             *trace(:,nPress)                               ! *tr
      end if

    end do

    if(notracegases) then
     factor=0.01  ! don't make compte zeros as it may blow up the code
    else
     factor=1.
    end if
      

    do m = 1,nTraceGases
      do k = 1,nzm+1
        godp = ggr/(tmp_presi(k) - tmp_presi(k+1))
        tmpTrace(k) = (trpath(k+1,m) - trpath(k,m))*godp
      end do
      if(TRIM(TraceGasNameOrder(m))=='O3') then
        o3(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='CO2') then
        co2(1:nzm+1) = tmpTrace(1:nzm+1)*nxco2
      elseif(TRIM(TraceGasNameOrder(m))=='CH4') then
        ch4(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='N2O') then
        n2o(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='O2') then
        o2(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='CFC11') then
        cfc11(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='CFC12') then
        cfc12(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='CFC22') then
        cfc22(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      elseif(TRIM(TraceGasNameOrder(m))=='CCL4') then
        ccl4(1:nzm+1) = tmpTrace(1:nzm+1)*factor
      end if
    
    end do

      print*,'RRTMG rrtmg_lw.nc trace gas profile: number of levels=',nPress
      print*,'gas traces vertical profiles (ppmv):'
      print*,' p (hPa) ', ('       ',TraceGasNameOrder(m),m=1,nTraceGases)
      do k=1,nzm+1
        write(*,999) tmp_pres(k),o3(k),co2(k),ch4(k),n2o(k),o2(k), &
             cfc11(k),cfc12(k), cfc22(k),ccl4(k)
999     format(f8.2,12e12.4)
      end do
      print*,'done...'

    deallocate(pMLS, trace, STAT=ierr)
    endif
    
    if(dompi) then
      call task_bcast_real8(0,o3,nzm+1)
      call task_bcast_real8(0,co2,nzm+1)
      call task_bcast_real8(0,ch4,nzm+1)
      call task_bcast_real8(0,n2o,nzm+1)
      call task_bcast_real8(0,o2,nzm+1)
      call task_bcast_real8(0,cfc11,nzm+1)
      call task_bcast_real8(0,cfc12,nzm+1)
      call task_bcast_real8(0,cfc22,nzm+1)
      call task_bcast_real8(0,ccl4,nzm+1)
    end if

  end subroutine tracesini
  ! ----------------------------------------------------------------------------
  !
  ! Astronomy-related procedures 
  ! 
  ! ----------------------------------------------------------------------------
  elemental real(kind_rb) function zenith(calday, clat, clon)
     implicit none
     real(kind_rb), intent(in ) :: calday, & ! Calendar day, including fraction
                                   clat,   & ! Current centered latitude (radians)
                                   clon      ! Centered longitude (radians)

     real(kind_rb)     :: delta, & ! Solar declination angle in radians
                          eccf
     integer  :: i     ! Position loop index

     call shr_orb_decl (calday, eccen, mvelpp, lambm0, obliqr, delta, eccf)
     !
     ! Compute local cosine solar zenith angle
     !
     zenith = shr_orb_cosz(calday, clat, clon, delta)
  end function zenith
  ! ----------------------------------------------------------------------------
  elemental real(kind_rb) function perpetual_factor(day, lat, lon)
    implicit none
    real(kind_rb), intent(in) :: day, lat, lon ! Day (without fraction); centered lat/lon (degrees) 
    real(kind_rb)     :: delta, & ! Solar declination angle in radians
                         eccf
    
    !  estimate the factor to multiply the solar constant
    !  so that the sun hanging perpetually right above
    !  the head (zenith angle=0) would produce the same
    !  total input the TOA as the sun subgect to diurnal cycle.
    !  coded by Marat Khairoutdinov, 2004
    
    ! Local:
    real(kind_rb) :: tmp
    real(kind_rb) :: dttime 
    real(kind_rb) :: coszrs, ttime
    real(kind_rb) :: clat, clon
    
    ttime = day
    dttime = dt*float(nrad)/86400.
    tmp = 0.
    
    clat = pi * lat/180.
    clon = pi * lon/180.
    do while (ttime.lt.day+1.)
      call shr_orb_decl (ttime, eccen, mvelpp, lambm0, obliqr, delta, eccf)
       coszrs = zenith(ttime, clat, clon)
       tmp = tmp + min(dttime, day+1. - ttime)*max(0._kind_rb, eccf * coszrs)
       ttime = ttime+dttime
    end do
    
    perpetual_factor = tmp
    
  end function perpetual_factor
  ! ----------------------------------------------------------------------------
  !
  ! Writing and reading binary restart files
  !
  ! ----------------------------------------------------------------------------
  subroutine write_rad()
    integer :: irank, ii

    !bloss: added a bunch of statistics-related stuff to the restart file
    !         to nicely handle the rare case when nrad exceeds nstat and 
    !         the model restarts with mod(nstep,nrad)~=0.  This would cause
    !         many of the radiation statistics to be zero before the next
    !         multiple of nrad.

    if(masterproc) print*,'Writting radiation restart file...'

    if(restart_sep) then
      open(56, file = trim(constructRestartFileName(case, caseId, rank)), &
           status='unknown',form='unformatted')
      write(56) nsubdomains
	  write(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, &
        radqrlw, radqrsw, radqrclw, radqrcsw, &
        lwDownSurface, lwDownSurfaceClearSky, lwUpSurface, lwUpSurfaceClearSky, & 
        lwUpToa,     lwUpToaClearSky,     &
        swDownSurface, swDownSurfaceClearSky, swUpSurface, swUpSurfaceClearSky, & 
        swDownToa,                            swUpToa,     swUpToaClearSky,     &
        insolation_TOA, &
          o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
      close(56)
    else
      do irank = 0, nsubdomains-1
        call task_barrier()
        if(irank == rank) then
          open(56, file = trim(constructRestartFileName(case, caseId, nSubdomains)), &
               status='unknown',form='unformatted')
          if(masterproc) then
            write(56) nsubdomains
          else
            read (56)
            do ii=0,irank-1 ! skip records
              read(56)
            end do
          end if
          write(56) nradsteps, qrad , radlwup, radlwdn, radswup, radswdn, &
               radqrlw, radqrsw, radqrclw, radqrcsw, &
               lwDownSurface, lwDownSurfaceClearSky, lwUpSurface, lwUpSurfaceClearSky, & 
               lwUpToa,     lwUpToaClearSky,     &
               swDownSurface, swDownSurfaceClearSky, swUpSurface, swUpSurfaceClearSky, & 
               swDownToa,                            swUpToa,     swUpToaClearSky,     &
               insolation_TOA, &
          o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
          close(56)
        end if
      end do

    end if ! restart_sep

	if(masterproc) print *,'Saved radiation restart file. nstep=',nstep
    call task_barrier()
  end subroutine write_rad
  ! ----------------------------------------------------------------------------
  subroutine read_rad()
    integer ::  irank, ii

    if(masterproc) print*,'Reading radiation restart file...'

    if(restart_sep) then
    
      if(nrestart.ne.2) then
        open(56, file = trim(constructRestartFileName(case, caseid, rank)), &
             status='unknown',form='unformatted')
      else
        open(56, file = trim(constructRestartFileName(case_restart, caseid_restart, rank)), &
             status='unknown',form='unformatted')
      end if
      read (56)
      read(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, &
        radqrlw, radqrsw, radqrclw, radqrcsw, &
        lwDownSurface, lwDownSurfaceClearSky, lwUpSurface, lwUpSurfaceClearSky, & 
        lwUpToa,     lwUpToaClearSky,     &
        swDownSurface, swDownSurfaceClearSky, swUpSurface, swUpSurfaceClearSky, & 
        swDownToa,                            swUpToa,     swUpToaClearSky,     &
        insolation_TOA, &
          o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
      close(56)
      
    else
    
      do irank=0,nsubdomains-1
        call task_barrier()
        if(irank == rank) then
          if(nrestart.ne.2) then
            open(56, file = trim(constructRestartFileName(case, caseId, nSubdomains)), &
                 status='unknown',form='unformatted')
          else
            open(56, file = trim(constructRestartFileName(case, caseId_restart, nSubdomains)), &
                 status='unknown',form='unformatted')
          end if
          read (56)
          do ii=0,irank-1 ! skip records
             read(56)
          end do
          read(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, &
               radqrlw, radqrsw, radqrclw, radqrcsw, &
               lwDownSurface, lwDownSurfaceClearSky, lwUpSurface, lwUpSurfaceClearSky, & 
               lwUpToa,     lwUpToaClearSky,     &
               swDownSurface, swDownSurfaceClearSky, swUpSurface, swUpSurfaceClearSky, & 
               swDownToa,                            swUpToa,     swUpToaClearSky,     &
               insolation_TOA, &
          o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
          close(56)
        end if
      end do
      
    end if ! restart_sep
    
    if(rank == nsubdomains-1) then
         print *,'Case:',caseid
         print *,'Restart radiation at step:',nstep
         print *,'Time:',nstep*dt
    endif
    
    call task_barrier()
  end subroutine read_rad      
  
  ! ----------------------------------------------------------------------------
  function constructRestartFileName(case, caseid, index) result(name) 
    character(len = *), intent(in) :: case, caseid
    integer,            intent(in) :: index
    character(len=256) :: name
    
    character(len=4) :: indexChar

    integer, external :: lenstr

    write(indexChar,'(i4)') index

    name = './RESTART/' // trim(case) //'_'// trim(caseid) //'_'// &
              indexChar(5-lenstr(indexChar):4) //'_restart_rad.bin'
!bloss              trim(indexChar) //'_restart_rad.bin'

  end function constructRestartFileName
  ! ----------------------------------------------------------------------------
end module rad
