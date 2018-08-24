MODULE rad_driver
	! --------------------------------------------------------------------------
	!
	! Interface to RRTM longwave and shortwave radiation code.
	!   Robert Pincus, November 2007
	!
	! Modified by Peter Blossey, July 2009.
	!   - interfaced to RRTMG LW v4.8 and SW v3.8.
	!   - reversed indices in tracesini to match bottom-up RRTM indexing.
	!   - fixed issue w/doperpetual caused by eccf=0.
	!   - added extra layer in calls to RRTM to improve heating rates at model top.
	!        Required changes to inatm in rrtmg_lw_rad.f90 and inatm_sw in rrtmg_sw_rad.f90.
	!   - fixed bug that would zero out cloudFrac if LWP>0 and IWP==0.
	!   - changed definition of interfaceT(:,1) to SST.
	!   - added o3, etc. profiles to restart file.  Only call tracesini if nrestart==0.
	!
	! Modified by Peter Blossey, August 2009.
	! Changes for CFMIP intercomparison:
	!   - converted rad_driver into a routine with input/output arguments for portability of
	!        implementation.
	!
	! Modified by Tak Yamaguchi, March 2013.
	!   - added an option to use the effective radius computed from microphysics.f90
	!   - disabled options: dovolume_mean_effective_radius, dolognormal_effective_radius.
	!   - created write_rad and read_rad.
	!   - kind_rb is applied to REALs except REALs with SAM's real kind. Conversions are explicitly
	!     stated for arithmetics.
	!
	! Comments by Tak Yamaguchi, October 2016.
	!   - Since the version of RRTMG previousely used for RAD_RRTM_PBL is slightly older version than
	!     RAD_RRTM, I changed it to the same version of RAD_RRTM.
	!   - The have_cloud_optics option introduced to SAM6.10.10 has not been inplemented yet. Also,
	!     calculations for satellite simulator. See lines between 363 and 447 in RAD_RRTM/rad.f90.
	!
	! Modified by Peter Blossey, July 2017
        !   - Merging Tak's version of my old RAD_RRTM_CFMIP code into our repository for SAM.
        !   - Consolidate USE statements at top of module.
        !   - Eliminate the SAMTAK ifdefs and include Tak's simple aerosol treatment by default.
        !   - Include aerosol flag iaer as an input to rrtmg_sw_rad, controlled by namelist variables.
        !   - 

	! --------------------------------------------------------------------------
        use parkind, only : kind_rb, kind_im 
        use params, only : doradaerosimple, nxco2, notracegases !bloss: added co2/trace gas options
        use grid, only : nx, compute_reffc, compute_reffi, &
             dt, nstep, nrestart, &
             dz, adz, masterproc, nsubdomains,          &
             case, &
             doisccp, domodis, domisr, &
             restart_sep, caseid, case_restart, caseid_restart, rank
        USE vars, ONLY: radqrlw, radqrsw, radlwup, radlwdn, radswup, radswdn

        USE rad, ONLY: nradsteps, qrad, radqrclw, radqrcsw, &
             NetlwUpSurface, NetlwUpSurfaceClearSky, NetlwUpToa, NetlwUpToaClearSky, &
             NetswDownSurface, NetswDownSurfaceClearSky, NetswDownToa, NetswDownToaClearSky, &
             insolation_TOA, swDownSurface, lwDownSurface, &
             npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, &
             tau_067, emis_105, rad_reffc, rad_reffi, &  ! For instrument simulators
             tau_067_cldliq, tau_067_cldice, tau_067_snow ! for MODIS simulator: wants individual phases

        !
        ! Radiation solvers
        !
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
        use cam_rad_parameterizations, only : albedo
        use microphysics, only : micro_scheme_name, reffc, reffi, &
             dorrtm_cloud_optics_from_effrad_LegacyOption
        use m2005_cloud_optics, &
             only : m2005_cloud_optics_init, compute_m2005_cloud_optics
        use thompson_cloud_optics, &
             only : thompson_cloud_optics_init, compute_thompson_cloud_optics

        use shr_orb_mod, only: shr_orb_params, shr_orb_decl, shr_orb_cosz


	IMPLICIT NONE
	PRIVATE
	
	! Public procedures
	PUBLIC :: rad_driver_rrtm, initialize_radiation, isInitialized_RadDriver, tracesini, &
		tracesupdate, read_rad, write_rad, land_frac
	
	! Constants
	REAL(KIND=kind_rb), PARAMETER :: Pi = 3.14159265358979312
	REAL(KIND=kind_rb), PARAMETER :: scon = 1367. ! solar constant
	
	! Molecular weights (taken from CAM shrc_const_mod.F90 and physconst.F90)
	REAL(KIND=kind_rb), PARAMETER :: mwdry =  28.966, & ! molecular weight dry air
	                   mwco2 =  44.,    & ! molecular weight co2
	                   mwh2o =  18.016, & ! molecular weight h2o
	                   mwn2o =  44.,    & ! molecular weight n2o
	                   mwch4 =  16.,    & ! molecular weight ch4
	                   mwf11 = 136.,    & ! molecular weight cfc11
	                   mwf12 = 120.,    & ! molecular weight cfc12
	                   mwo3  =  48.       ! ozone, strangely missing
	! mixingRatioMass = mol_weight/mol_weight_air * mixingRatioVolume
	
	! Global storage
	LOGICAL, SAVE :: isInitialized_RadDriver = .FALSE.
        logical :: use_m2005_cloud_optics = .false., &
             use_thompson_cloud_optics = .false., have_cloud_optics = .false.
	REAL(KIND=kind_rb) :: land_frac = 1.0
	
	!bloss(072009): changed from mass mixing ratios to volume mixing ratios
	!               because we're now using rrtmg_lw.nc sounding for trace gases.
	! Profiles of gas volume mixing ratios
	LOGICAL, SAVE :: isallocated_tracegases = .FALSE.
	INTEGER, SAVE :: nz_tracegases = -1
	REAL(KIND=kind_rb), DIMENSION(:), ALLOCATABLE, SAVE :: &
		o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
	
	REAL(KIND=kind_rb), SAVE :: p_factor, p_coszrs ! perpetual-sun factor, cosine zenith angle
	
	! Earth's orbital characteristics
	!   Calculated in shr_orb_mod which is called by rad_driver_rrtm
	REAL(KIND=kind_rb), SAVE ::  &
		eccf,  & ! eccentricity factor (1./earth-sun dist^2)
		eccen, & ! Earth's eccentricity factor (unitless) (typically 0 to 0.1)
		obliq, & ! Earth's obliquity angle (deg) (-90 to +90) (typically 22-26)
		mvelp, & ! Earth's moving vernal equinox at perhelion (deg)(0 to 360.0)
		!
		! Orbital information after processed by orbit_params
		!
		obliqr, &  ! Earth's obliquity in radians
		lambm0, &  ! Mean longitude of perihelion at the vernal equinox (radians)
		mvelpp  ! Earth's moving vernal equinox longitude of perihelion plus pi (radians)
	
        ! option for enabling aerosol treatment in RRTMG SW
        !   Note that LW aerosol treatment is already enabled but TauAerosolLW needs to
        !   be set for aerosols to have radiative impact.
        INTEGER, SAVE :: iaer_rrtmg = 0 

! Tak Yamaguchi 2015/10: RADAEROS - Simple absorbing aerosol
	REAL, ALLOCATABLE, DIMENSION(:), SAVE :: rh_lw, rh_sw ! relative humidity bin (bin center value)
	REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:,:), SAVE :: ext_coef_lw ! extinction coefficient
	REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:,:), SAVE :: ext_coef_sw ! extinction coefficient
	REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:,:), SAVE :: ssalbdo  ! single scattering albedo
	REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:,:), SAVE :: asym_fr  ! asymmetry parameter

CONTAINS

	! --------------------------------------------------------------------------
	SUBROUTINE rad_driver_rrtm( nx, nzm, nzm_dyn, lat, pres, presi, &
                tabs, qv, qcl, qci, rel, rei, tg, nca, & ! Tak Yamaguchi 2015/10: RADAEROS
		dolongwave, doshortwave, doperpetual, doseasons, &
		dosolarconstant, solar_constant, zenith_angle, &
		day, day0, latitude, longitude, &
		ocean, ggr, cp, masterproc, &
		lwUp, lwDown, lwUpClearSky, lwDownClearSky, &
		swUp, swDown, swUpClearSky, swDownClearSky, coszrs )
		
		IMPLICIT NONE
		
		! Inputs
		INTEGER, INTENT(IN) :: nx  ! number of columns for which radiation will be computed
		INTEGER, INTENT(IN) :: nzm ! number of model levels for radiation in each column.
		INTEGER, INTENT(IN) :: nzm_dyn ! number of model levels for dynamics in each column.
		INTEGER, INTENT(IN) :: lat ! index of coordinate in y-direction (from 1 to ny).
		LOGICAL, INTENT(IN) :: masterproc ! .true. if MPI rank==0 (or single-processor run)
		
		LOGICAL, INTENT(IN) :: dolongwave ! compute longwave radiation
		LOGICAL, INTENT(IN) :: doshortwave ! compute shortwave radiation
		LOGICAL, INTENT(IN) :: doperpetual ! use perpetual (diurnally-averaged) insolation
		LOGICAL, INTENT(IN) :: doseasons ! allow diurnally-varying insolation to vary with time of year
		
		LOGICAL, INTENT(IN) :: dosolarconstant ! allow insolation and zenith angle to be specified if doperpetual==.true.
		! (TAK) These REALs hold SAM's real kind.
		REAL, INTENT(IN) :: solar_constant ! mean insolation if doperpetual==dosolarconstant==.true.
		REAL, INTENT(IN) :: zenith_angle ! solar zenith angle if doperpetual==dosolarconstant==.true.
		
		REAL, INTENT(IN) :: day ! day of year during iyear (0.0 = 00Z Jan 1)
		REAL, INTENT(IN) :: day0 ! starting day of year for run
		REAL, INTENT(IN) :: latitude ! latitude
		REAL, INTENT(IN) :: longitude ! longitude
		LOGICAL, INTENT(IN) :: ocean ! .true. if ocean surface, .false. if land
		REAL, INTENT(IN) :: ggr ! gravitational acceleration (~9.8 m/s2)
		REAL, INTENT(IN) :: cp ! specific heat of dry air at constant pressure at 273 K in J/kg/K
		
		REAL, INTENT(IN) :: pres(nzm) ! pressure (mb) at center of model levels.
		REAL, INTENT(IN) :: presi(nzm+1) ! pressure (mb) at model interfaces.
		REAL, INTENT(IN) :: tabs(nx,nzm) ! absolute temperature (K) at model levels
		REAL, INTENT(IN) :: qv(nx,nzm) ! water vapor mass mixing ratio (kg/kg) 
		REAL, INTENT(IN) :: qcl(nx,nzm) ! cloud liquid mass mixing ratio (kg/kg) 
		REAL, INTENT(IN) :: qci(nx,nzm) ! cloud ice mass mixing ratio (kg/kg)
		REAL, INTENT(IN) :: rel(nx,nzm) ! (TAK) cloud water effective radius, microns
		REAL, INTENT(IN) :: rei(nx,nzm) ! (TAK) cloud ice effective radius, microns
		REAL, INTENT(IN) :: tg(nx) ! ground (or sea surface) temperature (K)

! Tak Yamaguchi 2015/10: RADAEROS
		REAL, INTENT(IN) :: nca(nx,nzm) ! aerosol number concentration (#/cm3)

		! Outputs: Note that fluxes are located at interfaces and have an extra level.
		!          The top fluxes, e.g. lwup(:,nzm+2), are approximate top-of-atmosphere fluxes.
		!          lwup(:,nzm+1) is the top-of-model flux.
		REAL(kind=kind_rb), INTENT(OUT) :: &
			lwUp(nx,nzm+2) , & ! upward longwave radiative flux (W/m2)
			lwDown(nx,nzm+2) , & ! downward longwave radiative flux (W/m2)
			lwUpClearSky(nx,nzm+2) , & ! clearsky upward longwave radiative flux (W/m2)
			lwDownClearSky(nx,nzm+2) , & ! clearsky downward longwave radiative flux (W/m2)
			swUp(nx,nzm+2) , & ! upward shortwave radiative flux (W/m2)
			swDown(nx,nzm+2) , & ! downward shortwave radiative flux (W/m2)
			swUpClearSky(nx,nzm+2) , & ! clearsky upward shortwave radiative flux (W/m2)
			swDownClearSky(nx,nzm+2) ! clearsky downward shortwave radiative flux (W/m2)
			
		REAL, INTENT(OUT) :: coszrs ! (TAK) SAM's real kind
		
		! Local variables
		!
		! Input and output variables for RRTMG SW and LW
		!   RRTM specifies the kind of real variables in
		!   Only one column dimension is allowed parkind
		!   RRTM is indexed from bottom to top
                !
                !bloss: add layer to top to improve top-of-model heating rates.
                REAL(KIND = kind_rb), dimension(nx, nzm+1) ::     &
                     layerP,     layerT, layerMass,         & ! layer mass is for convenience
                     h2ovmr,   o3vmr,    co2vmr,   ch4vmr, n2ovmr,  & ! Volume mixing ratios for H2O, O3, CH4, N20, CFCs
                     o2vmr, cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, &
                     swHeatingRate, swHeatingRateClearSky,  &
                     lwHeatingRate, lwHeatingRateClearSky, &
                     duflx_dt, duflxc_dt

                ! cloud physical/optical properties needed by the RRTM internal parmaertizations
                REAL(KIND = kind_rb), dimension(nx, nzm+1) ::     &
                     LWP, IWP, liqRe, iceRe, cloudFrac             ! liquid/ice water path (g/m2) and size (microns)
                REAL(KIND = kind_rb), dimension(nbndlw, nx, nzm+2) :: cloudTauLW 
                REAL(KIND = kind_rb), dimension(nbndsw, nx, nzm+2) :: &
                     cloudTauSW, cloudSsaSW, & ! SW optical depth, SW single scattering albedo
                     cloudAsmSW, cloudForSW, & ! SW asymmetry and forward scattering factors
                     cloudTauSW_cldliq, & ! SW optical depth partitioned by hydrometeor
                     cloudTauSW_cldice, &
                     cloudTauSW_snow

                INTEGER :: inflgsw, iceflgsw, liqflgsw, &
                     inflglw, iceflglw, liqflglw

                ! Aerosol properties for input to RRTM routines
                ! Tak Yamaguchi 2015/10: RADAEROS
		REAL(KIND=kind_rb), DIMENSION(nx, nzm+1, nbndlw) :: TauAerosolLW
		REAL(KIND=kind_rb), DIMENSION(nx, nzm+1, nbndsw) :: TauAerosolSW ! optical depth
		REAL(KIND=kind_rb), DIMENSION(nx, nzm+1, nbndsw) :: SsaAerosolSW ! single scattering albedo
		REAL(KIND=kind_rb), DIMENSION(nx, nzm+1, nbndsw) :: AsmAerosolSW ! aymmetry parameter

		REAL(KIND=kind_rb), DIMENSION(nx, nzm+1, naerec) :: AerosolProps2
		
                ! Arguments to RRTMG cloud optical depth routines
                REAL(KIND = kind_rb), dimension(nbndlw, nzm+1) :: prpLWIn
                REAL(KIND = kind_rb), dimension(nzm+1, nbndlw) :: tauLWOut
                REAL(KIND = kind_rb), dimension(nbndsw, nzm+1) :: prpSWIn
                REAL(KIND = kind_rb), dimension(nzm+1, jpband) :: tauSWOut, scaled1, scaled2, scaled3 
                INTEGER                                        :: ncbands  
                
                !bloss: add layer to top to improve top-of-model heating rates.
                real(kind = kind_rb), dimension(nx, nzm+2)  :: interfaceP, interfaceT

                real(kind = kind_rb), dimension(nx) :: surfaceT, solarZenithAngleCos
                ! Surface direct/diffuse albedos for 0.2-0.7 (s) 
                !   and 0.7-5.0 (l) microns
                real(kind = kind_rb), dimension(nx) ::  asdir, asdif, aldir, aldif 

                real(kind = kind_rb), dimension(nx, nbndlw) :: surfaceEmissivity
                integer :: overlap = 1
		INTEGER :: idrv = 0 ! option to have rrtm compute d(OLR)/dTABS for both full and clear sky
		
		INTEGER :: i, k
		REAL(KIND=kind_rb) :: dayForSW, delta

                !bloss: extra arrays for handling liquid-only and ice-only cloud optical depth
                !   computation for MODIS simulator
                real(kind = kind_rb), dimension(nx, nzm+1) ::     &
                     dummyWP, dummyRe, cloudFrac_liq, cloudFrac_ice ! liquid/ice water path (g/m2) and size (microns)

                ! ----------------------------------------------------------------------------
		
                surfaceEmissivity(:,:) = 0.95  ! Default Value For Sea Surface emissivity

		! Initialize coszrs to non-physical value as a placeholder
		coszrs = -2.0
		
                ! set trace gas concentrations.  Assumed to be uniform in the horizontal.
		o3vmr(:, 1:nzm+1)    = SPREAD(o3(:), DIM=1, NCOPIES=nx)
		co2vmr(:, 1:nzm+1)   = SPREAD(co2(:), DIM=1, NCOPIES=nx)
		ch4vmr(:, 1:nzm+1)   = SPREAD(ch4(:), DIM=1, NCOPIES=nx)
		n2ovmr(:, 1:nzm+1)   = SPREAD(n2o(:), DIM=1, NCOPIES=nx)
		o2vmr(:, 1:nzm+1)    = SPREAD(o2(:), DIM=1, NCOPIES=nx)
		cfc11vmr(:, 1:nzm+1) = SPREAD(cfc11(:), DIM=1, NCOPIES=nx)
		cfc12vmr(:, 1:nzm+1) = SPREAD(cfc12(:), DIM=1, NCOPIES=nx)
		cfc22vmr(:, 1:nzm+1) = SPREAD(cfc22(:), DIM=1, NCOPIES=nx)
		ccl4vmr(:, 1:nzm+1)  = SPREAD(ccl4(:), DIM=1, NCOPIES=nx)
		
		!
		! Fill out 2D arrays needed by RRTMG
		!
		layerP(:, 1:nzm) = SPREAD(pres(:), DIM=1, NCOPIES=nx)
		layerP(:, nzm+1) = 0.5*SPREAD(presi(nzm+1), DIM=1, NCOPIES=nx) ! add layer
		
		interfaceP(:, 1:nzm+1) = SPREAD(presi(:), DIM=1, NCOPIES=nx)
		interfaceP(:, nzm+2) = MIN(1.0E-4_kind_rb,0.25*layerP(1,nzm+1)) ! near-zero pressure at top of extra layer
		
		! Convert hPa to Pa in layer mass calculation (kg/m2)
		layerMass(:, 1:nzm+1) &
			= 100.0_kind_rb * (interfaceP(:,1:nzm+1) - interfaceP(:,2:nzm+2)) / REAL(ggr,kind_rb)
		
		! Set up for the radiation computation.
		lwHeatingRate(:, :) = 0.0; lwHeatingRateClearSky(:, :) = 0.0
		swHeatingRate(:, :) = 0.0; swHeatingRateClearSky(:, :) = 0.0
		
		layerT(:, 1:nzm) = tabs(:, 1:nzm)
		layerT(:, nzm+1) = 2.0*tabs(:, nzm) - tabs(:, nzm-1) ! add a layer at top.
		
		! Interpolate to find interface temperatures.
		interfaceT(:, 2:nzm+1) = (layerT(:, 1:nzm) + layerT(:, 2:nzm+1)) / 2.0_kind_rb
		!
		! Extrapolate temperature at top from lapse rate within the layer
		interfaceT(:, nzm+2) = 2.0_kind_rb*layerT(:, nzm+1) - interfaceT(:, nzm+1)
		!
		! Use SST as interface temperature of atmosphere at surface.
		interfaceT(:, 1)  = tg(1:nx) !bloss layerT(:, 1)   + (layerT(:, 1)   - interfaceT(:, 2))
		
		
		! -------------------------------------------------------------
		! Compute cloud IWP/LWP and particle sizes - convert from kg to g
		!
		LWP(:, 1:nzm) = qcl(:, 1:nzm) * 1.0E3 * layerMass(:, 1:nzm)
		LWP(:, nzm+1) = 0.0 ! zero out extra layer
		
		IWP(:, 1:nzm) = qci(:, 1:nzm) * 1.0E3 * layerMass(:, 1:nzm)
		IWP(:, nzm+1) = 0.0 ! zero out extra layer
		
		! Store/compute effective radii and cloud fraction
		cloudFrac(:, :) = 0.0
		liqRe(:, :) = 0.0
		iceRe(:, :) = 0.0
		
		! Cloud fraction
		! (TAK) Ported from RAD_RRTM/rad.f90 for SAM6.10.10
		cloudFrac(:,:) = MERGE( 1.0, 0.0, LWP(:,:) > 0.0 .OR. IWP(:,:) > 0.0 )
		
		! Cloud properties are specified through LWP, IWP, liqRe, iceRe, cloudFrac
                cloudTauLW = 0.
                cloudTauSW = 0.
                cloudSsaSW = 0.
                cloudAsmSW = 0.
                cloudForSW = 0.
                cloudTauSW_cldliq = 0.
                cloudTauSW_cldice = 0.
                cloudTauSW_snow = 0.
		
          if(have_cloud_optics) then
            ! ================================================================================
            ! Cloud optics determined using routines that compute cloud optical/LW properties
            !    based on the implied drop size distributions.

            ! set rrtm flags for cloud treatment
            inflgsw = 0; iceflgsw = 0; liqflgsw = 0
            inflglw = 0; iceflglw = 0; liqflglw = 0

            if(use_m2005_cloud_optics) then
              ! cloud optics routine only valid up to dynamical model top, nzm_dyn
              call compute_m2005_cloud_optics(nx, nzm_dyn, lat, &
                   layerMass(1:nx,1:nzm_dyn+1), cloudFrac(1:nx,1:nzm_dyn+1), &
                   cloudTauLW(1:nbndlw,1:nx,1:nzm_dyn+1), cloudTauSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudSsaSW(1:nbndsw,1:nx,1:nzm_dyn+1), cloudAsmSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudForSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_cldliq(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_cldice(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_snow(1:nbndsw,1:nx,1:nzm_dyn+1) )
            elseif(use_thompson_cloud_optics) then
              ! cloud optics routine only valid up to dynamical model top, nzm_dyn
              call compute_thompson_cloud_optics(nx, nzm_dyn, lat, &
                   layerMass(1:nx,1:nzm_dyn+1), cloudFrac(1:nx,1:nzm_dyn+1), &
                   cloudTauLW(1:nbndlw,1:nx,1:nzm_dyn+1), cloudTauSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudSsaSW(1:nbndsw,1:nx,1:nzm_dyn+1), cloudAsmSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudForSW(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_cldliq(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_cldice(1:nbndsw,1:nx,1:nzm_dyn+1), &
                   cloudTauSW_snow(1:nbndsw,1:nx,1:nzm_dyn+1) )
            end if
            !
            !  zero out cloud optical properties above dynamical model domain
            if(nzm.gt.nzm_dyn) then
              do k = nzm_dyn+2,nzm+1
                cloudTauLW(1:nbndlw,1:nx,k) = 0.
                cloudTauSW(1:nbndsw,1:nx,k) = 0.
                cloudSsaSW(1:nbndsw,1:nx,k) = 0.
                cloudAsmSW(1:nbndsw,1:nx,k) = 0.
                cloudForSW(1:nbndsw,1:nx,k) = 0.
                cloudTauSW_cldliq(1:nbndsw,1:nx,k) = 0.
                cloudTauSW_cldice(1:nbndsw,1:nx,k) = 0.
                cloudTauSW_snow(1:nbndsw,1:nx,k) = 0.
              end do
            end if
            !
            ! Normally simulators are run only when the sun is up,
            !    but in case someone decides to use nighttime values...
            !  NOTE: simulators also only valid up to top of dynamical model domain, nzm_dyn
            !
            if(doisccp .or. domodis .or. domisr) then
              ! band 9 is 625 - 778 nm, needed is 670 nm
              tau_067 (1:nx,lat,1:nzm_dyn) = cloudTauSW(9,1:nx,1:nzm_dyn)
              tau_067_cldliq (1:nx,lat,1:nzm_dyn) = cloudTauSW_cldliq(9,1:nx,1:nzm_dyn)
              tau_067_cldice (1:nx,lat,1:nzm_dyn) = cloudTauSW_cldice(9,1:nx,1:nzm_dyn)
              tau_067_snow (1:nx,lat,1:nzm_dyn) = cloudTauSW_snow(9,1:nx,1:nzm_dyn)
              ! band 6 is 820 - 980 cm-1, we need 10.5 micron
              emis_105(1:nx,lat,1:nzm_dyn) = 1. - exp(-cloudTauLW(6,1:nx,1:nzm_dyn))
            end if
          else
            ! ================================================================================
            ! Cloud optics determined internally by RRTMG based on LWP/IWP and effective radii

            ! set RRTMG flags
            inflgsw = 2; iceflgsw = 3; liqflgsw = 1
            inflglw = 2; iceflglw = 3; liqflglw = 1

            !bloss: Computation of effective radii has been moved to rad_full.f90
            liqRe(:,1:nzm) = REAL( rel(:, 1:nzm), kind_rb ) ! rel(1:nx,1:nzm) while LWP(:,1:nzm+1)
            iceRe(:, 1:nzm) = REAL( rei(:, 1:nzm), kind_rb )

!bloss		! Store/compute effective radius
!bloss		! (TAK) Ported from RAD_RRTM/rad.f90 for SAM6.10.10
!bloss		IF (compute_reffc) THEN
!bloss			liqRe(:,1:nzm) = REAL( rel(:, 1:nzm), kind_rb ) ! rel(1:nx,1:nzm) while LWP(:,1:nzm+1)
!bloss		ELSE
!bloss			liqRe(:, 1:nzm) = MERGE( computeRe_Liquid( layerT(:, 1:nzm), land_frac ), 0.0_kind_rb, &
!bloss			                            LWP(:, 1:nzm) > 0.0_kind_rb )
!bloss		ENDIF
!bloss		IF (compute_reffi) THEN
!bloss			iceRe(:, 1:nzm) = REAL( rei(:, 1:nzm), kind_rb )
!bloss		ELSE
!bloss			iceRe(:, 1:nzm) = MERGE( computeRe_Ice( layerT(:, 1:nzm) ), 0.0_kind_rb, &
!bloss			                         IWP(:, 1:nzm) > 0.0_kind_rb )
!bloss		ENDIF
		
		! Limit particle sizes to range allowed by RRTMG parameterizations
		! (TAK) Ported from RAD_RRTM/rad.f90 for SAM6.10.10
		WHERE ( LWP(:, 1:nzm) > 0.0 ) &
			liqRe(:, 1:nzm) = MAX( 2.5_kind_rb, MIN( 60.0_kind_rb, liqRe(:, 1:nzm) ) )
		WHERE ( IWP(:, 1:nzm) > 0.0 ) &
			iceRe(:, 1:nzm) = MAX( 5.0_kind_rb, MIN( 140.0_kind_rb, iceRe(:, 1:nzm) ) )
		
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
                tau_067 (i,lat,1:nzm_dyn) =           tauSWOut(1:nzm_dyn,24) ! RRTMG SW bands run from 16 to 29 (parrrsw.f90); we want 9th of these
                ! band 9 is 625 - 778 nm, needed is 670 nm
                emis_105(i,lat,1:nzm_dyn) = 1. - exp(-tauLWOut(1:nzm_dyn,6)) ! band 6 is 820 - 980 cm-1, we need 10.5 micron 
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
                tau_067_cldliq (i,lat,1:nzm_dyn) = tauSWOut(1:nzm_dyn,24) ! RRTMG SW band number 9 (625 - 778 nm), needed is 670 nm

                ! Same for cloud ice
                call cldprop_sw(nzm+1, 2, 3, 1, cloudFrac_ice(i,:), &
                     prpSWIn, prpSWIn, prpSWIn, prpSWIn, IWP(i,:), dummyWP(i,:), iceRe(i,:), dummyRe(i,:), &
                     tauSWOut, scaled1, scaled2, scaled3)
                tau_067_cldice (i,lat,1:nzm_dyn) = tauSWOut(1:nzm_dyn,24) ! RRTMG SW band number 9 (625 - 778 nm), needed is 670 nm

                tau_067_snow (i,lat,1:nzm_dyn) = 0.! snow is not radiatively active here.
              end do
            end if
          end if
          ! ---------------------------------------------------
          
		! -------------------------------------------------------------
		! Volume mixing fractions for gases.
		!bloss(072009): Note that o3, etc. are now in ppmv and don't need conversions.
		h2ovmr(:, 1:nzm)     = mwdry/mwh2o * qv(:, 1:nzm)
		h2ovmr(:, nzm+1)     = mwdry/mwh2o * qv(:, nzm) ! extrapolate into extra layer at top
		
		! -------------------------------------------------------------
		IF (dolongwave) THEN
			surfaceT(:) = tg(1:nx)
			duflx_dt(:,:) = 0.0
			duflxc_dt(:,:) = 0.0
			
                        SELECT CASE (iaer_rrtmg)
                          CASE (0, 6)
                            ! Only iaer=10 will work with the RRTMG LW parameterization, so that
                            !  the LW optical depth needs to be entered for each band.
                            TauAerosolLW(1:nx,1:nzm+1,1:nbndlw) = 0.
                          CASE (10)
                            ! iaer = 10, input total aerosol optical depth, single scattering albedo 
                            !            and asymmetry parameter (tauaer, ssaaer, asmaer) directly
                            
                            ! simple treatment from Tak Yamaguchi for absorbing aerosol.  
                            ! Tak Yamaguchi 2015/10:RADAEROS
                            TauAerosolLW(1:nx,1:nzm+1,1:nbndlw) = radaerosimple_lw( nzm, nzm_dyn, pres, tabs, qv, nca )
                          END SELECT

			CALL t_startf ('radiation-lw')
			CALL rrtmg_lw ( nx, nzm+1, overlap, idrv, &
				layerP, interfaceP, layerT, interfaceT, surfaceT, &
				h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr, &
				cfc11vmr, cfc12vmr, cfc22vmr, ccl4vmr, surfaceEmissivity,  &
				inflglw, iceflglw, liqflglw, cloudFrac, &
				cloudTauLW, IWP, LWP, iceRe, liqRe, &
				TauAerosolLW, & ! Tak Yamaguchi 2015/10: RADAEROS
				lwUp, lwDown, lwHeatingRate, lwUpClearSky, lwDownClearSky, lwHeatingRateClearSky, &
				duflx_dt, duflxc_dt )
			CALL t_stopf ('radiation-lw')
			
		ENDIF ! IF (dolongwave)
		
		
		! -------------------------------------------------------------
		! Initialize SW properties to zero, assuming sun is down.
		! Will only be changed if doshortwave==.TRUE. and sun is up.
		coszrs = 0.0
		swUp = 0.0
		swDown = 0.0
		swUpClearSky = 0.0
		swDownClearSky = 0.0
		swHeatingRate = 0.0
		swHeatingRateClearSky = 0.0
		
		IF (doshortwave) THEN
			
			! Solar insolation depends on several choices
			!-----------------------------
			IF (doperpetual) THEN
				IF (dosolarconstant) THEN
					solarZenithAngleCos(:) = COS(REAL(zenith_angle,kind_rb) * pi/180.0_kind_rb)
					eccf = REAL(solar_constant,kind_rb)/scon
				ELSE
					!---------------
					! Perpetual sun (diurnally-averaged insolation w/insolation-weighted coszrs)
					solarZenithAngleCos(:) = p_coszrs
					! Adjust insolation using the eccentricity factor
					eccf = p_factor/MAX(p_coszrs, EPSILON(p_coszrs))
				ENDIF
			ELSE
				!---------------
				! Diurnally-varying insolation
				IF (doseasons) THEN
					! The diurnal cycle of insolation will vary
					! according to time of year of the current day.
					dayForSW = day
				ELSE
					! The diurnal cycle of insolation from the calendar day on which the simulation
					! starts (day0) will be repeated throughout the simulation.
					dayForSW = FLOAT(FLOOR(day0)) + day - FLOAT(FLOOR(day))
				ENDIF
				CALL shr_orb_decl(dayForSW, eccen, mvelpp, lambm0, obliqr, delta, eccf)
				solarZenithAngleCos(:) = zenith(dayForSW, &
					pi * REAL(latitude,kind_rb)/180.0_kind_rb, &
					pi * REAL(longitude,kind_rb)/180.0_kind_rb)
			ENDIF
			!---------------
			! coszrs is found in params.f90 and used in the isccp simulator
			coszrs = MAX(0.0_kind_rb, solarZenithAngleCos(1))
			
			! We only call the shortwave if the sun is above the horizon.
			! We assume that the domain is small enough that the entire thing is either lit or shaded.
			IF (ALL(solarZenithAngleCos(:) >= tiny(solarZenithAngleCos))) THEN
				
                        SELECT CASE (iaer_rrtmg)
                          CASE (0)
                            ! no aerosols.
                            TauAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            SsaAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            AsmAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            AerosolProps2(1:nx,1:nzm+1,1:naerec) = 0.
                          CASE (6)
                            ! iaer = 6, use six ECMWF aerosol types 
                            !     ( 1/ continental average    2/ maritime             3/ desert
                            !       4/ urban                  5/ volcanic active      6/ stratospheric background )
                            !           input aerosol optical depth at 0.55 microns for each aerosol type (ecaer)
                            TauAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            SsaAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            AsmAerosolSW(1:nx,1:nzm+1,1:nbndsw) = 0.
                            AerosolProps2(1:nx,1:nzm+1,1:naerec) = 0.  ! This is where the aerosol optical depths should be specified.
                          CASE (10)
                            ! iaer = 10, input total aerosol optical depth, single scattering albedo 
                            !            and asymmetry parameter (tauaer, ssaaer, asmaer) directly
                            
                            ! simple treatment from Tak Yamaguchi
                            ! Tak Yamaguchi 2015/10:RADAEROS
                            !bloss: Use separate Tau, Ssa, Asm argument and turn radaerosimple into a subroutine
                            call radaerosimple_sw( nzm, nzm_dyn, pres, tabs, qv, nca , &
                                 TauAerosolSW, SsaAerosolSW, AsmAerosolSW )
                            AerosolProps2(1:nx,1:nzm+1,1:naerec) = 0.
                          END SELECT

				IF (lat == 1.AND.masterproc) PRINT*, "Let's do some shortwave"
				CALL albedo( ocean, solarZenithAngleCos(:), surfaceT, &
				             asdir(:), aldir(:), asdif(:), aldif(:) )
				IF (lat == 1.AND.masterproc) THEN
					PRINT*, "Range of zenith angles", MINVAL(solarZenithAngleCos), maxval(solarZenithAngleCos)
					PRINT*, "Range of surface albedo (asdir)", MINVAL(asdir), MAXVAL(asdir)
					PRINT*, "Range of surface albedo (aldir)", MINVAL(aldir), MAXVAL(aldir)
					PRINT*, "Range of surface albedo (asdif)", MINVAL(asdif), MAXVAL(asdif)
					PRINT*, "Range of surface albedo (aldif)", MINVAL(aldif), MAXVAL(aldif)
				ENDIF
				CALL t_startf ('radiation-sw')
				CALL rrtmg_sw( nx, nzm+1, overlap, iaer_rrtmg, &
					layerP, interfaceP, layerT, interfaceT, surfaceT, &
					h2ovmr, o3vmr, co2vmr, ch4vmr, n2ovmr, o2vmr,     &
					asdir, asdif, aldir, aldif, &
					solarZenithAngleCos, eccf, 0, scon, &
					inflgsw, iceflgsw, liqflgsw, cloudFrac, &
					cloudTauSW, cloudSsaSW, cloudAsmSW, cloudForSW, &
					IWP, LWP, iceRe, liqRe,  &
					TauAerosolSW, & ! optical depth  (only used if iaer==10)
					SsaAerosolSW, & ! single scattering albedo (only used if iaer==10)
					AsmAerosolSW, & ! asymmetry parameter (only used if iaer==10)
					AerosolProps2, & ! 550nm optical depth for six ECMWF aerosol types (only used if iaer==6)
					swUp, swDown, swHeatingRate, swUpClearSky, swDownClearSky, swHeatingRateClearSky )
				CALL t_stopf ('radiation-sw')
				
				IF (lat == 1.AND.masterproc) THEN
					IF (doshortwave) THEN
						IF (doperpetual) THEN
							WRITE(*,992) coszrs, SUM(swDown(1:nx,nzm+2))/float(nx)
						ELSE
							WRITE(*,991) coszrs, SUM(swDown(1:nx,nzm+2))/float(nx), eccf
						ENDIF
						991 FORMAT('radiation: diurnally-varying insolation, coszrs = ',F10.7, &
							' solin = ',f10.4,' eccf = ',f10.7)
						992 FORMAT('radiation: diurnally-averaged insolation, coszrs = ',F10.7, &
							' solin = ',f10.4)
					ENDIF
					WRITE(*,993) asdir(1), aldir(1), asdif(1), aldif(1)
					993 FORMAT('radiation: surface albedos, asdir= ',F10.7, &
						' aldir = ',f10.7,' asdif = ',f10.7,' aldif = ',f10.7)
				ENDIF
				
			ELSE ! if sun is down
				coszrs = 0.0
				swUp = 0.0
				swDown = 0.0
				swUpClearSky = 0.0
				swDownClearSky = 0.0
				swHeatingRate = 0.0
				swHeatingRateClearSky = 0.0
			ENDIF ! if sun is up
			
		ENDIF ! IF (doshortwave)
		
	END SUBROUTINE rad_driver_rrtm
	
	! --------------------------------------------------------------------------
	
	SUBROUTINE initialize_radiation( cp, iyear, day0, latitude, longitude, doperpetual, ocean )
	
		IMPLICIT NONE
		
		! Inputs
		REAL, INTENT(IN) :: cp ! specific heat of dry air at constant pressure, J/kg/K
		INTEGER, INTENT(IN) :: iyear ! year of simulation (for insolation computation)
		REAL, INTENT(IN) :: day0 ! day of year during iyear (0.0 = 00Z Jan 1) at start of simulaton
		REAL, INTENT(IN) :: latitude ! latitude
		REAL, INTENT(IN) :: longitude ! longitude
		LOGICAL, INTENT(IN) :: doperpetual ! use perpetual (diurnally-averaged) insolation
		LOGICAL, INTENT(IN) :: ocean ! .true. if ocean surface, .false. if land
		
		! Local variables
		REAL(KIND=kind_rb) :: cpdair
		INTEGER :: ierr
		
		!bloss  subroutine shr_orb_params
		!bloss  inputs:  iyear, log_print
		!bloss  ouptuts: eccen, obliq, mvelp, obliqr, lambm0, mvelpp
		CALL shr_orb_params(iyear, eccen, obliq, mvelp, obliqr, lambm0, mvelpp, .FALSE.)
		
		IF (doperpetual) THEN
			! Perpetual sun (no diurnal cycle)
			!   get diurnally-averaged insolation as a factor of solar constant
			!   as well as mean insolation-weighted solar zenith angle.
			CALL perpetual_factor_coszrs(REAL(day0,kind_rb), REAL(latitude,kind_rb), &
				REAL(longitude,kind_rb))
		ENDIF
		
		cpdair = cp
		CALL rrtmg_sw_ini(cpdair)
		CALL rrtmg_lw_ini(cpdair)
		
  if(trim(micro_scheme_name()) == 'm2005' .and. & 
       (compute_reffc .or. compute_reffi) .and. &
       (.NOT.dorrtm_cloud_optics_from_effrad_LegacyOption)) then
    call m2005_cloud_optics_init
    use_m2005_cloud_optics = .true.
    have_cloud_optics = .true.
    if(masterproc) write(*,*) 'Using cloud optics for RRTMG radiation from M2005 microphysics'
  end if

  if(trim(micro_scheme_name()) == 'thompson' .and. &
       (compute_reffc .or. compute_reffi) .and. &
       (.NOT.dorrtm_cloud_optics_from_effrad_LegacyOption)) then
    call thompson_cloud_optics_init
    use_thompson_cloud_optics = .true.
    have_cloud_optics = .true.
    if(masterproc) write(*,*) 'Using cloud optics for RRTMG radiation from THOM microphysics'
  end if

    if(.NOT.have_cloud_optics) then
       if(masterproc) write(*,*) 'Using built-in cloud optics within RRTMG'
     end if

                !bloss(RRTM4PBL): set up aerosol treatment
                iaer_rrtmg = 0 ! default is no aerosols

                ! Use simple treatment for aerosols from Tak Yamaguchi and used in the paper 
                !    Yamaguchi, T., G. Feingold, J. Kazil, and A. McComiskey (2015), 
                !       Stratocumulus to cumulus transition in the presence of elevated smoke
                !       layers, Geophys. Res. Lett., 42, doi:10.1002/2015GL066544.
                if(doradaerosimple) iaer_rrtmg = 10

                SELECT CASE (iaer_rrtmg)
                CASE (0)
                  ! no aerosols:  nothing to do...
                CASE (6)
                  ! set up aerosol treatment using six ECMWF aerosol types
                  !     ( 1/ continental average    2/ maritime             3/ desert
                  !       4/ urban                  5/ volcanic active      6/ stratospheric background )

                  if(masterproc) write(*,*) '**** ERROR: RRTMG aerosol setting iaer==6 IS NOT YET IMPLEMENTED *****'
                  if(masterproc) write(*,*) '**** STOPPING ... ******'
                  call task_abort()
                CASE (10)
                  ! Tak Yamaguchi 2015/10: RADAEROS
                  CALL initialize_simple_absorbing_aerosol()
                END SELECT
		
		land_frac = MERGE( 0.0, 1.0, ocean )
		isInitialized_RadDriver = .TRUE.
		
	END SUBROUTINE initialize_radiation
	
	! --------------------------------------------------------------------------
	!
	! Trace gas profiles
	!
	! --------------------------------------------------------------------------
	SUBROUTINE tracesini(nzm,pres,presi,ggr,nzsnd,psnd,o3snd,have_o3mmr,co2factor,masterproc)
	
		USE rrlw_ncpar
		USE netcdf
		
		IMPLICIT NONE
		!
		! Initialize trace gaz vertical profiles
		!   The files read from the top down
		!
		!bloss(072009): Get trace gas profiles from rrtmg_lw.nc, the data file provided with RRTMG.
		!               These are indexed from bottom to top and are in ppmv, so that no conversion
		!               is needed for use with RRTMG.
		!
		INTEGER, INTENT(IN) :: nzm
		REAL, INTENT(IN) :: pres(nzm), presi(nzm+1)
		REAL, INTENT(IN) :: ggr ! gravitational acceleration (~9.8 m/s2)
		INTEGER, INTENT(IN) :: nzsnd
		REAL, INTENT(IN) :: psnd(nzsnd), o3snd(nzsnd)
		LOGICAL, INTENT(IN) :: have_o3mmr
		REAL, INTENT(IN) :: co2factor ! scaling factor for CO2
		LOGICAL, INTENT(IN) :: masterproc
		INTEGER :: k, m, ierr
		REAL :: godp ! gravity over delta pressure
		REAL :: plow, pupp
		INTEGER(KIND=kind_im) :: ncid, varID, dimIDab, dimIDp
		
		INTEGER(KIND=kind_im) :: Nab, nPress, ab
		REAL(KIND=kind_rb) :: wgtlow, wgtupp, pmid
		REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:) :: pMLS, trace_single
		REAL(KIND=kind_rb), ALLOCATABLE, DIMENSION(:,:) :: trace, trace_in
		CHARACTER(LEN=nf90_max_name) :: tmpName
		
		REAL(KIND=kind_rb) :: pres_snd(nzsnd), o3mmr_snd(nzsnd)
		
                REAL(KIND=kind_rb) :: factor ! for modifying trace gas concentrations in notracegas option

		INTEGER, parameter :: nTraceGases = 9
		REAL(KIND=kind_rb), DIMENSION(nzm+1) :: tmppres ! pres w/extra level at top.
		REAL(KIND=kind_rb), DIMENSION(nzm+2) :: tmppresi ! presi w/extra level at top.
		REAL(KIND=kind_rb), DIMENSION(nzm+1) :: tmpTrace, tmpTrace2
		REAL(KIND=kind_rb), DIMENSION(nzm+2,nTraceGases) :: trpath
		CHARACTER(LEN=maxAbsorberNameLength), DIMENSION(nTraceGases), PARAMETER :: &
			TraceGasNameOrder = (/ &
				'O3   ',  &
				'CO2  ',  &
				'CH4  ',  &
				'N2O  ',  & 
				'O2   ',  &
				'CFC11',  &
				'CFC12',  &
				'CFC22',  &
				'CCL4 '  /)
		
		IF (isallocated_tracegases.AND.(nz_tracegases.ne.nzm+1)) THEN
			! Number of vertical levels for radiation has changed since last update of trace gas
			! concentrations (because the size of the patched sounding above the model domain has
			! changed with the pressure at the model top, perhaps).
			!
			! Deallocate trace gas arrays and re-allocate below.
			DEALLOCATE(o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4, STAT=ierr)
			IF (ierr /= 0) THEN
				WRITE(*,*) 'ERROR: could not deallocate trace gas arrays in tracesini'
				CALL rad_error()
			ELSE
				isallocated_tracegases = .FALSE.
				nz_tracegases = -1
			ENDIF
		ENDIF
		
		IF (.NOT.isallocated_tracegases) THEN
			! Allocate trace gas arrays. These have an extra level for the mean mass-weighted trace gas
			! concentration in the overlying atmosphere.
			!
			nz_tracegases = nzm+1  ! add one level to compute trace gas levels to TOA
			ALLOCATE(o3(nz_tracegases), co2(nz_tracegases), ch4(nz_tracegases), &
				n2o(nz_tracegases), o2(nz_tracegases), cfc11(nz_tracegases), &
				cfc12(nz_tracegases), cfc22(nz_tracegases), ccl4(nz_tracegases), &
				STAT=ierr)
			IF (ierr /= 0) THEN
				WRITE(*,*) 'ERROR: could not allocate trace gas arrays in tracesini'
				CALL rad_error()
			ELSE
				isallocated_tracegases = .TRUE.
			ENDIF
		ENDIF
		
		! Read profiles from rrtmg data file.
		status(:) = nf90_NoErr
		status(1) = nf90_open('RUNDATA/rrtmg_lw.nc',nf90_nowrite,ncid)
		IF (status(1) /= 0) THEN
			WRITE(*,*) 'ERROR: could not find RUNDATA/rrtmg_lw.nc'
			CALL rad_error()
		ENDIF
		
		status(2) = nf90_inq_dimid(ncid,"Pressure",dimIDp)
		status(3) = nf90_inquire_dimension(ncid, dimIDp, tmpName, nPress)
		
		status(4) = nf90_inq_dimid(ncid,"Absorber",dimIDab)
		status(5) = nf90_inquire_dimension(ncid, dimIDab, tmpName, Nab)
		
		ALLOCATE(pMLS(nPress), trace(nTraceGases,nPress), trace_in(Nab,nPress), &
			trace_single(nPress), STAT=ierr)
		IF (ierr /= 0) THEN
			WRITE(*,*) 'ERROR: could not declare arrays in tracesini'
			CALL rad_error()
		ENDIF
		
		! Initialize arrays
		pMLS = 0.0
		trace = 0.0
		trace_in = 0.0
		
		status(6) = nf90_inq_varid(ncid,"Pressure",varID)
		status(7) = nf90_get_var(ncid, varID, pMLS)
		
		status(8) = nf90_inq_varid(ncid,"AbsorberAmountMLS",varID)
		status(9) = nf90_get_var(ncid, varID, trace_in)
		
		DO m = 1, nTraceGases
			CALL getAbsorberIndex(TRIM(tracegasNameOrder(m)),ab)
			trace(m,1:nPress) = trace_in(ab,1:nPress)
			WHERE (trace(m,:) > 2.0)
				trace(m,:) = 0.0
			END WHERE
		ENDDO
		
		IF (MAXVAL(ABS(status(1:8+nTraceGases))) /= 0) THEN
			WRITE(*,*) 'Error in reading trace gas sounding from RUNDATA/rrtmg_lw.nc'
			CALL rad_error()
		ENDIF
		
!!$		DO k = 1, nPress
!!$			WRITE(*,999) pMLS(k), (trace(m,k),m=1,nTraceGases)
!!$		ENDDO
		999 FORMAT(f8.2,12e12.4)
		
		!bloss: modify routine to compute trace gas paths from surface to top of supplied sounding.
		! Then, interpolate these paths onto the interface pressure levels of the model grid, with
		! an extra level at the top for the overlying atmosphere. Differencing these paths and
		! dividing by dp/g will give the mean mass concentration in that level.
		!
		! This procedure has the advantage that the total trace gas path will be invariant to
		! changes in the vertical grid.
		
		! Model's pressure sounding
		tmppres(1:nzm) = pres(1:nzm) ! pressure at model levels (mb)
		tmppresi(1:nzm+1) = presi(1:nzm+1) ! pressure at model interfaces (mb)
		
		! Add a level for the overlying atmosphere.
		tmppres(nzm+1) = 0.5*presi(nzm+1) ! half of pressure at top of model
		tmppresi(nzm+2) = MIN(1.0E-4_kind_rb,0.25*tmppres(nzm+1)) ! near-zero pressure at top of extra laye
		
		! Trace gas paths at surface are zero.
		trpath(1,:) = 0.0
		
		DO k = 2, nzm+2
			! Start with trace path at interface below.
			trpath(k,:) = trpath(k-1,:)
			
			! If pressure greater than sounding, assume concentration at bottom.
			IF (tmppresi(k-1) > pMLS(1)) THEN
				trpath(k,:) = trpath(k,:) &
				           + (tmppresi(k-1) - MAX(tmppresi(k),pMLS(1)))/ggr & ! dp/g
				           * trace(:,1)                                 ! *tr
			ENDIF
			
			DO m = 2, nPress
				! Limit pMLS(m:m-1) so that they are within the model level tmppresi(k-1:k).
				plow = MIN(tmppresi(k-1),MAX(tmppresi(k),pMLS(m-1)))
				pupp = MIN(tmppresi(k-1),MAX(tmppresi(k),pMLS(m)))
				
				IF (plow > pupp) THEN
					pmid = 0.5*(plow+pupp)
					
					wgtlow = (pmid-pMLS(m))/(pMLS(m-1)-pMLS(m))
					wgtupp = (pMLS(m-1)-pmid)/(pMLS(m-1)-pMLS(m))
!!$					WRITE(*,*) pMLS(m-1),pmid,pMLS(m),wgtlow,wgtupp
					! Include this level of the sounding in the trace gas path
					trpath(k,:) = trpath(k,:) &
					            + (plow - pupp)/ggr*(wgtlow*trace(:,m-1)+wgtupp*trace(:,m)) ! dp/g*tr
				ENDIF
			ENDDO
			
			! If pressure is off top of trace gas sounding, assume concentration at top
			IF (tmppresi(k) < pMLS(nPress)) THEN
				trpath(k,:) = trpath(k,:) &
				            + (MIN(tmppresi(k-1),pMLS(nPress)) - tmppresi(k))/ggr & ! dp/g
				            * trace(:,nPress)                               ! *tr
			ENDIF
		
		ENDDO
		
		DO m = 1, nTraceGases
			DO k = 1, nzm+1
				godp = ggr/(tmppresi(k) - tmppresi(k+1))
				tmpTrace(k) = (trpath(k+1,m) - trpath(k,m))*godp
			ENDDO
			IF (TRIM(TraceGasNameOrder(m))=='O3') THEN
				o3(:) = tmpTrace(:)
			ELSEIF (TRIM(TraceGasNameOrder(m))=='CO2') THEN
				! Scale by co2factor (default = 1.)
				co2(:) = REAL(co2factor,KIND=kind_rb)*tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='CH4') THEN
				ch4(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='N2O') THEN
				n2o(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='O2') THEN
				o2(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='CFC11') THEN
				cfc11(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='CFC12') THEN
				cfc12(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='CFC22') THEN
				cfc22(:) = tmpTrace(:)
			ELSEIF(TRIM(TraceGasNameOrder(m))=='CCL4') THEN
				ccl4(:) = tmpTrace(:)
			ENDIF
		ENDDO
		
		IF (have_o3mmr) THEN
			! Use Ozone profile from SCAM netcdf input file in place of default RRTM profile.
			
			! Generate interface pressures for ozone input sounding
			pres_snd(:) = psnd(:)
			o3mmr_snd(:) = o3snd(:) ! convert to double precision
			
			tmpTrace2(:) = o3(:)
			
			CALL path_preserving_interpolation(nzsnd,pres_snd,o3mmr_snd,nzm+1,tmppresi,o3,ggr)
			
			o3 = (mwdry/mwo3)*o3
			
!!$			DO k = 1, nzm+1
!!$				WRITE(*,994) k, tmppres(k), o3(k), tmpTrace2(k), o3(k) - tmpTrace2(k)
!!$				994 FORMAT(i4,f8.2,4e12.4)
!!$			ENDDO
!!$			WRITE(*,*)
		ENDIF

                ! scale CO2 if desired.  (Default value of nxco2 == 1.)
                co2(:) = nxco2*co2(:) 

                ! turn off (almost) the radiative effects of trace gases
                !   Note that zeroing them out might cause numerical issues in the scheme.
                IF(notracegases) THEN
                  factor = 0.01
                  o3(:) = factor*o3(:)
                  ch4(:) = factor*ch4(:)
                  n2o(:) = factor*n2o(:)
                  o2(:) = factor*o2(:)
                  cfc11(:) = factor*cfc11(:)
                  cfc12(:) = factor*cfc12(:)
                  cfc22(:) = factor*cfc22(:)
                  ccl4(:) = factor*ccl4(:)
                END IF
                  
		
		IF (masterproc) THEN
			PRINT*,'RRTMG rrtmg_lw.nc trace gas profile: number of levels=',nPress
			PRINT*,'gas traces vertical profiles (ppmv):'
			PRINT*,'p, hPa', ('       ',TraceGasNameOrder(m),m=1,nTraceGases)
			DO k = 1, nzm+1
				WRITE(*,999) tmppres(k),o3(k),co2(k),ch4(k),n2o(k),o2(k), &
				cfc11(k),cfc12(k), cfc22(k),ccl4(k)
			ENDDO
			PRINT*,'done...'
		ENDIF
		
		DEALLOCATE(pMLS, trace, trace_in, STAT=ierr)
		IF (ierr /= 0) THEN
			WRITE(*,*) 'ERROR: could not deallocate arrays in tracesini'
			CALL rad_error()
		ENDIF
		
	END SUBROUTINE tracesini
	
	! --------------------------------------------------------------------------
	! (TAK)
	SUBROUTINE tracesupdate( nzm, pres, presi, ggr, nzsnd, psnd, o3snd, have_o3mmr, masterproc )
		
		IMPLICIT NONE
		
		! Update trace gas vertical profiles with time interpolated trace gas profiles
		
		INTEGER, INTENT(IN) :: nzm
		REAL, DIMENSION(nzm), INTENT(IN) :: pres
		REAL, DIMENSION(nzm+1), INTENT(IN) :: presi
		REAL, INTENT(IN) :: ggr ! gravitational acceleration (~9.8 m/s2)
		INTEGER, INTENT(IN) :: nzsnd
		REAL, DIMENSION(nzsnd), INTENT(IN) :: psnd, o3snd ! background profiles
		LOGICAL, INTENT(IN) :: have_o3mmr
		LOGICAL, INTENT(IN) :: masterproc
		
		! Local
		REAL(KIND=kind_rb), DIMENSION(nzm+1) :: tmppres  ! pres w/extra level at top.
		REAL(KIND=kind_rb), DIMENSION(nzm+2) :: tmppresi ! presi w/extra level at top.
		REAL(KIND=kind_rb), DIMENSION(nzsnd) :: pres_snd, o3mmr_snd
		INTEGER :: k, m, ierr
		
		! Model's pressure sounding
		tmppres(1:nzm) = pres(1:nzm) ! pressure at model levels (mb)
		tmppresi(1:nzm+1) = presi(1:nzm+1) ! pressure at model interfaces (mb)
		
		! Add a level for the overlying atmosphere.
		tmppres(nzm+1) = 0.5*presi(nzm+1) ! half of pressure at top of model
		tmppresi(nzm+2) = MIN(1.0E-4_kind_rb,0.25*tmppres(nzm+1)) ! near-zero pressure at top of extra layer
		
		! o3
		IF (have_o3mmr) THEN
			! Generate interface pressures for ozone input sounding
			pres_snd(:) = psnd(:)
			o3mmr_snd(:) = o3snd(:) ! convert to double precision
			CALL path_preserving_interpolation( nzsnd, pres_snd, o3mmr_snd, nzm+1, tmppresi, o3, ggr )
			o3 = (mwdry/mwo3)*o3
		ENDIF
		
	END SUBROUTINE tracesupdate
	
	! --------------------------------------------------------------------------
	!
	! Astronomy-related procedures
	! 
	! --------------------------------------------------------------------------
	ELEMENTAL REAL(KIND=kind_rb) FUNCTION zenith(calday, clat, clon)
		IMPLICIT NONE
		REAL(KIND=kind_rb), INTENT(in) :: calday, & ! Calendar day, including fraction
		                    clat,   & ! Current centered latitude (radians)
		                    clon      ! Centered longitude (radians)
		
		REAL(KIND=kind_rb) :: delta, & ! Solar declination angle in radians
		                      eccf
		INTEGER  :: i     ! Position loop index
		
		CALL shr_orb_decl (calday, eccen, mvelpp, lambm0, obliqr, delta, eccf)
		!
		! Compute local cosine solar zenith angle
		!
		zenith = shr_orb_cosz(calday, clat, clon, delta)
	END FUNCTION zenith
	
	! --------------------------------------------------------------------------
	
	SUBROUTINE perpetual_factor_coszrs(day, lat, lon)
		
		IMPLICIT NONE
		
		REAL(KIND=kind_rb), INTENT(IN) :: day, lat, lon ! Day (without fraction); centered lat/lon (degrees)
		REAL(KIND=kind_rb) :: delta, & ! Solar declination angle in radians
		                      eccf
		
		! Estimate the factor to multiply the solar constant so that the sun hanging perpetually
		! right above the head (zenith angle=0) would produce the same total input the TOA as the
		! sun subgect to diurnal cycle.
		! coded by Marat Khairoutdinov, 2004
		
		! Local
		INTEGER :: n
		REAL(KIND=kind_rb) :: tmp, tmp2, dttime, ttime
		REAL(KIND=kind_rb) :: coszrs
		REAL(KIND=kind_rb) :: clat, clon
		
		REAL(KIND=kind_rb), PARAMETER :: dtrad = 60. ! default 60 second interval between insolation computations
		
		tmp = 0.0
		tmp2 = 0.0
		
		clat = pi * lat/180.0_kind_rb
		clon = pi * lon/180.0_kind_rb
		
		DO n = 1, 60*24 ! compute insolation for each minute of the day.
			
			ttime = day+REAL(n,kind_rb)*dtrad/86400.0_kind_rb
			
			CALL shr_orb_decl(ttime, eccen, mvelpp, lambm0, obliqr, delta, eccf)
			coszrs = zenith(ttime, clat, clon)
			tmp  = tmp  + MAX(0.0_kind_rb, eccf * coszrs)
			tmp2 = tmp2 + MAX(0.0_kind_rb, eccf * coszrs) * MAX(0.0_kind_rb,coszrs)
		
		ENDDO
		
		tmp = tmp/REAL(60*24,kind_rb) ! average insolation/scon across minutes
		tmp2 = tmp2/REAL(60*24,kind_rb) ! average insolation*coszrs/scon across minutes
		
		p_factor = tmp ! mean insolation divided by solar constant
		p_coszrs = tmp2/MAX(tmp, EPSILON(tmp)) ! mean insolation-weighted cosine of solar zenith angle
		
!!$		WRITE(*,*) 'eccentricity factor = ', eccf
!!$		WRITE(*,*) 'delta = ', delta
!!$		WRITE(*,*) 'insolation factor = ', p_factor
!!$		WRITE(*,*) 'insolation-weighted coszrs = ', p_coszrs
!!$		WRITE(*,*) 'perpetual insolation = ', p_factor*scon
		
	END SUBROUTINE perpetual_factor_coszrs
	
	! --------------------------------------------------------------------------
	!bloss
	SUBROUTINE path_preserving_interpolation(N_in,pres_in,q_in,N_out,presi_out,q_out,ggr)
	
		IMPLICIT NONE
		
		! Inputs
		INTEGER, INTENT(IN) :: N_in, N_out
		REAL(KIND=kind_rb), INTENT(IN) :: pres_in(N_in), presi_out(N_out+1)
		REAL(KIND=kind_rb), INTENT(IN) :: q_in(N_in)
		REAL, INTENT(IN) :: ggr
		
		! Output
		REAL(KIND=kind_rb), INTENT(OUT) :: q_out(N_out)
		
		! Local variables
		REAL(KIND=kind_rb) :: dp, godp
		REAL(KIND=kind_rb) :: plow, pupp, pmid, wgtlow, wgtupp
		REAL(KIND=kind_rb) :: trpath(N_out+1)
		INTEGER :: k, m
		
		! Trace gas paths at surface are zero.
		trpath(1) = 0.0
		
		DO k = 2, N_out+1
			! Start with trace path at interface below.
			trpath(k) = trpath(k-1)
			
			! If pressure greater than sounding, assume concentration at bottom.
			IF (presi_out(k-1) > pres_in(1)) THEN
				dp = presi_out(k-1) - MAX(presi_out(k),pres_in(1))
				trpath(k) = trpath(k) + (dp/ggr)*q_in(1)
			ENDIF
			
			DO m = 2, N_in
				! Limit pres_in(m:m-1) so that they are within the model level presi_out(k-1:k).
				plow = MIN(presi_out(k-1),MAX(presi_out(k),pres_in(m-1)))
				pupp = MIN(presi_out(k-1),MAX(presi_out(k),pres_in(m)))
				
				IF (plow > pupp) THEN
					pmid = 0.5*(plow+pupp)
					
					wgtlow = (pmid-pres_in(m))/(pres_in(m-1)-pres_in(m))
					wgtupp = (pres_in(m-1)-pmid)/(pres_in(m-1)-pres_in(m))
					
					! Include this level of the sounding in the trace gas path
					trpath(k) = trpath(k) &
					          + (plow - pupp)/ggr*(wgtlow*q_in(m-1)+wgtupp*q_in(m)) ! dp/g*tr
				ENDIF
			ENDDO
			
			! If pressure is off top of trace gas sounding, assume concentration at top
			IF (presi_out(k) < pres_in(N_in)) THEN
				dp = MIN(presi_out(k-1),pres_in(N_in)) - presi_out(k)
				trpath(k) = trpath(k) + (dp/ggr)*q_in(N_in)
			ENDIF
		ENDDO
		
		DO k = 1, N_out
			godp = ggr/(presi_out(k) - presi_out(k+1))
			q_out(k) = (trpath(k+1) - trpath(k))*godp
		ENDDO
		
	END SUBROUTINE path_preserving_interpolation
	
	! --------------------------------------------------------------------------
	!
	! Writing and reading binary restart files
	!
	! --------------------------------------------------------------------------
	! (TAK) Ported from RAD_RRTM (SAM6.10.4) and adjusted for the PBL interface
	SUBROUTINE write_rad()
	
		IMPLICIT NONE
		
		INTEGER :: irank, ii
		
		!bloss: Added a bunch of statistics-related stuff to the restart file to nicely handle the
		!  rare case when nrad exceeds nstat and the model restarts with mod(nstep,nrad)~=0. This
		!  would cause many of the radiation statistics to be zero before the next multiple of nrad.
		
		IF (masterproc) PRINT*,'Writting radiation restart file...'
		
		IF (restart_sep) THEN
			OPEN(56, FILE=TRIM(constructRestartFileName(case, caseId, rank)), &
				STATUS='unknown', FORM='unformatted')
			WRITE(56) nsubdomains
			WRITE(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, radqrlw, radqrsw, &
				radqrclw, radqrcsw, &
				NetlwUpSurface, NetlwUpSurfaceClearSky, NetlwUpToa, NetlwUpToaClearSky, &
				NetswDownSurface, NetswDownSurfaceClearSky, NetswDownToa, NetswDownToaClearSky, &
				insolation_TOA, swDownSurface, lwDownSurface, &
				npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, &
				o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
			CLOSE(56)
		ELSE
			DO irank = 0, nsubdomains-1
				CALL task_barrier()
				IF (irank == rank) THEN
					OPEN(56, FILE=TRIM(constructRestartFileName(case, caseId, nSubdomains)), &
						STATUS='unknown', FORM='unformatted')
					IF (masterproc) THEN
						WRITE(56) nsubdomains
					ELSE
						READ(56)
						DO ii=0, irank-1 ! skip records
							READ(56)
						ENDDO
					ENDIF
					WRITE(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, radqrlw, radqrsw, &
						radqrclw, radqrcsw, &
						NetlwUpSurface, NetlwUpSurfaceClearSky, NetlwUpToa, NetlwUpToaClearSky, &
						NetswDownSurface, NetswDownSurfaceClearSky, NetswDownToa, NetswDownToaClearSky, &
						insolation_TOA, swDownSurface, lwDownSurface, &
						npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, &
						o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
					CLOSE(56)
				ENDIF
			ENDDO
		ENDIF ! restart_sep
		
		IF (masterproc) PRINT*,'Saved radiation restart file. nstep=', nstep
		CALL task_barrier()
		
	END SUBROUTINE write_rad
	
	! --------------------------------------------------------------------------
	
	SUBROUTINE read_rad()
	
		IMPLICIT NONE
		
		INTEGER :: irank, ii
		
		IF (masterproc) PRINT*,'Reading radiation restart file...'
		
		IF (restart_sep) THEN
			IF (nrestart /= 2) THEN
				OPEN(56, FILE=TRIM(constructRestartFileName(case, caseid, rank)), &
					STATUS='unknown', FORM='unformatted')
			ELSE
				OPEN(56, FILE=TRIM(constructRestartFileName(case_restart, caseid_restart, rank)), &
					STATUS='unknown', FORM='unformatted')
			ENDIF
			READ(56) ! skip
			READ(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, radqrlw, radqrsw, &
						radqrclw, radqrcsw, &
						NetlwUpSurface, NetlwUpSurfaceClearSky, NetlwUpToa, NetlwUpToaClearSky, &
						NetswDownSurface, NetswDownSurfaceClearSky, NetswDownToa, NetswDownToaClearSky, &
						insolation_TOA, swDownSurface, lwDownSurface, &
						npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, &
						o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
			CLOSE(56)
		ELSE
			DO irank = 0, nsubdomains-1
				CALL task_barrier()
				IF (irank == rank) THEN
					IF (nrestart /= 2) THEN
						OPEN(56, FILE=TRIM(constructRestartFileName(case, caseId, nSubdomains)), &
							STATUS='unknown', FORM='unformatted')
					ELSE
						OPEN(56, FILE=TRIM(constructRestartFileName(case, caseId_restart, nSubdomains)), &
							STATUS='unknown', FORM='unformatted')
					ENDIF
					READ(56)
					DO ii = 0, irank-1 ! skip records
						READ(56)
					ENDDO
					READ(56) nradsteps, qrad, radlwup, radlwdn, radswup, radswdn, radqrlw, radqrsw, &
						radqrclw, radqrcsw, &
						NetlwUpSurface, NetlwUpSurfaceClearSky, NetlwUpToa, NetlwUpToaClearSky, &
						NetswDownSurface, NetswDownSurfaceClearSky, NetswDownToa, NetswDownToaClearSky, &
						insolation_TOA, swDownSurface, lwDownSurface, &
						npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, &
						o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
					CLOSE(56)
				ENDIF
			ENDDO
		ENDIF ! restart_sep
		
		IF (rank == nsubdomains-1) THEN
			PRINT*,'Case:', caseid
			PRINT*,'Restart radiation at step:', nstep
			PRINT*,'Time:', nstep*dt
		ENDIF
		
		CALL task_barrier()
		
	END SUBROUTINE read_rad
	
	! --------------------------------------------------------------------------
	
	FUNCTION constructRestartFileName(case, caseid, index) result(name)
		CHARACTER(LEN=*), INTENT(IN) :: case, caseid
		INTEGER, INTENT(IN) :: index
		CHARACTER(LEN=256) :: name
		CHARACTER(LEN=4) :: indexChar
		INTEGER, EXTERNAL :: lenstr
		WRITE(indexChar,'(i4)') index
		name = './RESTART/' // TRIM(case) //'_'// TRIM(caseid) //'_'// &
		       indexChar(5-lenstr(indexChar):4) //'_restart_rad.bin'
!bloss		       trim(indexChar) //'_restart_rad.bin'
	END FUNCTION constructRestartFileName
	

	! --------------------------------------------------------------------------
	!
	! Routines for simple absorbing aerosol ! Tak Yamaguchi 2015/10: RADAEROS
	!
	! --------------------------------------------------------------------------
	SUBROUTINE initialize_simple_absorbing_aerosol()
	
		IMPLICIT NONE
		
		! Local
		INTEGER :: nbin
		REAL :: r1, r2
		REAL, ALLOCATABLE, DIMENSION(:,:) :: tmp
		INTEGER :: i, j, ierr, i1, i2
		
		! Exit if the option is off.
		IF ( .NOT.doradaerosimple ) RETURN
		
		! Find nbin
		OPEN( 12, FILE='RUNDATA/RADAEROSIMPLE/aer_opts_lw', STATUS='OLD' )
		nbin = 0
		DO
			READ( 12, *, IOSTAT=ierr )
			IF ( ierr < 0 ) EXIT
			nbin = nbin + 1
		ENDDO
		CLOSE( 12 )
		nbin = nbin / nbndlw
		
		! Allocation
		ALLOCATE( rh_lw(nbin), rh_sw(nbin), ext_coef_lw(nbndlw,nbin), ext_coef_sw(nbndsw,nbin), &
		          ssalbdo(nbndsw,nbin), asym_fr(nbndsw,nbin), tmp(nbndlw,nbin), STAT=ierr )
		IF (ierr /= 0) THEN
			WRITE(*,*) 'Could not allocate arrays in initialize_simple_absorbing_aerosol'
			CALL rad_error()
		ENDIF
		
		! Read lookup table
		! LW
		IF ( masterproc ) PRINT*,'RADAEROSIMPLE: Reading aer_opts_lw'
		OPEN( 12, FILE='RUNDATA/RADAEROSIMPLE/aer_opts_lw', STATUS='OLD' )
		DO i = 1, nbndlw
			DO j = 1, nbin
				READ( 12, *, IOSTAT=ierr ) r1, r2, rh_lw(j), ext_coef_lw(i,j), &
				                           tmp(i,j), tmp(i,j), i1, i2
			!	IF ( masterproc ) PRINT*, 'nbndlw, nbin:', i, j, r1, r2, i1, i2
				IF ( ierr > 0 ) THEN
					IF ( masterproc ) PRINT*,'Input error. Stop.'
					CALL rad_error()
				ENDIF
				IF ( ierr < 0 ) EXIT
			END DO
		END DO
		CLOSE( 12 )
		
		! SW
		IF ( masterproc ) PRINT*,'RADAEROSIMPLE: Reading aer_opts_sw'
		OPEN( 13, FILE='RUNDATA/RADAEROSIMPLE/aer_opts_sw', STATUS='OLD' )
		DO i = 1, nbndsw
			DO j = 1, nbin
				READ( 13, *, IOSTAT=ierr ) r1, r2, rh_sw(j), ext_coef_sw(i,j), &
				                           ssalbdo(i,j), asym_fr(i,j), i1, i2
			!	IF ( masterproc ) PRINT*, 'nbndsw, nbin:', i, j, r1, r2, i1, i2
				IF ( ierr > 0 ) THEN
					IF ( masterproc ) PRINT*,'Input error. Stop.'
					CALL rad_error()
				ENDIF
				IF ( ierr < 0 ) EXIT
			END DO
		END DO
		CLOSE( 13 )
		
		!
		DEALLOCATE( tmp )
		
	END SUBROUTINE initialize_simple_absorbing_aerosol
	
	! --------------------------------------------------------------------------
	
	FUNCTION radaerosimple_lw( nzrad, nzm_dyn, pres, tabs, qv, nca ) RESULT( tau )
	
		IMPLICIT NONE
		
		! IN
		INTEGER, INTENT(IN) :: nzrad, nzm_dyn ! nzrad >= nzm_dyn
		REAL, DIMENSION(nzrad), INTENT(IN) :: pres ! pressure (mb) at center of model levels
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: tabs ! absolute temperature (K) at model levels
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: qv   ! water vapor mass mixing ratio (kg/kg) 
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: nca  ! aerosol number concentration #/cm3
		
		! OUT
		REAL(KIND=kind_rb), DIMENSION(nx,nzrad+1,nbndlw) :: tau
		
		! Local
		REAL, EXTERNAL :: qsatw
		REAL :: rh
		INTEGER :: i, k, n, ibin
		
		
		! Zero output array
		tau(:,:,:) = 0.0
		
		! Exit if doradaerosimple = .FALSE.
		IF ( .NOT.doradaerosimple ) RETURN
		
		!
		DO n = 1, nbndlw
			DO k = 1, nzm_dyn ! UPTO ACTUAL DOMAIN TOP (NZM_DYN <= NZRAD). READ COMMENTS IN RAD_FULL.F90
				DO i = 1, nx
					rh = qv(i,k) / qsatw( tabs(i,k), pres(k) )
					ibin = MINLOC( ABS( rh-rh_lw ), DIM=1 )
					tau(i,k,n) = ext_coef_lw(n,ibin) * nca(i,k) * dz * adz(k) * 1.0E2
					! ext_coef_lw: cm-1 assuming 1 particl cm-3
					! nca: # cm-3
					! dz * adz * 1.0E2: m * 100 = cm
				ENDDO
			ENDDO
		ENDDO
		
	END FUNCTION radaerosimple_lw
	
	! --------------------------------------------------------------------------
	
        SUBROUTINE radaerosimple_sw( nzrad, nzm_dyn, pres, tabs, qv, nca , tau, ssa, asm )

		IMPLICIT NONE
		
		! IN
		INTEGER, INTENT(IN) :: nzrad, nzm_dyn ! nzrad >= nzm_dyn
		REAL, DIMENSION(nzrad), INTENT(IN) :: pres ! pressure (mb) at center of model levels
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: tabs ! absolute temperature (K) at model levels
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: qv   ! water vapor mass mixing ratio (kg/kg) 
		REAL, DIMENSION(nx,nzrad), INTENT(IN) :: nca  ! aerosol number concentration #/cm3
		
		! OUT
		REAL(KIND=kind_rb), DIMENSION(nx,nzrad+1,nbndsw) :: tau ! optical depth
		REAL(KIND=kind_rb), DIMENSION(nx,nzrad+1,nbndsw) :: ssa ! single scattering albedo
		REAL(KIND=kind_rb), DIMENSION(nx,nzrad+1,nbndsw) :: asm ! asymmetry parameter
		
		! Local
		REAL, EXTERNAL :: qsatw
		REAL :: rh
		INTEGER :: i, k, n, ibin
		
		! Set default value in output array
		! See lines 591-597 in rrtmg_sw_rad.f90 where these values are specified without aerosol
		! (i.e., integer flag, iaer=0). Compare with lines 631-640 with specified aerosol properties
		! (iaer=10). iaer in rrtmg_sw_rad.f90 has been set to 10.
                tau(:,:,:) = 0.0
		ssa(:,:,:) = 1.0
		asm(:,:,:) = 0.0
		
		! Exit if doradaerosimple = .FALSE.
		IF ( .NOT.doradaerosimple ) RETURN
		
		!
		DO n = 1, nbndsw
			DO k = 1, nzm_dyn ! UPTO ACTUAL DOMAIN TOP (NZM <= NZRAD). READ COMMENTS IN RAD_FULL.F90
				DO i = 1, nx
					rh = qv(i,k) / qsatw( tabs(i,k), pres(k) )
					ibin = MINLOC( ABS( rh-rh_sw ), DIM=1 )
					tau(i,k,n) = ext_coef_sw(n,ibin) * nca(i,k) * dz * adz(k) * 1.0E2
					ssa(i,k,n) = ssalbdo(n,ibin)
					asm(i,k,n) = asym_fr(n,ibin)
				ENDDO
			ENDDO
		ENDDO
		
        END SUBROUTINE radaerosimple_sw
	! --------------------------------------------------------------------------

END MODULE rad_driver
