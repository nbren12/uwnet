MODULE rad

	USE grid, ONLY: nx, ny, nzm ! Required for qrad static allocation
	USE parkind, ONLY: kind_rb ! RRTM expects reals with this kind parameter (8 byte reals)
	
	IMPLICIT NONE
	
	! This module holds a bunch of stuff used in the PBL (shallow domain) interface to the RRTM
	! radiation. The PBL interface is based on P. Blossey's GCSS-CFMIP LES interface.
	
	!
	INTEGER, SAVE :: nradsteps = 0 ! Current number of steps done before calling radiation
	
	!
	INTEGER, SAVE :: nzrad, nzpatch, npatch_start, npatch_end
	INTEGER, SAVE :: nzrad_old = -1
!!NOT USED	REAL, SAVE :: day_when_PatchedSounding_last_updated = -10.
	
	! Background sounding, use of default soundings or read from SCAM forcing file
	INTEGER, SAVE :: nzsnd ! number of levels
	INTEGER, SAVE :: ntsnd ! number of time record
	REAL, DIMENSION(:), ALLOCATABLE, SAVE :: psnd ! pressure sounding, mb
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: tsndng ! temperature sounding, K
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: qsndng ! water vapor sounding, kg/kg
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: o3sndng ! ozone mass mixing ratio, kg/kg
	REAL, DIMENSION(:), ALLOCATABLE, SAVE :: tsnd ! time-interpolated temperature
	REAL, DIMENSION(:), ALLOCATABLE, SAVE :: qsnd ! time-interpolatedwater vapor
	REAL, DIMENSION(:), ALLOCATABLE, SAVE :: o3snd ! time-interpolated ozone mass mixing ratio
	LOGICAL :: have_o3mmr ! TRUE if iopfile has ozone profile
	
	! Radiative heating rate (K/s) on model domain
	! Originally ALLOCATABLE but changded to statically allocated array since rad_simple.f90 &
	! rad_simple_smoke.f90 uses it.
	REAL, DIMENSION(nx,ny,nzm), SAVE :: qrad
	
	! Surface and top-of-atmosphere (TOA) radiative fluxes
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: &
		NetlwUpSurface, & ! net longwave up at surface
		NetlwUpSurfaceClearSky, & ! net clearsky longwave up at surface
		NetlwUpToM, & ! net longwave up at Top of Model
		NetlwUpToa, & ! net longwave up at TOA
		NetlwUpToaClearSky, & ! net clearsky longwave up at TOA
		NetswDownSurface, & ! net shortwave down at surface
		NetswDownSurfaceClearSky, & ! net clearsky shortwave down at surface
		NetswDownToM, & ! net shortwave down at Top of Model
		NetswDownToa, & ! net shortwave down at TOA
		NetswDownToaClearSky, & ! net clearsky shortwave down at TOA
		insolation_TOA, & ! shortwave down at TOA
		swDownSurface, & ! shortwave down at surface
		lwDownSurface, & ! longwave down at surface
		swnsxy, lwnsxy ! for ocean evolution
	
	! Effective radius from microphysics routines (if available)
        !bloss(2017/07): Rename rel_rad/rei_rad as rad_reffc/rad_reffi for consistency
        !   with changes in the rest of SAM
	real, dimension(nx, ny, nzm) :: rad_reffc, rad_reffi
	
	! Input CRM/LES fields to radiation
	LOGICAL, SAVE :: isAllocated_RadInputsOutputs = .FALSE.
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: &
		tabs_slice, & ! absolute temperature, K
		qv_slice, &  ! water vapor mass mixing ratio, kg/kg
		qcl_slice, & ! cloud liquid mass mixing ratio, kg/kg
		qci_slice, & ! cloud ice mass mixing ratio, kg/kg
		rel_slice, & ! cloud liquid effective radius, m (TAK)
		rei_slice    ! cloud ice effctive radius, m (TAK)
	
! Tak Yamaguchi 2015/10: RADAEROS
	REAL, DIMENSION(:,:,:), ALLOCATABLE, SAVE :: nca_rad ! aerosol number concentration, #/cm3
	REAL, DIMENSION(:,:), ALLOCATABLE, SAVE :: nca_slice ! aerosol number concentration, #/cm3
	
	REAL, DIMENSION(:), ALLOCATABLE, SAVE :: &
		tg_slice, & ! surface temperature, K
		pres_input, &
		presi_input
	
	! Fluxes output from RRTM radiation scheme
	REAL(KIND=kind_rb), DIMENSION(:,:), ALLOCATABLE, SAVE :: &
		lwUp , & ! upward longwave radiative flux (W/m2)
		lwDown , & ! downward longwave radiative flux (W/m2)
		lwUpClearSky , & ! clearsky upward longwave radiative flux (W/m2)
		lwDownClearSky , & ! clearsky downward longwave radiative flux (W/m2)
		swUp , & ! upward shortwave radiative flux (W/m2)
		swDown , & ! downward shortwave radiative flux (W/m2)
		swUpClearSky , & ! clearsky upward shortwave radiative flux (W/m2)
		swDownClearSky ! clearsky downward shortwave radiative flux (W/m2)
	
	! Conditional average output for clear sky radiative heating rate
	LOGICAL, PARAMETER :: do_output_clearsky_heating_profiles = .TRUE.
	REAL, DIMENSION(nzm) :: radqrclw, radqrcsw ! Domain-average diagnostic fields for clearsky radiation
	
	! Dummy arguments for cloud optics option added by P. Blossey for SAM6.10.10
	! Variables used in /SRC/SIMULATORS/instrument_diagnostics.f90
	! For instrument simulators
	REAL, DIMENSION(nx, ny, nzm) :: tau_067 ! Optical thickness at 0.67 microns
	REAL, DIMENSION(nx, ny, nzm) :: emis_105 ! emissivity at 10.5 microns
	! For MODIS simulator
	REAL, DIMENSION(nx, ny, nzm) :: tau_067_cldliq, tau_067_cldice, tau_067_snow 

	! Effective radius assumed for radiation calculation when microphysics doesn't provide them

	
END MODULE rad
