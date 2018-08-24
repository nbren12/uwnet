!bloss #include <misc.h>
!bloss #include <params.h>

module ppgrid
  use shr_kind_mod, only: r4 => shr_kind_r4
  use grid, only: nz_gl, masterproc, day, dompi, rundatadir
  !----------------------------------------------------------------------- 
  ! 
  ! Purpose: 
  ! Initialize physics grid resolution parameters
  !  for a chunked data structure
  ! 
  ! Author: 
  ! 
  !-----------------------------------------------------------------------

  !bloss

  !
  ! Grid point resolution parameters
  !
  integer pcols      ! number of columns (max)
  integer pver       ! number of vertical levels
  integer pvermx     ! number of subsurface levels
  integer pverp      ! pver + 1

  integer plev       ! number of vertical levels
  integer plevp      ! plev + 1

  parameter (pcols  = 1)
  parameter (pver   = nz_gl)
  parameter (pvermx = 4)
  parameter (pverp  = pver + 1  )

  parameter (plev   = pver)
  parameter (plevp  = pverp)
  integer, parameter :: iyear = 1999
  !
  !flag for abs/ems computation
  !
  logical doabsems
  !
  !flag for volcanic aerosols (taken from prescribed_aerosols.f90)
  !
  logical, parameter :: strat_volcanic = .false.
  !
  !key parameters (taken from crdcon.h)
  !
  real(r4) gravit     ! Acceleration of gravity
  real(r4) rga        ! 1./gravit
  real(r4) cpair      ! Specific heat of dry air
  real(r4) epsilo     ! Ratio of mol. wght of H2O to dry air
  real(r4) sslp       ! Standard sea-level pressure
  real(r4) stebol     ! Stefan-Boltzmann's constant
  real(r4) rgsslp     ! 0.5/(gravit*sslp)
  real(r4) dpfo3      ! Voigt correction factor for O3
  real(r4) dpfco2     ! Voigt correction factor for CO2
  real(r4) dayspy     ! Number of days per 1 year
  real(r4) pie        ! 3.14.....
  !
  integer ntoplw      ! top level to solve for longwave cooling
  !
  !input filenames (taken from filename module)
  character(LEN=256) :: absems_data = '/abs_ems_factors_fastvx.c030508.nc'
  character(LEN=256) :: aeroptics   = '/AerosolOptics_c040105.nc'
  !
  !molecular weights (taken from shrc_onst_mod.F90 and physconst.F90)
  !
  real(r4), public, parameter :: mwdry =  28.966_r4 ! molecular weight dry air
  real(r4), public, parameter :: mwco2 =  44.       ! molecular weight co2
  real(r4), public, parameter :: mwh2o =  18.016_r4 ! molecular weight h2o
  real(r4), public, parameter :: mwn2o =  44.       ! molecular weight n2o
  real(r4), public, parameter :: mwch4 =  16.       ! molecular weight ch4
  real(r4), public, parameter :: mwf11 = 136.       ! molecular weight cfc11
  real(r4), public, parameter :: mwf12 = 120.       ! molecular weight cfc12
  !
  !co2 parameters (taken from ghg_surfvals module)
  !
  real(r4) :: co2vmr       ! co2   volume mixing ratio 
  real(r4), parameter :: rmwco2 = mwco2/mwdry    ! ratio of molecular weights
  ! of co2 to dry air 
  real(r4) :: co2mmr  ! co2   mass mixing ratio
  !
  !Earth's orbital characteristics
  !
  real(r4), parameter :: scon = 1.367e6 ! solar constant (cgs units)
  real(r4) eccf  ! eccentricity factor (1./earth-sun dist^2)
  real(r4) eccen ! Earth's eccentricity factor (unitless) (typically 0 to 0.1)
  real(r4) obliq ! Earth's obliquity angle (deg) (-90 to +90) (typically 22-26)
  real(r4) mvelp ! Earth's moving vernal equinox at perhelion (deg)(0 to 360.0)
!  integer iyear_AD ! Year (AD) to simulate above earth's orbital parameters
  !
  ! Orbital information after processed by orbit_params
  !
  real(r4) obliqr      ! Earth's obliquity in radians
  real(r4) lambm0      ! Mean longitude of perihelion at the
  !                          ! vernal equinox (radians)
  real(r4) mvelpp      ! Earth's moving vernal equinox longitude
  !                          ! of perihelion plus pi (radians)
  !
  ! constants for ozone path length integrals
  !
  real(r4) cplos    ! constant for ozone path length integral
  real(r4) cplol    ! constant for ozone path length integral
  !
  ! aerosol indices (taken from prescribed_aerosols module)
  !
  integer, parameter :: naer_all = 12  ! naer_all is total number of species
  ! naer is number of species in climatology
  ! naer_all = naer + 1 (background "species") + 1 (volcanic)

  ! indices to aerosol array (species portion)
  integer, public, parameter :: &
       idxSUL   =  1, &
       idxSSLT  =  2, &
       idxOCPHO =  7, &
       idxBCPHO =  8, &
       idxOCPHI =  9, &
       idxBCPHI = 10, &
       idxBG    = 11, &
       idxVOLC  = 12

  ! indices to sections of array that represent 
  ! groups of aerosols
  integer, public, parameter :: &
       idxDUSTfirst    = 3, &
       numDUST         = 4, &
       idxCARBONfirst = 7, &
       numCARBON      = 4
  !
  ! size parameters for chunks of data in x,y directions used in
  ! computation of absorbtivity/emissivity computations
  !
  integer, parameter :: ndiv = 2
  integer nxdiv, nydiv
  !
  ! start, end indices for chunks owned by a given MPI task
  ! (set in phys_grid_init).
  !
  integer :: begchunk, endchunk   ! 

end module ppgrid
