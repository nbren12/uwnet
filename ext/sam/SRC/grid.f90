module grid

use domain
use advection, only: NADV, NADVS

implicit none

character(6), parameter :: version = '6.10.9'
character(8), parameter :: version_date = 'Feb 2016'
        
integer, parameter :: nx = nx_gl/nsubdomains_x
integer, parameter :: ny = ny_gl/nsubdomains_y 
integer, parameter :: nz = nz_gl+1
integer, parameter :: nzm = nz-1
        
integer, parameter :: nsubdomains = nsubdomains_x * nsubdomains_y

logical, parameter :: RUN3D = ny_gl.gt.1
logical, parameter :: RUN2D = .not.RUN3D

integer, parameter :: nxp1 = nx + 1
integer, parameter :: nyp1 = ny + 1 * YES3D
integer, parameter :: nxp2 = nx + 2
integer, parameter :: nyp2 = ny + 2 * YES3D
integer, parameter :: nxp3 = nx + 3
integer, parameter :: nyp3 = ny + 3 * YES3D
integer, parameter :: nxp4 = nx + 4
integer, parameter :: nyp4 = ny + 4 * YES3D

integer, parameter :: dimx1_u = -1                !!-1        -1        -1        -1
integer, parameter :: dimx2_u = nxp3              !!nxp3      nxp3      nxp3      nxp3
integer, parameter :: dimy1_u = 1-(2+NADV)*YES3D  !!1-5*YES3D 1-4*YES3D 1-3*YES3D 1-2*YES3D
integer, parameter :: dimy2_u = nyp2+NADV         !!nyp5      nyp4      nyp3      nyp2
integer, parameter :: dimx1_v = -1-NADV           !!-4        -3        -2        -1
integer, parameter :: dimx2_v = nxp2+NADV         !!nxp5      nxp4      nxp3      nxp2
integer, parameter :: dimy1_v = 1-2*YES3D         !!1-2*YES3D 1-2*YES3D 1-2*YES3D 1-2*YES3D
integer, parameter :: dimy2_v = nyp3              !!nyp3       nyp3      nyp3      nyp3
integer, parameter :: dimx1_w = -1-NADV           !!-4        -3        -2        -1
integer, parameter :: dimx2_w = nxp2+NADV         !!nxp5      nxp4      nxp3      nxp2
integer, parameter :: dimy1_w = 1-(2+NADV)*YES3D  !!1-5*YES3D 1-4*YES3D 1-3*YES3D 1-2*YES3D
integer, parameter :: dimy2_w = nyp2+NADV         !!nyp5      nyp4      nyp3      nyp2
integer, parameter :: dimx1_s = -2-NADVS          !!-4        -3        -2        -2
integer, parameter :: dimx2_s = nxp3+NADVS        !!nxp5      nxp4      nxp3      nxp3
integer, parameter :: dimy1_s = 1-(3+NADVS)*YES3D !!1-5*YES3D 1-4*YES3D 1-3*YES3D 1-3*YES3D
integer, parameter :: dimy2_s = nyp3+NADVS        !!nyp5      nyp4      nyp3      nyp3

integer, parameter :: ncols = nx*ny
integer, parameter :: nadams = 3

! Vertical grid parameters:
real z(nz)      ! height of the pressure levels above surface,m
real pres(nzm)  ! pressure,mb at scalar levels
real zi(nz)     ! height of the interface levels
real presi(nz)  ! pressure,mb at interface levels
real adz(nzm)   ! ratio of the thickness of scalar levels to dz 
real adzw(nz)	! ratio of the thinckness of w levels to dz
real pres0      ! Reference surface pressure, Pa

integer:: nstep =0! current number of performed time steps 
integer  ncycle  ! number of subcycles over the dynamical timestep
integer icycle  ! current subcycle 
real cfl_adv    ! CFL due to advection
real cfl_sgs    ! CFL due to SGS
integer:: na=1, nb=2, nc=3 ! indeces for swapping the rhs arrays for AB scheme
real at, bt, ct ! coefficients for the Adams-Bashforth scheme 
real dtn	! current dynamical timestep (can be smaller than dt)
real dt3(3) 	! dynamical timesteps for three most recent time steps
real(8):: time=0.	! current time in sec.
real day	! current day (including fraction)
real dtfactor   ! dtn/dt
        
!  MPI staff:     
integer rank   ! rank of the current subdomain task (default 0) 
integer ranknn ! rank of the "northern" subdomain task
integer rankss ! rank of the "southern" subdomain task
integer rankee ! rank of the "eastern"  subdomain task
integer rankww ! rank of the "western"  subdomain task
integer rankne ! rank of the "north-eastern" subdomain task
integer ranknw ! rank of the "north-western" subdomain task
integer rankse ! rank of the "south-eastern" subdomain task
integer ranksw ! rank of the "south-western" subdomain task
logical dompi  ! logical switch to do multitasking
logical masterproc ! .true. if rank.eq.0 
	
character(80) case   ! id-string to identify a case-name(set in CaseName file)

logical dostatis     ! flag to permit the gathering of statistics
logical dostatisrad  ! flag to permit the gathering of radiation statistics
integer nstatis	! the interval between substeps to compute statistics

logical :: compute_reffc = .false. 
logical :: compute_reffi = .false. 
logical :: compute_reffl = .false. !bloss(2018-02): Include rain/drizzle as radiatively active 

logical notopened2D  ! flag to see if the 2D output datafile is opened	
logical notopened3D  ! flag to see if the 3D output datafile is opened	
logical notopenedmom ! flag to see if the statistical moment file is opened

!-----------------------------------------
! Parameters controled by namelist PARAMETERS

real:: dx =0. 	! grid spacing in x direction
real:: dy =0.	! grid spacing in y direction
real:: dz =0.	! constant grid spacing in z direction (when dz_constant=.true.)
logical:: doconstdz = .false.  ! do constant vertical grid spacing set by dz

integer:: nstop =0   ! time step number to stop the integration
integer:: nelapse =999999999! time step number to elapse before stoping
integer:: nelapsemin=999999999 !bloss: number of wallclock minutes to run before stopping

real:: dt=0.	! dynamical timestep
real:: day0=0.	! starting day (including fraction)

integer:: nrad =1  ! frequency of calling the radiation routines
integer:: nprint =1000	! frequency of printing a listing (steps)
integer:: nrestart =0 ! switch to control starting/restarting of the model
integer:: nstat =1000	! the interval in time steps to compute statistics
integer:: nstatfrq =50 ! frequency of computing statistics 

logical:: restart_sep =.false.  ! write separate restart files for sub-domains
integer:: nrestart_skip =0 ! number of skips of writing restart (default 0)
logical:: output_sep =.false.   ! write separate 3D and 2D files for sub-domains

character(80):: caseid =''! id-string to identify a run	
character(80):: caseid_restart =''! id-string for branch restart file 
character(80):: case_restart =''! id-string for branch restart file 

logical:: doisccp = .false.
logical:: domodis = .false.
logical:: domisr = .false.
logical:: dosimfilesout = .false.

logical:: doSAMconditionals = .false. !core updraft,downdraft conditional statistics
logical:: dosatupdnconditionals = .false.!cloudy updrafts,downdrafts and cloud-free
logical:: doscamiopdata = .false.! initialize the case from a SCAM IOP netcdf input file
logical:: dozero_out_day0 = .false.
character(len=120):: iopfile=''
character(256):: rundatadir ='./RUNDATA' ! path to data directory

integer:: nsave3D =1000     ! frequency of writting 3D fields (steps)
integer:: nsave3Dstart =99999999! timestep to start writting 3D fields
integer:: nsave3Dend  =99999999 ! timestep to end writting 3D fields
logical:: save3Dbin =.true.   ! save 3D data in binary format(no 2-byte compression) -- Changed from SAM default of false
logical:: save3Dsep =.false.   ! use separate file for each time point for2-model
real   :: qnsave3D =0.    !threshold manimum cloud water(kg/kg) to save 3D fields
logical:: dogzip3D =.false.    ! gzip compress a 3D output file   
logical:: rad3Dout = .false. ! output additional 3D radiation foelds (like reff)

integer:: nsave2D =1000     ! frequency of writting 2D fields (steps)
integer:: nsave2Dstart =99999999! timestep to start writting 2D fields
integer:: nsave2Dend =99999999  ! timestep to end writting 2D fields
logical:: save2Dbin =.true.   ! save 2D data in binary format, rather than compressed -- Changed from SAM default of false
logical:: save2Dsep =.false.   ! write separate file for each time point for 2D output
logical:: save2Davg =.false.   ! flag to time-average 2D output fields (default .false.)
logical:: dogzip2D =.false.    ! gzip compress a 2D output file if save2Dsep=.true.   

integer:: nstatmom =1000! frequency of writting statistical moment fields (steps)
integer:: nstatmomstart = 1 ! timestep to start writting statistical moment fields
integer:: nstatmomend = 0  ! timestep to end writting statistical moment fields
logical:: savemomsep =.false.! use one file with stat moments for each time point
logical:: savemombin =.false.! save statistical moment data in binary format

integer:: nmovie =1000! frequency of writting movie fields (steps)
integer:: nmoviestart =99999999! timestep to start writting statistical moment fields
integer:: nmovieend =99999999  ! timestep to end writting statistical moment fields

logical :: isInitialized_scamiopdata = .false.
logical :: wgls_holds_omega = .false.

!bloss: options to change flux divergence in simple radiation scheme.
real :: rad_simple_fluxdiv1 = 70. ! W/m2
real :: rad_simple_fluxdiv2 = 22. ! W/m2
real :: rad_simple_kappa = 85. ! m2/kg

!bloss: option to use fixed divergence and to balance subsidence
!   heating above the inversion with cooling based on a fixed lapse 
!   rate (used in Uchida papers)
logical :: dofixdivg = .false. 
logical :: do_linear_subsidence = .true.  ! if both true, w_ls=divg_ls*z
real :: divg_ls = 3.75e-6 ! in 1/s, w_ls = -D*z, where [z] = m, [w] = m/s
real :: divg_lapse = 6.e-3 ! lapse rate of dry static energy above inversion

!bloss: option to use extrapolation within upperbound routine, even if
!  dolargescale==.true.  Useful for better consistency of water vapor
!  isotopologues at upper boundary
logical :: doExtrapolate_UpperBound = .false.

!bloss: option to fix wind speed used in calculation of surface fluxes
logical :: doFixedWindSpeedForSurfaceFluxes = .false.
real :: WindSpeedForFluxes = 8.

!bloss: option to run Derbyshire et al (2004) cases
logical :: doDerbyshire = .false.

! heights for bottom of nudging, top of transition layer, top of nudging
real :: Derbyshire_z1 = 1.e3, Derbyshire_z2 = 2.e3, Derbyshire_z3 = 15.e3

! Relative humidity for nudging between z1 and z2 (Low) and z2 and z3 (High)
!    In Derbyshire et al (2004), RelH_Low=0.8 for all cases and
!    RelH_High = 0.25, 0.5, 0.7 and 0.9
real :: Derbyshire_RelH_Low = 0.8, Derbyshire_RelH_High = 0.25

! Theta profile is defined as theta(z) = theta0 + LapseRate*z
!   with theta0 in K, LapseRate for theta in K/m.
real :: Derbyshire_theta0 = 290., Derbyshire_LapseRate = 3.e-3

real :: Derbyshire_tau = 3600. ! nudging timescale is one hour.

! option to require that advective fluxes be computed everywhere.
!   In ADV_UM5b, advective fluxes and updates are not computed
!   if there are chunks of the domain at the top/bottom with
!   only zero values.
logical :: compute_advection_everywhere = .false.

real :: wsub_ref(nz) = 0. ! used to output reference large-scale vertical motion
real :: w_wtg(nz) ! for statistics output

logical :: dowtg_blossey_etal_JAMES2009 = .false. ! alternate implementation
real :: am_wtg  = 2. ! momentum damping rate in 1/d -- note must be non-zero.
            ! default= 2.
real :: am_wtg_exp = 0. ! exponent of p/p0 in momentum damping rate.
real :: lambda_wtg = 650.e3 ! quarter wavelength in m. default = 650.e3 (=650 km).

logical :: dowtg_qnudge = .false. ! if T, nudge q profile to observations on below timescale
real :: itau_wtg_qnudge = 0. ! inverse nudging timescale for q in 1/day.

real :: tauz0_wtg_qnudge = 3.e3
real :: taulz_wtg_qnudge = 1.5e3

logical :: dowtg_tnudge = .false. ! enable temperature nudging above tropopause
real :: itau_wtg_tnudge = 0. ! inverse nudging timescale for temperature, 1/day
real :: taulz_wtg_tnudge = 1.5e3 ! length scale for ramping up t nudging in vertical.

! If true, use Dge computed within microphysics for computing radiative properties
!   of snow and cloud ice
logical :: doDge_SnowAndIce = .false.

! If true, use raw Thompson reff values as Dge for computing radiative properties
!   of snow and cloud ice.  I believe that this is inconsistent with the definition of
!   size in the radiation scheme, but it follows the approach in Thompson
!   et al (2016, Atmos Res).  Ignored if doDge_SnowAndIce==.true.
logical :: doThompsonReffIce = .false.

! Noah additions:
character(len=120) :: initial_condition_netcdf = ''

!-----------------------------------------
end module grid
