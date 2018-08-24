module micro_params

use grid, only: nzm

implicit none

!  Microphysics stuff:

! Densities of hydrometeors

real, parameter :: rhor = 1000. ! Density of water, kg/m3

real, parameter :: qp_threshold = 1.e-12 ! minimal rain/snow water content

real, parameter :: rd_min = 25.e-6 ! Minimum drizzle drop size
real, parameter :: rd_max = 200.e-6 ! Maximum drizzle volume radius
real, parameter :: coefrv = 3./(4.*3.1415*1000.) ! coef to compute volume radius: rv = (coefrv * qr * N)^1/3
real, parameter :: coefconpmin = coefrv/rd_max**3
real, parameter :: coefconpmax = coefrv/rd_min**3

! Note: Using the MICRO_DRIZZLE namelist, the user can specify:
!    + the cloud droplet number concentration (Nc0, in units of #/cm3),
!    + the geometric standard deviation (sigmag, unitless) of the assumed
!        lognormal droplet size distribution, and 
!    + whether to use the computed cloud droplet effective radius in
!        the radiation scheme (douse_reffc, logical).
!
real :: Nc0 = 40. ! Prescribed cloud drop concentration, cm-3
real :: sigmag = 1.5 ! geometric standard deviation of cloud drop size distribution
logical :: douse_reffc = .true. ! compute cloud droplet effective radius from drop size distribution.

! Possible values for sigmag:
!   1.5 (default): Ackerman et al (2004, Nature, doi:10.1038/nature03174)
!                  Ackerman et al (2009, MWR, doi:10.1175/2008MWR2582.1)
!   1.34: Geoffroy et al (2010, ACP, doi:10.5194/acp-10-4835-2010)
!       Observations from various field campaigns.  See their table 2.
!   1.2: Van Zanten et al (2005, MWR, doi:10.1175/2008MWR2582.1)
!       DYCOMS observations.  See their figure 5d.
!
! The value of Nc0 affects the autoconversion rate, the cloud droplet sedimentation rate and the 
!    cloud droplet effective radius.  The value of sigmag affects sedimentation and the effective radius.
!
!  Using douse_reffc==.true. will only affect results if using full interactive radiation.

real evapr1(nzm),evapr2(nzm)

!bloss: The following parameters are needed for the cloud optics schemes in M2005 and Thompson.
!  However, we use the rho_water in MICRO_DRIZZLE in the computation of effective radius
!  and cloud droplet sedimentation.
real, parameter :: rho_snow = 100., rho_water = 1000., rho_cloud_ice = 500.

end module micro_params
