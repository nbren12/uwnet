module micro_params
  !bloss: Holds microphysical parameter settings separately, so that they can be used
  !   inside both the microphysics module and the module(s) from WRF.

  logical :: doicemicro = .true. ! Turn on ice microphysical processes

  logical :: doaerosols = .false. ! Use Thompson-Eidhammer water- and ice-friendly aerosols

  logical :: doisotopes = .false. ! Enable water isotopologues

  real :: Nc0 = 100. ! initial/specified cloud droplet number concentration (#/cm3).

  ! option to allow the gamma exponent for rain, graupel and cloud ice to be specified.
  !   Note that mu=0 (the default) corresponds to an exponential distribution.
  real :: fixed_mu_r = 0.
  real :: fixed_mu_i = 0.
  real :: fixed_mu_g = 0.

  ! Fix the exponent in the gamma droplet size distribution for cloud liquid to a single value.
  logical :: dofix_mu_c = .false. 
  real :: fixed_mu_c = 10.3 ! fixed value from Geoffroy et al (2010).
  ! Full citation: Geoffroy, O., Brenguier, J.-L., and Burnet, F.:
  !   Parametric representation of the cloud droplet spectra for LES warm
  !   bulk microphysical schemes, Atmos. Chem. Phys., 10, 4835-4848,
  !   doi:10.5194/acp-10-4835-2010, 2010.

  !..Densities of rain, snow, graupel, and cloud ice. --> Needed in radiation
        REAL, PARAMETER :: rho_water = 1000.0
        REAL, PARAMETER :: rho_snow = 100.0
        REAL, PARAMETER :: rho_graupel = 500.0
        REAL, PARAMETER :: rho_cloud_ice = 890.0

  ! Fix rain number generation from melting snow (backported from WRF V3.9)      
  logical :: BrownEtAl2017_pnr_sml_fix = .true. 

  !bloss(2018-02): Enable choice of snow moment
  !parameterizations between Field et al (2005), the default,
  !and Field et al (2007).
  logical :: doFieldEtAl2007Snow = .false.
  
  ! Field et al (2007) has two snow size distributions: tropical and mid-latitude.
  !   The size distribution is used in the computation of sedimentation.
  logical :: TropicalSnow = .true. ! if false, use mid-latitude size distribution

  logical :: do_output_process_rates = .false.
  integer, parameter ::  nproc_rates_mass_thompson_cold = 31, nproc_rates_number_thompson_cold = 17
  integer, parameter ::  nproc_rates_mass_thompson_warm = 11, nproc_rates_number_thompson_warm = 5
  integer ::  nproc_rates_mass = 1 , nproc_rates_number = 1 ! default value is one

  character*80 :: lookup_table_location = './RUNDATA/'

end module micro_params
