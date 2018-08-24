module rad
  use shr_kind_mod, only: r4 => shr_kind_r4
  use grid

implicit none

!--------------------------------------------------------------------
!
! Variables accumulated between two calls of radiation routines


        real(r4) tabs_rad(nx, ny, nzm)       ! accumulated temperature
        real(r4)  qc_rad  (nx, ny, nzm)       ! accumulated cloud water (g/g)
        real(r4)  qi_rad  (nx, ny, nzm)       ! accumulated cloud ice (g/g)
        real(r4)  qv_rad  (nx, ny, nzm)       ! accumulated water vapor (g/g)
        real(r4)  cld_rad  (nx, ny, nzm)      ! accumulated cloud fraction 
        real(r4)  rel_rad  (nx, ny, nzm)      ! accumulated effective radius for liquid water (mkm)
        real(r4)  rei_rad  (nx, ny, nzm)      ! accumulated effective radius for ice water (mkm)
	real qrad    (nx, ny, nzm)	 ! radiative heating(K/s)
	real lwnsxy  (nx, ny)
	real swnsxy  (nx, ny)
	real lwntxy  (nx, ny)
	real swntxy  (nx, ny)
	real lwntmxy (nx, ny)
	real swntmxy (nx, ny)
	real lwnscxy  (nx, ny)
	real swnscxy  (nx, ny)
	real lwntcxy  (nx, ny)
	real swntcxy  (nx, ny)
	real lwdsxy  (nx, ny)
	real swdsxy  (nx, ny)
	real solinxy  (nx, ny)

        ! Fields for radiatively-active snow
        real(r4)  qs_rad  (nx, ny, nzm)       ! accumulated snow mass mixing ratio (g/g)
        real(r4)  res_rad  (nx, ny, nzm)      ! accumulated effective radius for snow (mkm)

    ! Instrument simulator fields
    real tau_067   (nx, ny, nzm)
    real emis_105  (nx, ny, nzm)
    real rad_reffc (nx, ny, nzm)
    real rad_reffi (nx, ny, nzm)

    ! separate optical depths for liquid and ice for MODIS simulator
    real tau_067_cldliq   (nx, ny, nzm) 
    real tau_067_cldice   (nx, ny, nzm)
    real tau_067_snow   (nx, ny, nzm)


	logical initrad		! flag to initialize profiles of traces
	integer nradsteps	! curent number of steps done before
				!   calling radiation routines
	data initrad/.true./

        logical, parameter :: do_output_clearsky_heating_profiles = .true.
        real :: radqrclw(nz), radqrcsw(nz)

!       Gas traces (mass mixing ratios):  
        
        real(r4) o3(nzm)            ! Ozone
        real(r4) n2o(nzm)           ! N2O
        real(r4) ch4(nzm)           ! CH4
        real(r4) cfc11(nzm)         ! CFC11
        real(r4) cfc12(nzm)         ! CFC12

	real(r4) p_factor(nx, ny) ! perpetual-sun factor

end module rad
