module rad

use grid

implicit none

!--------------------------------------------------------------------
!
! Variables accumulated between two calls of radiation routines


	real lwnsxy  (nx, ny)
	real swnsxy  (nx, ny)
	real lwntxy  (nx, ny)
	real swntxy  (nx, ny)
	real lwnscxy  (nx, ny)
	real swnscxy  (nx, ny)
	real lwntcxy  (nx, ny)
	real swntcxy  (nx, ny)
	real lwdsxy  (nx, ny)
	real swdsxy  (nx, ny)
	real solinxy  (nx, ny)

 ! variables that need to exist in order for simulators to compile.
 real, allocatable, dimension(:,:,:) :: tau_067, emis_105, &
                    tau_067_cldliq, tau_067_cldice, tau_067_snow, &
                    rad_reffc, rad_reffi 

 ! variables needed so that clearsky heating rates can be output from other radiation schemes.
 logical, parameter :: do_output_clearsky_heating_profiles = .false.
 real, dimension(nz) :: radqrclw, radqrcsw

end module rad
