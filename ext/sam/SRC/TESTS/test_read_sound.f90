program test_sound
  use read_netcdf_3d
  implicit none

  call test_ncreadvarxzy()

contains
  subroutine test_ncreadvarxzy ()
    character(len=256) :: file = 'test.nc'
    include 'netcdf.inc'
    integer ncid, status
    integer, parameter :: nx = 10, ny = 11, nz = 12
    logical, parameter :: use_nf_real = .true., required= .true.
    real :: f(nx, ny, nz)

    STATUS = NF_OPEN( file, NF_NOWRITE, NCID )
    call ncreadvarxyz(NCID, "U", f, use_nf_real, status, required)
    print *, f(:,1,1)
    print *, 'next column', f(:,2,1)
  end subroutine test_ncreadvarxzy
end program test_sound
