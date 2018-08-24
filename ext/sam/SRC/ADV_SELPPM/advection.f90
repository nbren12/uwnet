module advection

  implicit none

  integer, parameter :: NADV = 2 ! added for compatibility with current version of SAM
  integer, parameter :: NADVS = 1 ! added for compatibility with current version of SAM
  integer, parameter :: npad_s = 4 ! pad scalar array with four ghost cells
  integer, parameter :: npad_uv = 2 ! pad velocity arrays with two ghost cells in direction of advection.

end module advection

