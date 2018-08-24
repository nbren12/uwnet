MODULE module_wrf_error
  INTEGER           :: wrf_debug_level = 0
  CHARACTER*256     :: wrf_err_message

CONTAINS
  subroutine wrf_debug(errnum, error_message)
    implicit none
    integer :: errnum
    CHARACTER(LEN=*) error_message

!bloss    write(*,*) 'Error number = ', errnum
    write(*,*) error_message
  end subroutine wrf_debug
end MODULE module_wrf_error
