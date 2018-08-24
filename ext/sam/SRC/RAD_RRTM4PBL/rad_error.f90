subroutine rad_error()
  implicit none

  write(*,*) 'Error encountered in the PBL interface to RRTM '
  write(*,*) 'Stopping model...'

  ! PORTABILITY NOTE: EITHER CHANGE THE FOLLOWING TO STOP
  !   OR CALL YOUR OWN ROUTINE THAT STOPS THE MODEL NICELY.
  call task_abort()
  ! STOP

end subroutine rad_error
