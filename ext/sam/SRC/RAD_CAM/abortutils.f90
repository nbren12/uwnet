!bloss #include <misc.h>

module abortutils
   use ppgrid, only: masterproc

   private
   save

   public :: endrun

CONTAINS

subroutine endrun (msg)
!-----------------------------------------------------------------------
! Purpose:
!
! Abort the model for abnormal termination
!
! Author: CCM Core group
! Modified: Peter Blossey, 08/2004, adapted to SAM
!-----------------------------------------------------------------------
   implicit none
!-----------------------------------------------------------------------
!
! Arguments
!
   character(len=*), intent(in), optional :: msg    ! string to be printed

   if (masterproc) write(*,*) 'ABORT:', msg
   call task_abort()

end subroutine endrun


end module abortutils
