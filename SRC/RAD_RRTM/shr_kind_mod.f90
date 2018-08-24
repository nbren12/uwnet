!===============================================================================
! CVS: $Id: shr_kind_mod.f90,v 1.1 2004/09/08 19:46:04 samcvs Exp $
! CVS: $Source: /home/disk/eos1/samcvs/cvsroot/SAM/SRC/RAD_CAM/shr_kind_mod.f90,v $
! CVS: $Name:  $
!===============================================================================

MODULE shr_kind_mod

  !bloss(120709): Match integer, real kinds with those used in RRTM
  use parkind, only: kind_rb, kind_im

   !----------------------------------------------------------------------------
   ! precision/kind constants add data public
   !----------------------------------------------------------------------------
   public
   integer,parameter :: SHR_KIND_R16= selected_real_kind(24) ! 16 byte real
   integer,parameter :: SHR_KIND_R8 = selected_real_kind(12) ! 8 byte real
   !bloss integer,parameter :: SHR_KIND_R4 = selected_real_kind( 6) ! 4 byte real
   integer,parameter :: SHR_KIND_R4 = kind_rb !bloss(120709): matches RRTM
   integer,parameter :: SHR_KIND_RN = kind(1.0)              ! native real
   integer,parameter :: SHR_KIND_I8 = selected_int_kind (13) ! 8 byte integer
   integer,parameter :: SHR_KIND_I4 = selected_int_kind ( 6) ! 4 byte integer
   !bloss integer,parameter :: SHR_KIND_IN = kind(1)                ! native integer
   integer,parameter :: SHR_KIND_IN = kind_im !bloss(120709): matches RRTM
   integer,parameter :: SHR_KIND_CL = 256                    ! long char
   integer,parameter :: SHR_KIND_CS = 80                     ! short char

END MODULE shr_kind_mod
