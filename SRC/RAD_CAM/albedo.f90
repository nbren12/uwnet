subroutine albedo(lchnk,ncol, ocean, coszrs, ts, asdir,aldir,asdif,aldif )
  !-----------------------------------------------------------------------
  ! Computes surface albedos over ocean for Slab Ocean Model (SOM)

  ! and the surface (added by Marat Khairoutdinov)

  ! Two spectral surface albedos for direct (dir) and diffuse (dif)
  ! incident radiation are calculated. The spectral intervals are:
  !   s (shortwave)  = 0.2-0.7 micro-meters
  !   l (longwave)   = 0.7-5.0 micro-meters
  !
  ! Uses knowledge of surface type to specify albedo, as follows:
  !
  ! Ocean           Uses solar zenith angle to compute albedo for direct
  !                 radiation; diffuse radiation values constant; albedo
  !                 independent of spectral interval and other physical
  !                 factors such as ocean surface wind speed.
  !
  ! For more details , see Briegleb, Bruce P., 1992: Delta-Eddington
  ! Approximation for Solar Radiation in the NCAR Community Climate Model,
  ! Journal of Geophysical Research, Vol 97, D7, pp7603-7612).
  !
  !---------------------------Code history--------------------------------
  !
  ! Original version:         CCM1
  ! Standardized:             J. Rosinski, June 1992
  ! Reviewed:                 J. Kiehl, B. Briegleb, August 1992
  ! Rewritten for ocean only: J. Rosinski, May 1994
  ! Modified for SOM:         B. Briegleb  April 1995
  ! Reviewed:                 B. Briegleb  March 1996
  ! Reviewed:                 J. Kiehl     April 1996
  ! Reviewed:                 B. Briegleb  May   1996
  ! Added Land albedo	    M. Kairoutdinov Sep. 1999
  !
  !-----------------------------------------------------------------------
  !
  ! $Id: albedo.f90,v 1.1 2004/09/08 19:46:02 samcvs Exp $
  ! $Author: samcvs $
  !
  use shr_kind_mod, only: r4 => shr_kind_r4
  use ppgrid
  implicit none
  !------------------------------Arguments--------------------------------
  !
  ! Input arguments

  !
  integer lchnk,ncol

  real(r4) coszrs(pcols)    ! Cosine solar zenith angle
  real(r4) ts(pcols) ! surface temperature
  logical ocean	! logical switch for ocean
  !
  ! Output arguments
  !
  real(r4) asdir(pcols)     ! Srf alb for direct rad   0.2-0.7 micro-ms
  real(r4) aldir(pcols)     ! Srf alb for direct rad   0.7-5.0 micro-ms
  real(r4) asdif(pcols)     ! Srf alb for diffuse rad  0.2-0.7 micro-ms
  real(r4) aldif(pcols)     ! Srf alb for diffuse rad  0.7-5.0 micro-ms
  !
  !---------------------------Local variables-----------------------------
  !
  integer i             ! Longitude index
  real(r4), parameter :: adif = 0.06
  !
  !
  ! Initialize all ocean surface albedos to zero
  !
  do i=1,ncol
     asdir(i) = 0.
     aldir(i) = 0.
     asdif(i) = 0.
     aldif(i) = 0.
  enddo

  if (ocean) then
     !
     !
     ! Ice-free ocean albedos function of solar zenith angle only, and
     ! independent of spectral interval:
     !
     do i=1,ncol
       if(ts(i).gt.271.) then
        if (coszrs(i).gt.0.) then
           aldir(i) = (.026/(coszrs(i)**1.7 + .065)) + &
                (.15*(coszrs(i) - 0.10)*               &
                (coszrs(i) - 0.50)*                    &
                (coszrs(i) - 1.00)  )
           asdir(i) = aldir(i)
           aldif(i) = adif
           asdif(i) = adif
        endif
       else
         ! albedo of sea-ice/snow surface
           aldir(i) = 0.45
           asdir(i) = 0.75
           aldif(i) = 0.45
           asdif(i) = 0.75
       end if
     enddo

  else ! land

     ! Albedos for land type I (Briegleb)

     do i=1,ncol
        if (coszrs(i).gt.0.) then
           asdir(i) = 1.4 * 0.06 / ( 1. + 0.8 * coszrs(i))
           asdif(i) = 1.2 * 0.06
           aldir(i) = 1.4 * 0.24 / ( 1. + 0.8 * coszrs(i))
           aldif(i) = 1.2 * 0.24
        endif
     enddo

  endif
  !     
  return
end subroutine albedo

