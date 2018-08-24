      subroutine pres2z(ps, pm,  rair, gravit, tv, z, nlev)
C-----------------------------------------------------------------------
C
C Compute the geopotential height *ABOVE THE SURFACE* at layer
C midpoints from the virtual temperatures and pressures.
c The code is a modified version of the subroutine zmid in the NCAR
c Community Climate Model v3.6.
C
C------------------------------Arguments--------------------------------
C
C Input arguments
C
      implicit none

      integer nlev	  ! number of levels
      real ps             ! Surface pressures, mb
      real pm(nlev)       ! Midpoint pressures, mb
      real rair           ! Gas constant for dry air
      real gravit         ! Acceleration of gravity
      real tv(nlev)       ! Virtual temperature
C
C Output arguments
C
      real z(nlev)        ! Height above surface at midpoints
C
C---------------------------Local variables-----------------------------
C
      integer i,k,l              ! Lon, level, level indices
      real rog                   ! Rair / gravit
      real pmln(nlev)            ! Log of midpoint pressures
      real psln                  ! Log surface pressures
C
C-----------------------------------------------------------------------
      do k=1,nlev
          pmln(k) = log(pm(k)*100.)
      end do
      psln=log(ps*100.)
C
C Diagonal term of hydrostatic equation
C
      rog = rair/gravit
      do k=2,nlev
          z(k) = rog*tv(k)*0.5*(pmln(k-1) - pmln(k))
      end do
      z(1) = rog*tv(1)*(psln - pmln(1))
C
C Bottom level term of hydrostatic equation
C
      do  k=2,nlev
          z(k) = z(k) + rog*tv(1)*(psln - 0.5*(pmln(2) + pmln(1)))
      end do
C
C Interior terms of hydrostatic equation
C
      do k=3,nlev
        do l=2, k-1
            z(k) = z(k) + rog*tv(l) * 0.5*(pmln(l-1) - pmln(l+1))
        end do
      end do

      return
      end
 
