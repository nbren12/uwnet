      subroutine cldefr(ps      ,pmid,  rel     ,rei)
C-----------------------------------------------------------------------
C
C Compute cloud drop size
C
C---------------------------Code history--------------------------------
C
C Original version:  J. Kiehl, Jan 1993
C Reviewed:          J. Kiehl, Feb 1996
C Standardized:      L. Buja,  Feb 1996
C Reviewed:          J. Kiehl, Apr 1996
C
C-----------------------------------------------------------------------
c
c $Id: cldefr.f,v 1.1.1.1 1999/06/08 16:15:43 marat Exp $
c
C-----------------------------------------------------------------------
      use domain, only: nz_gl
 	implicit none
C------------------------------Parameters-------------------------------
	include 'prgrid.h'
C------------------------------Arguments--------------------------------
C
C Input arguments
C
      real ps(plond)            ! Surface pressure
      real pmid(plond,plev)     ! Midpoint pressures
C
C Output arguments
C
      real rel(plond,plev)      ! Liquid effective drop size (microns)
      real rei(plond,plev)      ! Ice effective drop size (microns)
c------------------------------------------------
      real pirnge               ! Nrmlzd pres range for ice particle changes
      real picemn               ! Normalized pressure below which rei=reimax
      real rirnge               ! Range of ice radii (reimax - 10 microns)
      real reimax               ! Maximum ice effective radius
      real pnrml                ! Normalized pressure
      real weight               ! Coef. for determining rei as fn of P/PS
C
C---------------------------Local workspace-----------------------------
C
      integer i,k               ! Lon, lev indices
C
      do k=1,plev
        do i=1,plon

          rel(i,k) = 10.
C
C Determine rei as function of normalized pressure
C
          reimax   = 30.0
          rirnge   = 20.0 
          pirnge   = 0.4
          picemn   = 0.4

          pnrml    = pmid(i,k)/ps(i)
          weight   = max(min((pnrml-picemn)/pirnge,1.0),0.)
          rei(i,k) = reimax - rirnge*weight

        end do
      end do
C
      return
      end
 
