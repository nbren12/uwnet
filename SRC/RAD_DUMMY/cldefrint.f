      subroutine cldefrint(m,n,tmp1,t0_down,rel,rei,ps, 
     &                                 pmid,landm,tmp2,tmp3)
C-----------------------------------------------------------------------
C
C interface for cldefr to work with isccp simulator calls and CAM3 radiation
C
C-----------------------------------------------------------------------
      use domain, only: nz_gl
        implicit none
C------------------------------Parameters-------------------------------
        include 'prgrid.h'

C Input arguments
C
      real ps(plond)            ! Surface pressure
      real pmid(plond,plev)     ! Midpoint pressures
      real tmp1(plond), tmp2(plond), tmp3(plond), landm(plond) ! dummy argument
      real t0_down(1, plev)
      integer m, n

C
C Output arguments
C
      real rel(plond,plev)      ! Liquid effective drop size (microns)
      real rei(plond,plev)      ! Ice effective drop size (microns)


      call cldefr(ps ,pmid, rel, rei)

      return
      end
 
