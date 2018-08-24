c Diagnostics of CAPE, CIN, and othere useful stuff from
c the vertical profiles of temperature and moisture.
c-------------------------------------------------------------------
      subroutine cape_analysis (t,q,p,z,cnt,cnb,capemax,cinmax,nlm)

      implicit none
   
c------------------------ input -----------------------------------

      integer nlm               ! number of middle levels      
      real t(nlm)		! middle level temperature(k)
      real q(nlm)		! middle level water vapor  (g g-1)
      real p(nlm)		! middle level pressure (mb)
      real z(nlm)		! height above sfc at mid-layer (m)

c------------------------ output -----------------------------------

      integer cnt		! top cumulus convection level (index)
      integer cnb 		! bottom cumulus convection level (index)
      real capemax		! maxinmum CAPE
      real cinmax		! corresponding CIN
      
c------------------------ local variables -----------------------
 
      real h(nlm)	! moist static energy of environment (J/kg)
      real hs(nlm)	! saturation moist static energy of environment (J/kg)
      real qs(nlm)      ! saturation mixing ratio of the environment (g g-1)
      real qc(nlm)      ! in-parcel mixing ratio (g g-1)
      real dqs(nlm)     ! dqs/dT
      real dlogp(nlm)   ! Rg * dlog(p)
      real h_crit(nlm)  ! critical effective moist static energy (J/kg)
      real hcc(nlm)	! in-cloud "effective" moist static energy (J/kg)
      real cape(nlm)
      real cin(nlm)
      
      real facb(nlm)	
      real facp(nlm)
      real om(nlm)

      real gam, del(nlm), gamm(nlm), buoy(nlm)
      real gams(nlm)

      real plcl 
      integer i, j, k, l, m, nbase
      integer top, lcl
      real cc00
      real tmp 
      
      real, parameter :: hlat = 2.5e6
      real, parameter :: hlatf = hlat * 2./15. 
      real, parameter :: cp = 1004.
      real, parameter :: Ra = 287.
      real, parameter :: cc0 = 0.e-3
      real, parameter :: grav = 9.81
      real, parameter :: fac_cond =  hlat/cp
      integer :: maxtop ! should be greater than 1 
      integer :: mindepth = 2 ! should be greater than 1
      
      real, external :: qsatw, qsati, dtqsatw, dtqsati
c--------------------------------------------------------------------------- 

	capemax = 0.
	cinmax = 0.
	nbase = 0
	cnt = 1
	cnb = nlm
	maxtop = nlm-2

        do k = 1,maxtop

          om(k) = min(1.,max(0.,0.05*(t(k)-253.)))
c          om(k) = 1.
          qs(k)=om(k)*qsatw(t(k),p(k))+(1.-om(k))*qsati(t(k),p(k))
          dqs(k)=om(k)*dtqsatw(t(k),p(k))+(1.-om(k))*dtqsati(t(k),p(k)) 

          h(k) = grav * z(k) + cp * t(k) + hlat * q(k)
          hs(k)= grav * z(k) + cp * t(k) + hlat * qs(k)
c	  cc00=cc0*max(0.,1.-sqrt(pint(i,k)*1.e-5))
	  cc00=cc0	  
	  facb(k) = 1./(1.+cc00*(z(k+1)-z(k)))
	  facp(k) = 1. - facb(k)
          gam = fac_cond * dqs(k)
          gamm(k)=(1.+(0.61+facb(k))*t(k)*dqs(k)) / (1.+gam) / cp
          gams(k) = gam / (1.+gam) / hlat
          tmp = t(k)/gamm(k)
	  del(k) = tmp*facb(k)
          h_crit(k)= hs(k)-del(k)*qs(k)-0.61*tmp*(qs(k)-q(k))
          dlogp(k) = Ra*log(p(k)/p(k+1))
	end do

c
c     Determine if a column is conditionally unstable:
c
c
	do k = 1,maxtop-mindepth+1  ! loop over cloud bases

	  cape(k) = 0.
	  cin(k) = 0.
	  
	  do l = k+1,maxtop-mindepth+1

            hcc(l) = h(k)-del(l)*q(k) 
	    if(hcc(l).gt.h_crit(l)) then  ! Yes, it is unstable!
c
c  Compute lifting condensation level (assumed to be always 1 level higher
c  than the base level k:
c

	      plcl = min(p(k),
     &               p(k)*(1.-(qs(k)-q(k))/(t(k)*dqs(k)))**3.4965)
              do m=k+1,maxtop
	        if(p(m).le.plcl) then
	          lcl = m
                  EXIT
	        endif	         
	      end do

c
c    If LCL is higher then LFC then ignore the current cloud base:    
c

	      if(lcl.gt.l) CYCLE
	       	
c	    
c     Compute the convective inhibition (CIN) and
c     the convective available potential energy (CAPE) for 
c     a given cloud base level; determine the level of the highest 
c     cloud top for the current cloud base:      
c

              buoy(k) = 0. ! buoyancy at the base 
                           !(some perturbation may be added in the future)

	      do m = k+1,lcl-1  ! dry CIN
               buoy(m) = (h(k)-h(m))/cp-
     &			(fac_cond-0.61*t(m))*(q(k)-q(m)) 
               cin(k) = cin(k) + 0.5*(buoy(m)+buoy(m-1))*dlogp(m-1)
              end do

	      do m = lcl,l-1  ! wet CIN
               buoy(m) = gamm(m)*(hcc(l)-h_crit(m))
               cin(k) =cin(k) + 0.5*(buoy(m)+buoy(m-1))*dlogp(m-1)
              end do

              buoy(l) = gamm(l)*(hcc(l)-h_crit(l))
              tmp = 0.5*(buoy(l)+buoy(l-1))*dlogp(l-1)
              cin(k) = cin(k) + min(0.,tmp)
              cape(k) = cape(k) + max(0.,tmp) 

              top = l
              qc(l) = q(k)
	      do m = l+1,maxtop  ! CAPE
               qc(m) = (qc(m-1) + facp(m)*
     &		  (qs(m)+gams(m)*(h(k)-hs(m))))/(1.+facp(m))
               buoy(m)=gamm(m)*(h(k)-del(m)*qc(m)-h_crit(m))
               if(buoy(m).lt.0.) EXIT
               cape(k) = cape(k) + 0.5*(buoy(m)+buoy(m-1))*dlogp(m-1)
               top = m
              end do
              cape(k) = cape(k) + cin(k)
	      
	      if(cape(k).gt.0..and.cape(k).gt.capemax) then
		  capemax = capemax + cape(k)
		  cinmax = cinmax + cin(k)
	          nbase = nbase + 1 
              cnb = min(cnb, k)
              cnt = max(cnt, top)
	      endif


              EXIT 
             
	     endif ! unstable

	  end do ! l

	end do ! k

	capemax = capemax / (nbase+1.e-5)
	cinmax = cinmax / (nbase+1.e-5)

      return
      end


