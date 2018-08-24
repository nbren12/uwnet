c-------------------------------------------------------------------
      subroutine cupmcb 
     *  (tl, ql, pmid, pint, pdel, zint, zmid, dt, pblh, 
     *   tpert, qpert, cnt, cnb, dqccup, dqicup, dqvcup, dtcup, dmcup,
     *   cuprec, mc, capemax, cinmin, nlen, len, nlm)

      implicit none
   
c-------------------------------------------------------------------
c     Arakawa-Schubert cumulus convection with multiple cloud bases.
c     Exponential mass-flux profile; diagnostic mass-flux closure 
c     assuming quasistationary cumulus kinetic energy.
c     Original version was coded for CCM3.6.6, April-June 1999.
c     Author: Marat Khairoutdinov, CSU
c--------------------------------------------------------------------------
      include "cup.h"
c------------------------ input -----------------------------------

      integer nlen              ! first dimension
      integer len               ! actual number of columns
      integer nlm               ! number of middle levels
      
      real tl(nlen, nlm)	! middle level temperature(k)
      real ql(nlen,nlm)		! middle level water vapor  (g g-1)
      real pmid(nlen,nlm)	! middle level pressure (pa)
      real pint(nlen,nlm+1)	! interface level pressures (pa)
      real pdel(nlen,nlm)	! layer pressure thickness
      real zint(nlen,nlm)	! height above sfc at interface (m)
      real zmid(nlen,nlm)	! height above sfc at mid-layer (m)
      real pblh(nlen)		! boundary layer height (m)	
      real tpert(nlen)		! PBL temperature perturbation (K)
      real qpert(nlen)    	! PBL moisture perturbation (kg/kg)
      real dt			! timestep (s)
      	
c------------------------ output -----------------------------------

      real dqccup(nlen,nlm)	! detrain. rate of liquid water (g g-1 s-1)
      real dqicup(nlen,nlm)	! detrain. rate of ice (g g-1 s-1)
      real dtcup(nlen,nlm)	! cumulus heating rate (K/s)
      real dqvcup(nlen,nlm)	! prod. rate of water vapor (g g-1 s-1)
      real dmcup(nlen,nlm)	! mass detrainment rate (s-1)
      real cuprec(nlen)		! cumulus precipitation rate (m/s)
      real mc(nlen, nlm)	! cumulus mass flux (kg/m2/s)
      real cnt(nlen)		! top cumulus convection level (index)
      real cnb(nlen) 		! bottom cumulus convection level (index)
      real capemax(nlen)	! maxinmum CAPE
      real cinmin(nlen)		! corresponding CIN
      
c------------------------ local variables -----------------------

      real t2(nlen,nlm)         ! interface temperature (K)
      real q2(nlen,nlm)         ! interface mixing ratio (g g-1)
      real qss(nlen,nlm)        ! interface saturation mixing ratio (g g-1)
      real ess(nlen,nlm)        ! interface saturation vapor pressure (Pa)


c   At interfaces:
 
      real h(nlm)	! moist static energy of environment (J/kg)
      real hs(nlm)	! saturation moist static energy of environment (J/kg)
      real hc(nlm)	! in-cloud moist static energy (J/kg)

      real q(nlm) 	! water vapor mixing ratio of the environment (g g-1)
      real qs(nlm)      ! saturation mixing ratio of the environment (g g-1)
      real qc(nlm)	! in-cloud total water mixing ratio (g g-1)
      real dqs(nlm)     ! dqs/dT

      real t(nlm) 	! environment temperature (K)

      real z(nlm+1)	! height above the surface (m)
      real dz(nlm) 	! layer thikness between interfaces (m)
      real imass(nlm)   ! inverse middle-layer mass (g/dP)
      real dlogp(nlm)   ! Rg * dlog(p)


      real h_crit(nlm)  ! critical effective moist static energy (J/kg)
      real hcc(nlm)	! in-cloud "effective" moist static energy (J/kg)
      real cape(nlm)    ! CAPE (J/kg)
      real cin(nlm)     ! CIN (J/kg)
 
      real nu(nlm)      ! normalized mass flux
      real qliq(nlm)    ! in-cloud liquid water (g g-1)
      real qice(nlm)    ! in-cloud ice (g g-1)
      real prec(nlm)    ! precipitate

      real facb(nlm)	
      real facp(nlm)
      real om(nlm)
      real facmu, facmuh, facmuq, facmul  
      real fluxh(nlm+1), fluxq(nlm+1), pdry(nlm+1)
      real gam, del(nlm), gamm(nlm), buoy(nlm)
      real gams(nlm)

      real lamda, dlamda, cwf
      real hcloud, dhcloud, dqcloud, hwave, qwave
      real alphi, mb, dqcup, dhcup, sumq, sumh
      real rat(nlen), drying

      real a, b, capp, fac_cond, plcl, hlatf
      integer i, j, k, l, m, nbases, base(nlm), lnb(nlm), lfc(nlm)
      integer niter, top, lcl, maxtop, mindepth, pbltop, pblbase
      logical condition
      real dqvmax, cc0, cc00, lamdaup, maxmc
      integer nclouds
      real tmp, c1, c2, c3, tpertmax, qpertmax, factor
      real hprime(nlm)
      real qprime(nlm)
      real alpha_factor(nlm)	
c--------------------------------------------------------------------------- 
      parameter (maxtop=2)   ! should be greater than 1 !
      parameter (mindepth=2) ! should be greater than 1 !
      parameter (dqvmax=0.25) 
      parameter (cc0=2.e-3)
      parameter (maxmc=0.05) ! maximum allowed mass flux
      parameter (tpertmax=2.0)
      parameter (qpertmax=2.0e-3)
c--------------------------------------------------------------------------- 

      hlatf = hlat * 2./15. ! latent heat of fusion
      fac_cond = hlat/cp	      
      alphi = taudis/alpham

c
c Compute temperature and mixing ratio at interface levels:
c

      do k=1,nlm
       do i=1,len
	 qss(i,k) = (pmid(i,k)*1.e-5)**0.286
	 ess(i,k) = (pint(i,k)*1.e-5)**0.286
       end do
      end do	
      do k = maxtop,nlm
       do i=1,len
          a  = (ess(i,k) - qss(i,k-1)) / (qss(i,k) - qss(i,k-1))
          b = 1. - a
          t2(i,k)=(a*tl(i,k-1)/qss(i,k-1)+b*tl(i,k)/qss(i,k))*ess(i,k)
          q2(i,k)=a * ql(i,k-1) + b * ql(i,k)
       end do
      end do
      do k = 1,maxtop-1
       do i=1,len
        t2(i,k)=tl(i,k)
        q2(i,k)=ql(i,k)
       end do
      end do

c
c  Compute water vapor saturation stuff:
c

      call aqsat(t2, pint, ess, qss, nlen, len, nlm, 1, nlm)


c      c1 = 6.112*100.
c      c2 = 17.67
c      c3 = 243.5  
c      do k=1,nlm
c       do i=1,len
c         ess(i,k) = c1*exp((c2* (t2(i,k)-273.16))/((t2(i,k)-273.16)+c3))
c         if ( pint(i,k)-ess(i,k) .gt. 0. ) then
c             qss(i,k) = 0.622*ess(i,k)/ (pint(i,k)-ess(i,k))
c         else
c             qss(i,k) = 1.
c         end if 
c       end do
c      end do
c
c Compute dqs/dT
c
      do k=1,nlm
       do i=1,len
         ess(i,k) = 5417.1*qss(i,k)*pint(i,k)/
     %     ((pint(i,k) - (1-0.622)*ess(i,k))*t2(i,k)**2)
       end do
      end do

c
c  Initialize stuff:
c
      do k=1,nlm
       do i=1,len
	 dqvcup(i,k)=0.
	 dqccup(i,k)=0.
	 dqicup(i,k)=0.
	 dtcup(i,k)=0.
	 dmcup(i,k)=0.
	 mc(i,k) = 0.
       end do
      end do

      do i=1,len
        cuprec(i)=0.
	rat(i) = 1.
	cnb(i) = maxtop
	cnt(i) = nlm
	capemax(i) = 0.
	cinmin(i) = 0.
      end do

c--------------------------------------------------------------------------- 
c     main loop over horizontal index 
c--------------------------------------------------------------------------

      do i = 1, len

	nbases = 0	
c
c   Initialize column stuff:
c
        z(nlm+1)=0.

        do k = nlm,maxtop,-1

          t(k) = t2(i,k)
          qs(k) = qss(i,k) 
          q(k) = min(qss(i,k),q2(i,k)) 
          dqs(k) = ess(i,k) 

          z(k)= zint(i,k)
          h(k) = grav * z(k) + cp * t(k) + hlat * q(k)
          hs(k)= grav * z(k) + cp * t(k) + hlat * qs(k)

	  dz(k) = z(k)-z(k+1)

c	  om(k) = min(1.,max(0.,0.05*(t(k)-253.)))
          om(k) = 1.

c	  cc00=cc0*max(0.,1.-sqrt(pint(i,k)*1.e-5))
	  cc00=cc0	  


	  facb(k) = 1./(1.+cc00*dz(k))
	  facp(k) = 1. - facb(k)

          gam = fac_cond * dqs(k)
          gamm(k)=(1.+(0.61+facb(k))*t(k)*dqs(k)) / (1.+gam) / cp
          gams(k) = gam / (1.+gam) / hlat
          tmp = t(k)/gamm(k)
	  del(k) = tmp*facb(k)
          h_crit(k)= hs(k)-del(k)*qs(k)-0.61*tmp*(qs(k)-q(k))

          imass(k) = grav / pdel(i,k)
          dlogp(k) = 287.*log(pint(i,k)/pint(i,k-1))
	  alpha_factor(k) = max(1.,(pint(i,k)*2.e-5)**15)

c
c  Add temperature amd moisture perturbations within the PBL
c  based on tpert and qpert past to this routine. This is done similar
c  to the Hack's scheme to stimulate shallow convection.
c
          if( z(k).lt.pblh(i) ) then
            factor = 1.-z(k)/pblh(i)
            hprime(k) = min(tpert(i),tpertmax)*factor
            qprime(k) = min(min(qpert(i),qpertmax)*factor,
     &               max(0.,(qs(k)+dqs(k)*hprime(k))-q(k)))
            hprime(k)=cp * hprime(k) + hlat * qprime(k)
	  else
	    hprime(k) = 0.
	    qprime(k) = 0.
          end if

	end do
	
	do k=1,nlm+1
	  fluxh(k)=0.
	  fluxq(k)=0.
	  pdry(k)=0.
	end do

c
c  The lowest cloud base is in PBL at the level where moist static
c  energy has a local maximum: 
c
	pbltop=nlm
        pblbase=nlm
	do k=nlm,nlm-5,-1
	  if(z(k).le.pblh(i)) pbltop=k
	end do
c	a=0.
c	do k=nlm,pbltop,-1
c	  if(h(k).gt.a) then
c	     a = h(k)
c	     pblbase = k
c	  endif
c	end do


c
c     Determine if a column is conditionally unstable:
c
c
	do k = pblbase,maxtop+mindepth-1,-1  ! loop over cloud bases

	  cape(k) = 0.
	  cin(k) = 0.
	  
	  do l = k-1,maxtop+mindepth-1,-1

            hcc(l) = h(k)+hprime(k)-del(l)*(q(k)+qprime(k)) 
	    if(hcc(l).gt.h_crit(l)) then  ! Yes, it is unstable!
c
c  Compute lifting condensation level (assumed to be always 1 level higher
c  than the base level k:
c

	      plcl = min(pint(i,k),
     &	pint(i,k)*(1.-(qs(k)-q(k)-qprime(k))/(t(k)*dqs(k)))**3.4965)
              do m=k-1,maxtop,-1
	        if(pint(i,m).le.plcl) then
	          lcl = m
                  goto 88
	        endif	         
	      end do
 88	      continue

c
c    If LCL is higher then LFC then ignore the current cloud base:    
c

	      if(lcl.lt.l) goto 777
	       	
c	    
c     Compute the convective inhibition (CIN) and
c     the convective available potential energy (CAPE) for 
c     a given cloud base level; determine the level of the highest 
c     cloud top for the current cloud base:      
c

              buoy(k) = 0. ! buoyancy at the base 
                           !(some perturbation may be added in the future)

	      do m = k-1,lcl+1,-1  ! dry CIN
               buoy(m) = (h(k)+hprime(k)-h(m))/cp-
     &			(fac_cond-0.61*t(m))*(q(k)+qprime(k)-q(m)) 
               cin(k) =cin(k) + 0.5*(buoy(m)+buoy(m+1))*dlogp(m+1)
              end do

	      do m = lcl,l+1,-1  ! wet CIN
               buoy(m) = gamm(m)*(hcc(l)-h_crit(m))
               cin(k) =cin(k) + 0.5*(buoy(m)+buoy(m+1))*dlogp(m+1)
              end do

              buoy(l) = gamm(l)*(hcc(l)-h_crit(l))
              tmp = 0.5*(buoy(l)+buoy(l+1))*dlogp(l+1)
              cin(k) = cin(k) + min(0.,tmp)
              cape(k) = cape(k) + max(0.,tmp) 

              top = l
              qc(l) = q(k)+qprime(k)
	      do m = l-1,maxtop,-1  ! CAPE
               qc(m) = (qc(m+1) + facp(m)*
     &		  (qs(m)+gams(m)*(h(k)+hprime(k)-hs(m))))/(1.+facp(m))
               buoy(m)=gamm(m)*(h(k)+hprime(k)-del(m)*qc(m)-h_crit(m))
               if(buoy(m).lt.0.) goto 11
               cape(k) = cape(k) + 0.5*(buoy(m)+buoy(m+1))*dlogp(m+1)
               top = m
              end do
 11           cape(k) = cape(k) + cin(k)

c
c     Convection trigger is here:
c              

              if(l-top+1.ge.mindepth .and. cape(k).gt.0.) then

                nbases = nbases+1
                base(nbases) = k
	        lfc(nbases) = l
                lnb(nbases) = top
                cnb(i) = max(cnb(i), k)
                cnt(i) = min(cnt(i), top)
	        if(cape(k).gt.capemax(i)) then
		  capemax(i) = cape(k)
		  cinmin(i) = cin(k)
	        endif

c	  	if(k.eq.nlm.and.top.eq.nlm-2)
c     &     	write(6,'(4i5,5g12.4)') k,lcl,l,top,cape(k),cin(k)

              endif

              goto 777
              
	     endif ! unstable

	  end do ! l

 777	  continue

	end do ! k

c----------------------------------------------------------------------
c    loop over cloud bases
c----------------------------------------------------------------------- 

	if(nbases.eq.0) goto 1000	

        nclouds = 0

        do m = 1, nbases

            lamdaup = 0.

c
c       for a current cloud base, loop over cloud tops
c
	    do top=lnb(m),lfc(m)-mindepth+1 
c
c     Iterate for the entrainment coefficient using "shooting" algorithm
c     with the Newton-type aiming:
c
	      niter=0
	      condition = .true.
	      lamda=0.
	      hc(lfc(m))= h(base(m))+hprime(base(m))
	      qc(lfc(m))= q(base(m))+qprime(base(m))

 	      do while(condition) 

	        niter=niter+1

	        dhcloud = 0.
	        dqcloud = 0.

	        do j=lfc(m)-1,top,-1

		  a = lamda*dz(j)
                  b = 1.+ a
		  hc(j) = (hc(j+1) + a*h(j))/b
		  dhcloud = (dhcloud + dz(j)*(h(j)-hc(j)))/b
		  qc(j) = (qc(j+1) + a*q(j) + 
     &	            facp(j)*(qs(j)+gams(j)*(hc(j)-hs(j))) )/(b+facp(j))
		  dqcloud = (dqcloud + dz(j)*(q(j)-qc(j)) + 
     &	            facp(j)*gams(j)*dhcloud )/(b+facp(j))

	        end do 

		hcloud = hc(top) - del(top)*qc(top)
		dhcloud = dhcloud - del(top)*dqcloud

	        if(abs(dhcloud).lt.1.e-5) then
		  print*,'dhcloud = 0.!!!',dhcloud,lamda
	          do j=lfc(m),top,-1
		    write(6,'(2i4,4f8.2,2g13.4)') 
     &			lnb(m),j,(hc(j)-del(j)*qc(j))/cp,
     &			h_crit(j)/cp,hc(j)/cp,h(j)/cp,
     &			qc(j),q(j)
	          end do
	        endif  

	        dlamda = (h_crit(top) - hcloud)/ dhcloud
	 	lamda = lamda + dlamda
	        condition = abs(h_crit(top)-hcloud).gt.100.
     &			and.niter.le.10.and.lamda.ge.0.

c                write(6,'(3i3,4f8.1,4g11.4,f6.2)') 
c     &			niter,lfc(m),lnb(m),h(top)/cp,hcloud/cp,
c     &		        h_crit(top)/cp,hs(top)/cp,dhcloud,
c     &			dqcloud,lamda

	      end do ! while(condition)

c	      print*

c
c  Ignore cloud bases that produce too many iterations as well as
c  those that have lower entrainment coefficient than the one 
c  corresponding to the higher cloud top:

	      if(niter.gt.10.or.lamda.le.lamdaup) goto 3333

c
c  Compute cloud work function. Ignore the cloud base that produces
c  negative cloud work function at some point of vertical integration:         
c

	      do j = base(m),lfc(m),-1
	        nu(j) = 1.
	        hc(j) = h(base(m))+hprime(base(m))
	        qc(j) = q(base(m))+qprime(base(m))
	        prec(j) =0.
	        qliq(j) =0.
              end do

              l = lfc(m)
              buoy(l)= gamm(l)*(hc(l)-del(l)*qc(l)-h_crit(l))
	      cwf = 0. 

	      do j = l-1,top,-1  
	        nu(j) = (1.+lamda*dz(j))*nu(j+1)
                buoy(j) = gamm(j)*(hc(j)-del(j)*qc(j)-h_crit(j))
                cwf = cwf + 0.5*nu(j)*(buoy(j)+buoy(j+1))*dlogp(j+1)
                if(cwf.lt.0..or.cwf.gt.cape(base(m))) goto 3333
              end do

c
c   In-cloud profiles for the curent cloud type.
c   Ignore the cloud type if cloud water vanishes at some point.
c
	      do j=lfc(m),top,-1
	        qliq(j) = qc(j)-(qs(j)+gams(j)*(hc(j)-hs(j)))
	        if(qliq(j).le.0.) goto 3333
		prec(j) = facp(j)*qliq(j)
		qliq(j) = qliq(j)-prec(j)
		qice(j) = qliq(j) * (1.-om(j))
		qliq(j) = qliq(j) - qice(j)
	      end do

              lamdaup = lamda

c
c  Compute mass flux and other stuff:
c

              mb = cwf * alphi * alpha_factor(top)
	      

	      do j=base(m),top,-1
		nu(j)=nu(j)*mb
	        mc(i,j)=mc(i,j)+nu(j)
	        fluxh(j)=fluxh(j)+nu(j)*(hc(j)-h(j))
	        fluxq(j)=fluxq(j)+nu(j)*(qc(j)-q(j))
		pdry(j)=pdry(j)+prec(j)*nu(j)
              end do		

              dqccup(i,top-1) = dqccup(i,top-1)+qliq(top)*nu(top)
              dqicup(i,top-1) = dqicup(i,top-1)+qice(top)*nu(top)
              dmcup(i,top-1) = dmcup(i,top-1)+nu(top)

c	      if(base(m).eq.nlm.and.lnb(m).eq.nlm-2)
c     &	         write(6,'(3i3,4g11.3)') niter,base(m),lnb(m),
c     &		 cape(base(m)),cwf,lamda,mb

              nclouds = nclouds + 1


 3333         continue

	    end do ! top

	end do ! m

c--------------------------------------------------------------------------

        if(nclouds.eq.0) goto 1000

c        print*,nclouds,cnt(i),cnb(i) 

	sumq=0.
	sumh=0.

        do k = cnt(i)-1,cnb(i)
	  cuprec(i)=cuprec(i)+pdry(k+1)
	  dqcup = ((fluxq(k+1)-fluxq(k))-pdry(k+1))*imass(k)
	  dhcup = (fluxh(k+1)-fluxh(k))*imass(k)
	  dqccup(i,k) = dqccup(i,k) * imass(k)
	  dqicup(i,k) = dqicup(i,k) * imass(k)
	  dqvcup(i,k)=dqcup - dqccup(i,k) - dqicup(i,k)
	  dmcup(i,k) = dmcup(i,k) * imass(k)
	  dtcup(i,k)=(dhcup-dqvcup(i,k)*hlat)/cp
c
c  prevent excessive cumulus drying
c
	  drying=-dqvmax*ql(i,k)/dt
	  if(dqvcup(i,k).lt.drying) then
	     rat(i) = min(rat(i),drying/dqvcup(i,k))  
	  endif

c
c  prevent excessive cumulus mass flux
c
	  if(mc(i,k).gt.maxmc) then
	     rat(i) = min(rat(i),maxmc/mc(i,k))  
	  endif


        end do
c	  if(cuprec(i)*3.6e3 * 24..gt.10.) 
c	  if(cnb(i)-cnt(i).gt.9) 
c     &	print*,'prec=',cuprec(i)* 3.6e3 * 24.

c--------------------------------------------------------------------------

 1000 continue	

c--------------------------------------------------------------------------- 
c     end of main loop over horizontal index 
c--------------------------------------------------------------------------

      end do ! i


      do k=1,nlm
	do i=1,len

	  dtcup(i,k) = dtcup(i,k) * rat(i)
	  dqvcup(i,k) = dqvcup(i,k) * rat(i)
	  dqccup(i,k) = dqccup(i,k) * rat(i)
	  dqicup(i,k) = dqicup(i,k) * rat(i)
	  dmcup(i,k) = dmcup(i,k) * rat(i)
	  mc(i,k) = mc(i,k)*rat(i)
	  tl(i,k) = tl(i,k) + dt * dtcup(i,k)
	  ql(i,k) = max(1.e-9,ql(i,k) + dt * dqvcup(i,k))
	end do
      end do	

      do i=1,len
	  cuprec(i)=cuprec(i)*rat(i)*1.e-3
      end do

      return
      end


