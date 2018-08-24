subroutine setdata()
	
use vars
use params
!use micro_params
use simple_ocean, only: set_sst
use microphysics, only: micro_init, micro_proc
use sgs, only: sgs_init, sgs_proc
implicit none
	
integer ndmax,n,i,j,k,kb,iz,it,jt
real presr(nz), qc0(nzm),qi0(nzm)	
parameter (ndmax = 1000)
real zz(ndmax),tt(ndmax),qq(ndmax),uu(ndmax),vv(ndmax) 
real zz1(ndmax),tt1(ndmax),qq1(ndmax),uu1(ndmax),vv1(ndmax) 
real rrr1,rrr2, pres1, pp(ndmax),ta(ndmax)
real pp1(ndmax)
real ratio_t1,ratio_t2,ratio_p1,ratio_p2
real tpert0(ndmax), qpert0(ndmax)
real latit,long
logical zgrid
integer status
real  coef
!-------------------------------------------------------------
!	read subensemble perturbation file first:

if(doensemble) then
	
open(76,file=trim(rundatadir)//'/tqpert',status='old',form='formatted')  
read(76,*)
  do j=0,nensemble
    read(76,*) i,n
    do i=1,n
      read(76,end=766,fmt=*) pp(i),tpert0(i),qpert0(i)
      tpert0(i)=tpert0(i)*(1000./pp(i))**(rgas/cp)
    end do
  end do
  close(76)
  if(masterproc) then
    print*,'Subensemble run. nensemble=',nensemble
    print*,'tpert:',(tpert0(i),i=1,n)
    print*,'qpert:',(qpert0(i),i=1,n)
  end if
  goto 767
766  print*,'Error: nensemble is too large.'  
  call task_abort()
767  continue
else
  do i=1,ndmax
    tpert0(i)=0.
    qpert0(i)=0.
  end do
end if



!**************************************************************
!	Read Initial Sounding

if(doscamiopdata) then

   !bloss: doensemble not implemented in conjunction with doscamiopdata yet
   if(doensemble) then
      if(masterproc)print *,'doensemble does not work with doscamiopdata yet'
      call task_abort()
   end if

   !bloss: doradforcing not implemented in conjunction with doscamiopdata yet
   if(doradforcing) then
      if(masterproc)print *,'doradforcing does not work with doscamiopdata yet'
      call task_abort()
   end if

   !bloss: read sounding/forcing data from SCAM input file.
   call readiopdata(status)

   isInitialized_scamiopdata = .true.

   !bloss: Interpolate sounding data to initial time.
   !       It has already been interpolated onto the model's z grid
   !         within readiopdata.
   do i = 1,nsnd-1
      if((day.ge.daysnd(i)).and.(day.lt.daysnd(i+1))) then
         coef = (day-daysnd(i)) / (daysnd(i+1)-daysnd(i))
         pres0 = (1-coef)*pres0ls(i) + coef*pres0ls(i+1) !surface pressure [mb]
         do k = 1,nzsnd
            pp(k) = (1-coef)*psnd(k,i) + coef*psnd(k,i+1) !absolute temp [K]
            tt(k) = (1-coef)*tsnd(k,i) + coef*tsnd(k,i+1) !absolute temp [K]
            qq(k) = (1-coef)*qsnd(k,i) + coef*qsnd(k,i+1) !tot water [g/kg]
            uu(k) = (1-coef)*usnd(k,i) + coef*usnd(k,i+1) !u wind [m/s]
            vv(k) = (1-coef)*vsnd(k,i) + coef*vsnd(k,i+1) !v wind [m/s]
            ta(k)=  tt(k)*(pp(k)/1000.)**(rgas/cp)
         end do
         exit !break out of do i=1,nsnd-1
      elseif(i.eq.nsnd-1) then
         if(masterproc) print*,'Error: day is beyond the sounding time range'
         call task_abort()
      end if
   end do

   zgrid = .false. ! SCAM input based on pressure.

   n = nzsnd

else

open(77,file='./'//trim(case)//'/snd',status='old',form='formatted')
read(77,*)

do while(.true.)

  read(77,err=55,end=55,fmt=*) rrr1, n, pres0
  do i=1,n
      read(77,*) zz(i),pp(i),tt(i),qq(i),uu(i),vv(i)
  end do	      
  read(77,err=55,end=55,fmt=*) rrr2, n, pres1
  do i=1,n
      read(77,*) zz1(i),pp1(i),tt1(i),qq1(i),uu1(i),vv1(i)
  end do	      

  if(day.ge.rrr1.and.day.le.rrr2) then
    if(zz(2).gt.zz(1)) then
      zgrid = .true.
      do i=1,n
      zz(i)=zz(i)+(zz1(i)-zz(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      tt(i)=tt(i)+(tt1(i)-tt(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      qq(i)=qq(i)+(qq1(i)-qq(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      uu(i)=uu(i)+(uu1(i)-uu(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      vv(i)=vv(i)+(vv1(i)-vv(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      tt(i)=tt(i)+tpert0(i)
      qq(i)=qq(i)+qpert0(i) 
      end do
    else if(pp(2).lt.pp(1)) then
      zgrid = .false.
      do i=1,n
      pp(i)=pp(i)+(pp1(i)-pp(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      tt(i)=tt(i)+(tt1(i)-tt(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      qq(i)=qq(i)+(qq1(i)-qq(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      uu(i)=uu(i)+(uu1(i)-uu(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      vv(i)=vv(i)+(vv1(i)-vv(i))/(rrr2-rrr1+1.e-5)*(day-rrr1)
      tt(i)=tt(i)+tpert0(i)
      qq(i)=qq(i)+qpert0(i) 
      ta(i)=tt(i)*(pp(i)/1000.)**(rgas/cp)
      end do
    else  
      if(masterproc) print*,'vertical grid is undefined...'
    end if
    pres0=pres0+(pres1-pres0)/(rrr2-rrr1+1.e-5)*(day-rrr1)      
    goto 56
  endif
  do i=1,n+1
    backspace(77)
!    backspace(77) ! these two lines were addedf because of 
!    read(77)      ! a bug in gfortran compiler
  end do	      

end do

55 continue
if(masterproc) then
  print*,'Error: day is beyond the sounding time range'
  print*,day,rrr1,rrr2
end if
call task_abort()
56 continue	

close (77)

end if ! if(doscamiopdata)

if(masterproc) then
  print *	
  print *,'surface pressure: ',pres0
endif  

! compute heights from pressure:

if(.not.zgrid) then
 zz(1) = rgas/ggr*ta(1)*log(pres0/pp(1))
 do i=2,n
  zz(i)=zz(i-1)+0.5*rgas/ggr*(ta(i)+ta(i-1))*log(pp(i-1)/pp(i))
 end do
end if  	
!-----------------------------------------------------------
!       Interpolate sounding into vertical grid:

presr(1)=(pres0/1000.)**(rgas/cp)
presi(1)=pres0
do k= 1,nzm
 do iz = 2,n
  if(z(k).le.zz(iz)) then
    t0(k)=tt(iz-1)+(tt(iz)-tt(iz-1))/(zz(iz)-zz(iz-1))*(z(k)-zz(iz-1))	
    q0(k)=qq(iz-1)+(qq(iz)-qq(iz-1))/(zz(iz)-zz(iz-1))*(z(k)-zz(iz-1))
    u0(k)=uu(iz-1)+(uu(iz)-uu(iz-1))/(zz(iz)-zz(iz-1))*(z(k)-zz(iz-1))  
    v0(k)=vv(iz-1)+(vv(iz)-vv(iz-1))/(zz(iz)-zz(iz-1))*(z(k)-zz(iz-1)) 
    goto 12
  endif
 end do

!  Utilize 1976 standard atmosphere for points above sounding:


 call atmosphere(z(k-1)/1000.,ratio_p1,rrr1,ratio_t1)
 call atmosphere(z(k)/1000.,ratio_p2,rrr1,ratio_t2)

 tabs0(k)=ratio_t2/ratio_t1*tabs0(k-1)
 presi(k+1)=presi(k)*exp(-ggr/rgas/tabs0(k)*(zi(k+1)-zi(k)))
 pres(k) = 0.5*(presi(k)+presi(k+1))
 prespot(k)=(1000./pres(k))**(rgas/cp)
! q0(k)=max(0.,2.*q0(k-1)-q0(k-2))
  q0(k) = q0(k-1)*exp(-(z(k)-z(k-1))/3000.) ! always decrease q0 with height
 u0(k)=u0(k-1)
 v0(k)=v0(k-1)
 goto 13
12 continue
 q0(k)=q0(k)*1.e-3
 tv0(k)=t0(k)*(1.+epsv*q0(k))
 presr(k+1)=presr(k)-ggr/cp/tv0(k)*(zi(k+1)-zi(k))
 presi(k+1)=1000.*presr(k+1)**(cp/rgas)
 pres(k) = exp(log(presi(k))+log(presi(k+1)/presi(k))* &
                             (z(k)-zi(k))/(zi(k+1)-zi(k)))
 prespot(k)=(1000./pres(k))**(rgas/cp)
 tabs0(k)=t0(k)/prespot(k)
13 continue
 ug0(k)=u0(k)
 vg0(k)=v0(k)
end do


! recompute pressure levels (for consistancy):

!	call pressz()
        
!-------------------------------------------------------------
!       Initial thernodynamic profiles: 
	
do k=1,nzm

  gamaz(k)=ggr/cp*z(k)
  t0(k) = tabs0(k)+gamaz(k) 
  qv0(k) = q0(k)
  qc0(k) = 0.
  qi0(k) = 0.
  qn0(k) = 0.
  qp0(k) = 0.
  p0(k) = 0.

  rho(k) = (presi(k)-presi(k+1))/(zi(k+1)-zi(k))/ggr*100.
  bet(k) = ggr/tabs0(k)
 
  u0(k) = u0(k) - ug
  v0(k) = v0(k) - vg
  ug0(k) = ug0(k) - ug
  vg0(k) = vg0(k) - vg

end do

do k=2,nzm
  rhow(k) =  (pres(k-1)-pres(k))/(z(k)-z(k-1))/ggr*100.
end do
rhow(1) = 2*rhow(2) - rhow(3)
rhow(nz)= 2*rhow(nzm) - rhow(nzm-1)


do k=1,nzm
 do j=1,ny
  do i=1,nx
   u(i,j,k)= u0(k)
   v(i,j,k)= v0(k)
   w(i,j,k)= 0.
   t(i,j,k)= t0(k)
   tabs(i,j,k) = tabs0(k)
   qcl(i,j,k)=0.
   qci(i,j,k)=0.
   qpl(i,j,k)=0.
   qpi(i,j,k)=0.
   p(i,j,k)=0.
   w(i,j,nz)=0.
   fluxbu(i,j)=0.
   fluxbv(i,j)=0.
   fluxbt(i,j)=0.
   fluxbq(i,j)=0.
   fluxtu(i,j)=0.
   fluxtv(i,j)=0.
   fluxtt(i,j)=0.
   fluxtq(i,j)=0.
   precsfc(i,j)=0.
   sstxy(i,j)=0.
  end do 
 end do 
end do 

dudt = 0.
dvdt = 0.
dwdt = 0.	   
	
if(docloud.or.dosmoke) call micro_init()  !initialize microphysics

do k=1,nzm
  qc0(k) = sum(dble(qcl(1:nx,1:ny,k)))/float(nx*ny)
  qi0(k) = sum(dble(qci(1:nx,1:ny,k)))/float(nx*ny)
  qn0(k) = qc0(k) + qi0(k)
  qv0(k) = q0(k) - qn0(k)
  tabs0(k) = sum(dble(tabs(1:nx,1:ny,k)))/float(nx*ny)
end do


if(masterproc) then
 print *,'Initial Sounding:'
 print *, ' k      z    rho     rhoi    s      h     h*      qt      u      v     adz     Nsq'
 do k=nzm,1,-1
   kb=max(1,k-1)
   write(6,'(i4,1x,f7.1,2f7.3,f7.2,f7.2,4f7.2,5g11.4)') k,z(k),rho(k),rhow(k),tabs0(k)+gamaz(k), &
          t0(k)+lcond/cp*qv0(k), t0(k)+lcond/cp*qsatw(tabs0(k),pres(k)), &
		q0(k)*1.e3,u0(k)+ug,v0(k)+vg, adz(k),bet(k)*(t0(k)-t0(kb))/(adzw(k)*dz)
 end do  
 print *, ' k      z    rho     rhoi    s      h     h*      qt      u      v     adz     Nsq'

 print *  
 print *,'  k      z      dz     pres   presi   Tabs     tp     tpl      qt      Qc      Qi      REL' 
 coef=1.
 if(dosmoke) coef = 0.
 do k = nzm,1,-1
  write(6,'(i4,1x,7f8.2,3f8.4,f8.2)')   k,z(k),zi(k+1)-zi(k),pres(k),presi(k),tabs0(k), &
     tabs0(k)*prespot(k),tabs0(k)*prespot(k)-lcond/cp*qc0(k),q0(k)*1.e3, qc0(k)*1.e3,qi0(k)*1.e3, &
     coef*100.*qv0(k)/qsatw(tabs0(k),pres(k)) 
 end do
 print *,'  k      z      dz     pres   presi   Tabs     tp     tpl      qt      Qc      Qi      REL' 
endif

if(dosgs) call sgs_init()

call setperturb()

if(.not.dosfcforcing) call set_sst()

call task_barrier()

call boundaries(4)


end

