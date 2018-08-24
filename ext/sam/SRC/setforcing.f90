subroutine setforcing()
	
use vars
use params
implicit none
	
integer i,n,status
real tmp, tmpu, tmpv

if(masterproc) print*,'Initializeing forcing data...'

!----------------------------------------------------------------
!    Read forcings/sounding from SCAM netcdf IOP forcing file.
!         (if this has not already been done in setdata().)
!         THIS IS OPTIONAL.

if(doscamiopdata) then
  if(.NOT.isInitialized_scamiopdata) then
      !bloss: read sounding/forcing data from SCAM input file.
          call readiopdata(status)
          isInitialized_scamiopdata = .true. 
  end if
  return ! rest of routine not needed if SCAM input is used.
end if

!-------------------------------------------------
!	Read sounding file (snd):

open(77,file='./'//trim(case)//'/snd', status='old',form='formatted') 
read(77,*)
nsnd=0
do while(.true.)
  read(77,err=44,end=44,fmt=*) tmp,nzsnd
  do i=1,nzsnd
      read(77,*) tmp,tmp,tmp,tmp,tmp,tmp
  end do
  nsnd=nsnd+1
end do ! while...
44 continue
if(nsnd.eq.0.and.masterproc) then
  print*,'Error: no sounding data in file snd'
  call task_abort()
endif
rewind(77)
nsnd=nsnd+1 ! just in case ...
read(77,*)
if(masterproc) print*,'sounding data: nsnd=',nsnd,'  nzsnd=',nzsnd
allocate(usnd(nzsnd,nsnd),vsnd(nzsnd,nsnd), &
         tsnd(nzsnd,nsnd),qsnd(nzsnd,nsnd), &
         zsnd(nzsnd,nsnd),psnd(nzsnd,nsnd),daysnd(nsnd))
do n=1,nsnd-1
  read(77,*) daysnd(n)
  do i=1,nzsnd
      read(77,*) zsnd(i,n),psnd(i,n),tsnd(i,n),qsnd(i,n),usnd(i,n),vsnd(i,n)
  end do
end do
close(77)
if(nsnd.gt.2) then
  daysnd(nsnd)=daysnd(nsnd-1)+daysnd(nsnd-1)-daysnd(nsnd-2)
else
  daysnd(nsnd)=daysnd(nsnd-1)
end if
zsnd(1:nzsnd,nsnd) = zsnd(1:nzsnd,nsnd-1)
psnd(1:nzsnd,nsnd) = psnd(1:nzsnd,nsnd-1)
tsnd(1:nzsnd,nsnd) = tsnd(1:nzsnd,nsnd-1)
qsnd(1:nzsnd,nsnd) = qsnd(1:nzsnd,nsnd-1)
usnd(1:nzsnd,nsnd) = usnd(1:nzsnd,nsnd-1)
vsnd(1:nzsnd,nsnd) = vsnd(1:nzsnd,nsnd-1)
if(masterproc)print*,'Observed sounding interval (days):',daysnd(1),daysnd(nsnd)

!-------------------------------------------------
!	Read Large-scale forcing arrays:


if(dolargescale.or.dosubsidence) then

open(77,file='./'//trim(case)//'/lsf',status='old',form='formatted') 
read(77,*)
nlsf=0
do while(.true.)
  read(77,err=55,end=55,fmt=*) tmp,nzlsf
  do i=1,nzlsf
      read(77,*) tmp,tmp,tmp,tmp,tmp,tmp,tmp
  end do
  nlsf=nlsf+1
end do
55 continue
if(nlsf.eq.0.and.masterproc) then
  print*,'Error: no forcing data in file lsf.'
  call task_abort()
endif
rewind(77)
read(77,*)
nlsf=nlsf+1   ! just in case ...
if(masterproc)print*,'forcing data: nlsf=',nlsf,'  nzlsf=',nzlsf
allocate(ugls(nzlsf,nlsf),vgls(nzlsf,nlsf), wgls(nzlsf,nlsf), &
         dtls(nzlsf,nlsf),dqls(nzlsf,nlsf), &
         zls(nzlsf,nlsf),pls(nzlsf,nlsf),pres0ls(nlsf),dayls(nlsf))
do n=1,nlsf-1
  read(77,*) dayls(n),i,pres0ls(n)
  do i=1,nzlsf
      read(77,*) zls(i,n),pls(i,n),dtls(i,n),dqls(i,n),ugls(i,n),vgls(i,n),wgls(i,n)
  end do
end do
close(77)
if(nlsf.gt.2) then
   dayls(nlsf)=dayls(nlsf-1)+dayls(nlsf-1)-dayls(nlsf-2)
else
   dayls(nlsf)=dayls(nlsf-1)+1000.
end if
pres0ls(nlsf) = pres0ls(nlsf-1)
zls(1:nzlsf,nlsf) = zls(1:nzlsf,nlsf-1)
pls(1:nzlsf,nlsf) = pls(1:nzlsf,nlsf-1)
dtls(1:nzlsf,nlsf) = dtls(1:nzlsf,nlsf-1)
dqls(1:nzlsf,nlsf) = dqls(1:nzlsf,nlsf-1)
ugls(1:nzlsf,nlsf) = ugls(1:nzlsf,nlsf-1)
vgls(1:nzlsf,nlsf) = vgls(1:nzlsf,nlsf-1)
wgls(1:nzlsf,nlsf) = wgls(1:nzlsf,nlsf-1)

if(masterproc)print*,'Large-Scale Forcing interval (days):',dayls(1),dayls(nlsf)


endif

!-------------------------------------------------
!	Read Radiation forcing arrays:



if(doradforcing) then

open(77,file='./'//trim(case)//'/rad',status='old',form='formatted') 

read(77,*)
nrfc=0
do while(.true.)
  read(77,err=66,end=66,fmt=*) tmp,nzrfc
  do i=1,nzrfc
      read(77,*) tmp,tmp
  end do
  nrfc = nrfc+1
end do
66 continue
if(nrfc.eq.0.and.masterproc) then
  print*,'Error: no data found in file rad'
  call task_abort()
endif
rewind(77)
nrfc = nrfc+1  ! just in case
read(77,*)
if(masterproc)print*,'rad forcing data: nrfc=',nrfc,'  nzrfc=',nzrfc
allocate (dtrfc(nzrfc,nrfc),prfc(nzrfc,nrfc),dayrfc(nrfc))
do n=1,nrfc-1
  read(77,*) dayrfc(n)
  do i=1,nzrfc
      read(77,*) prfc(i,n),dtrfc(i,n)
  end do
end do
close(77)
if(nrfc.gt.2) then
   dayrfc(nrfc)=dayrfc(nrfc-1)+dayrfc(nrfc-1)-dayrfc(nrfc-2)
else
   dayrfc(nrfc)=dayrfc(nrfc-1)+1000.
end if
prfc(1:nzrfc,nrfc) = prfc(1:nzrfc,nrfc-1)
dtrfc(1:nzrfc,nrfc) = dtrfc(1:nzrfc,nrfc-1)
if(masterproc)print*,'Radiative Forcing interval (days):',dayrfc(1),dayrfc(nrfc)

endif ! doradforcing

!-------------------------------------------------------
! Read Surface Forcing Arrays:
!

if(dosfcforcing) then

open(77,file='./'//trim(case)//'/sfc',status='old',form='formatted') 
read(77,*)

nsfc=0
do while(.true.)
  read(77,err=77,end=77,fmt=*) tmp,tmp,tmp,tmp,tmp
  nsfc=nsfc+1
end do
77 continue
if(nsfc.eq.1.and.masterproc) then
  print*,'Error: minimum two surface forcing time samples are needed.'
  call task_abort()
endif
rewind(77)
read(77,*)
if(masterproc)print*,'surface forcing data: nsfc=',nsfc
allocate(daysfc(nsfc),sstsfc(nsfc),shsfc(nsfc),lhsfc(nsfc),tausfc(nsfc))
do n=1,nsfc
  read(77,*) daysfc(n),sstsfc(n),shsfc(n),lhsfc(n),tausfc(n)
end do
close(77)
if(masterproc)print*,'Surface Forcing interval (days):',daysfc(1),daysfc(nsfc)

else
  fluxt0 = fluxt0/rhow(1)/cp
  fluxq0 = fluxq0/rhow(1)/lcond
end if ! dosfcforcing

end

