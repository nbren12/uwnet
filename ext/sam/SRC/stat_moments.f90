! Module to compute to compute statistics over user specified area
! domains (i.e. goal is to compute moments for each 1 km x 1 km area for the
! projected super run).  The computation of these moments/statistics can
! easily be turned off by changing a parameter in the setparm.f90 file
! Original implementation by Peter Bogenschutz, U. Uta, March 2008

module stat_moments

use vars
use params
implicit none

integer, parameter :: nxm = max(1,(nx_gl/nsubdomains_x)/navgmom_x)
integer, parameter :: nym = max(1,(ny_gl/nsubdomains_y)/navgmom_y)

real moment1(nzm,5,nxm,nym), moment2(nzm,5,nxm,nym), moment3(nzm,9,nxm,nym)
real mom1cld(nzm,8,nxm,nym), moment5(nzm,8,nxm,nym), moment4(nzm,3,nxm,nym)

integer, external :: lenstr

CONTAINS

subroutine compmoments()

! SUBROUTINE TO COMPUTE STATISTICAL PROFILES 
! Profiles computed for given domain area (controlled in domain.f90 file)
! Output can be used to construct moments and covariances for selected 
! variables.  

integer i, j, k, ii,jj, starti, startj
real(8) thlfac, qwfac, wfac, divfac
real(8) wfac2, wfac3, wfac4
real(8) thetal(nzm,4), totalqw(nzm,4), vertw(nzm,12), qclavg(nzm,5), cldcnt(nzm)
real(8)  uavg(nzm,3), vavg(nzm,3), micro(nzm,6)
real auto, accre, rflux, evapa, evapb

call t_startf ('moments')

starti=1
startj=1

do jj=1,nym
  starti=1.
  do ii=1,nxm

    thetal(:,:)=0.
    totalqw(:,:)=0.
    vertw(:,:)=0.
    qclavg(:,:)=0.
    cldcnt(:)=0.
    uavg(:,:)=0.
    vavg(:,:)=0.
    micro(:,:)=0.

    divfac=float(navgmom_x)*float(navgmom_y)

    do k=1,nzm
      do j=startj,startj+(navgmom_y-1)
        do i=starti,starti+(navgmom_x-1)

	  ! Liquid water potential temperature
          thlfac=(tabs(i,j,k)*prespot(k))*(1-fac_cond*(qcl(i,j,k)+qci(i,j,k))/tabs(i,j,k))
 
          ! Total water mixing ratio
          qwfac=qv(i,j,k)+qcl(i,j,k)+qci(i,j,k)

	  ! Vertical Velocity
          wfac=(w(i,j,k)+w(i,j,k+1))/2.

	  call autoconversion(qcl(i,j,k),auto)
	  call accretion(qcl(i,j,k),qpl(i,j,k),accre)
	  call evaporation(qpl(i,j,k),tabs(i,j,k),pres(k),qv(i,j,k),evapa,evapb)
          call rainflux(rho(k),qpl(i,j,k),rflux)

          qclavg(k,1)=qclavg(k,1)+qcl(i,j,k)	! First mom. cloud water
          uavg(k,1)=uavg(k,1)+u(i,j,k)		! First mom. u-wind
          vavg(k,1)=vavg(k,1)+v(i,j,k)		! First mom. v-wind

          if (qcl(i,j,k)+qci(i,j,k) .gt. 0) then
            cldcnt(k)=cldcnt(k)+1.
          endif  

          thetal(k,1)=thetal(k,1)+thlfac	! First mom. theta_l
          totalqw(k,1)=totalqw(k,1)+qwfac	! First mom. total water
          vertw(k,1)=vertw(k,1)+wfac 		! First mom. vert. vel.

          thetal(k,2)=thetal(k,2)+thlfac**2.	! Sum squares theta_l
          totalqw(k,2)=totalqw(k,2)+qwfac**2.	! Sum squares total water
          vertw(k,2)=vertw(k,2)+wfac**2.	! Sum squares vert. vel.
          uavg(k,2)=uavg(k,2)+u(i,j,k)**2.	! Sum squares u-wind
          vavg(k,2)=vavg(k,2)+v(i,j,k)**2.	! Sum squares v-wind

          thetal(k,3)=thetal(k,3)+(thlfac*wfac)	! Flux theta_l and vert. vel.	
          totalqw(k,3)=totalqw(k,3)+(thlfac*qwfac) ! Flux theta_l and tot wat
          vertw(k,3)=vertw(k,3)+(wfac*qwfac)	! Flux tot wat and vert vel
          uavg(k,3)=uavg(k,3)+(wfac*u(i,j,k))	! Flux vert vel and u-wind
          vavg(k,3)=vavg(k,3)+(wfac*v(i,j,k))   ! Flux vert vel and v-wind

          qclavg(k,2)=qclavg(k,2)+(qcl(i,j,k)*wfac)	! Flux qc and vert vel
          qclavg(k,3)=qclavg(k,3)+(wfac*wfac*qcl(i,j,k)) ! Flux w^2 and qc
          qclavg(k,4)=qclavg(k,4)+(thlfac*qcl(i,j,k))	! Flux theta_l and qc
          qclavg(k,5)=qclavg(k,5)+(qwfac*qcl(i,j,k))	! flux qw and qc

          thetal(k,4)=thetal(k,4)+thlfac**3.	! Sum cubes theta_l
          totalqw(k,4)=totalqw(k,4)+qwfac**3.	! Sum cubes tot water
          vertw(k,4)=vertw(k,4)+wfac**3.	! Sum cubes vert vel

          vertw(k,5)=vertw(k,5)+wfac**4.	! Sum quad vert vel
          
          vertw(k,6)=vertw(k,6)+(wfac*wfac*thlfac) ! Flux w^2 and theta_l
          vertw(k,7)=vertw(k,7)+(wfac*wfac*qwfac) ! Flux w^2 and tot wat
          vertw(k,8)=vertw(k,8)+(wfac*thlfac*thlfac) ! Flux theta_l^2 and w
          vertw(k,9)=vertw(k,9)+(wfac*qwfac*qwfac) ! Flux qw^2 and vert vel
          vertw(k,10)=vertw(k,10)+(wfac*qwfac*thlfac) ! Flux w, theta_l, qw
          vertw(k,11)=vertw(k,11)+(wfac*u(i,j,k)*u(i,j,k)) ! Flux u^2, w
          vertw(k,12)=vertw(k,12)+(wfac*v(i,j,k)*v(i,j,k)) ! Flux v^2, w

	  micro(k,1)=micro(k,1)+qpl(i,j,k)  ! Precipitation (liquid)
	  micro(k,2)=micro(k,2)+auto
	  micro(k,3)=micro(k,3)+accre
	  micro(k,4)=micro(k,4)+rflux
	  micro(k,5)=micro(k,5)+evapa
	  micro(k,6)=micro(k,6)+evapb

!          write(*,*) 'print i and j ', i, j

        enddo
      enddo
    enddo

    ! Compute the domain area averaged statistics 
    do k=1,nzm
      moment1(k,2,ii,jj)=thetal(k,1)/(divfac)
      moment1(k,3,ii,jj)=totalqw(k,1)/(divfac)
      moment1(k,1,ii,jj)=vertw(k,1)/(divfac)
      moment1(k,4,ii,jj)=uavg(k,1)/(divfac)
      moment1(k,5,ii,jj)=vavg(k,1)/(divfac)

      moment2(k,2,ii,jj)=(thetal(k,2)/(divfac))
      moment2(k,3,ii,jj)=(totalqw(k,2)/(divfac))
      moment2(k,1,ii,jj)=(vertw(k,2)/(divfac))
      moment2(k,4,ii,jj)=(uavg(k,2)/(divfac))
      moment2(k,5,ii,jj)=(vavg(k,2)/(divfac))

      moment3(k,2,ii,jj)=(thetal(k,3)/(divfac))
      moment3(k,3,ii,jj)=(totalqw(k,3)/(divfac))
      moment3(k,1,ii,jj)=(vertw(k,3)/(divfac))
      moment3(k,4,ii,jj)=(uavg(k,3)/(divfac))
      moment3(k,5,ii,jj)=(vavg(k,3)/(divfac))
      moment3(k,6,ii,jj)=(qclavg(k,2)/(divfac))
      moment3(k,7,ii,jj)=(qclavg(k,3)/(divfac))
      moment3(k,8,ii,jj)=(qclavg(k,4)/(divfac))
      moment3(k,9,ii,jj)=(qclavg(k,5)/(divfac))

      moment4(k,2,ii,jj)=(thetal(k,4)/(divfac))
      moment4(k,3,ii,jj)=(totalqw(k,4)/(divfac))
      moment4(k,1,ii,jj)=(vertw(k,4)/(divfac))

      moment5(k,1,ii,jj)=(vertw(k,5)/(divfac))
      moment5(k,2,ii,jj)=(vertw(k,6)/(divfac))
      moment5(k,3,ii,jj)=(vertw(k,7)/(divfac))
      moment5(k,4,ii,jj)=(vertw(k,8)/(divfac))
      moment5(k,5,ii,jj)=(vertw(k,9)/(divfac))
      moment5(k,6,ii,jj)=(vertw(k,10)/(divfac))
      moment5(k,7,ii,jj)=(vertw(k,11)/(divfac))
      moment5(k,8,ii,jj)=(vertw(k,12)/(divfac))      

      mom1cld(k,1,ii,jj)=(qclavg(k,1)/(divfac))
      mom1cld(k,2,ii,jj)=(cldcnt(k)/(divfac))
      mom1cld(k,3,ii,jj)=(micro(k,1)/(divfac))
      mom1cld(k,4,ii,jj)=(micro(k,2)/(divfac))
      mom1cld(k,5,ii,jj)=(micro(k,3)/(divfac))
      mom1cld(k,6,ii,jj)=(micro(k,4)/(divfac))
      mom1cld(k,7,ii,jj)=(micro(k,5)/(divfac))
      mom1cld(k,8,ii,jj)=(micro(k,6)/(divfac))


    enddo

    starti=starti+navgmom_x

  enddo

  startj=startj+navgmom_y
enddo

  ! Call function to output the moments
  call write_moments()

call t_stopf ('moments')

return

end subroutine compmoments

!----------------------------------------------------------------------
subroutine autoconversion(qc1,auto)

real qc1, auto
real, parameter :: alpha = 0.001  ! autoconversion rate
real, parameter :: q_co =  0.001

auto=max(0.,alpha*(qc1-q_co))

return

end subroutine autoconversion

!----------------------------------------------------------------------
subroutine accretion(qc1,qr1,accre)

real qc1, qr1, accre
real, parameter :: br = 0.8

accre = qc1*qr1**((3.+br)/4.)

return

end subroutine accretion

!----------------------------------------------------------------------
subroutine evaporation(qr1,tabs1,pres1,qv1,evapa,evapb)

real qr1, tabs1, pres1, qv1, evapa, evapb
real S
real, parameter :: br = 0.8

S = qv1/qsatw(tabs1,pres1)

evapa=qr1**(1./2.)*(S-1.)
evapb=qr1**((5.+br)/8.)*(S-1.)

return

end subroutine evaporation

!----------------------------------------------------------------------
subroutine rainflux(rho1,qr1,rflux)

real rho1, qr1, rflux
real, parameter :: br = 0.8

rflux = (rho1*qr1)**((1.+br)/4.)

return

end subroutine rainflux


subroutine write_moments()

implicit none

character *120 filename
character *80 long_name
character *8 name
character *10 timechar
character *4 rankchar
character *5 sepchar
character *6 filetype
character *10 units
character *10 c_z(nzm),c_p(nzm),c_dx, c_dy, c_time
integer i,j,k,nfields,nfields1
real(4) tmp(nxm,nym,nzm)

nfields=38

if(masterproc.or.output_sep) then

  if(output_sep) then
     write(rankchar,'(i4)') rank
     sepchar="_"//rankchar(5-lenstr(rankchar):4)
  else
     sepchar=""
  end if
  write(rankchar,'(i4)') nsubdomains
  write(timechar,'(i10)') nstep
  do k=1,11-lenstr(timechar)-1
    timechar(k:k)='0'
  end do

  if(RUN3D) then
    if(savemombin) then
      filetype = '.bin3D'
    else
      filetype = '.com3D'
    end if
    filename='./OUT_MOMENTS/'//trim(case)//'_moments_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//'_'//timechar(1:10)//filetype//sepchar
    open(46,file=filename,status='unknown',form='unformatted')

  else
    if(savemombin) then
     if(savemomsep) then
       filetype = '.bin3D'
     else
       filetype = '.bin2D'
     end if
    else
     if(savemomsep) then
       filetype = '.com3D'
     else
       filetype = '.com2D'
     end if
    end if
    if(savemomsep) then
      filename='./OUT_MOMENTS/'//trim(case)//'_moments_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//'_'//timechar(1:10)//filetype//sepchar
      open(46,file=filename,status='unknown',form='unformatted')	
    else
      filename='./OUT_MOMENTS/'//trim(case)//'_moments_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//filetype//sepchar
      if(nrestart.eq.0.and.notopenedmom) then
         open(46,file=filename,status='unknown',form='unformatted')	
      else
         open(46,file=filename,status='unknown', &
                              form='unformatted', position='append')
      end if
      notopenedmom=.false.
    end if  

  end if

  if(masterproc) then

   if(savemombin) then

     write(46) nxm,nym,nzm,nsubdomains,nsubdomains_x,nsubdomains_y,nfields
     print*,nxm,nym,nzm,nsubdomains,nsubdomains_x,nsubdomains_y,nfields
     do k=1,nzm
       write(46) z(k) 
     end do
     do k=1,nzm
       write(46) pres(k)
     end do
     write(46) dx*float(navgmom_x)
     write(46) dy*float(navgmom_x)
     write(46) nstep*dt/(3600.*24.)+day0

   else

     write(long_name,'(8i4)') nxm,nym,nzm,nsubdomains, &
                                    nsubdomains_x,nsubdomains_y,nfields
     do k=1,nzm
        write(c_z(k),'(f10.3)') z(k)
     end do
     do k=1,nzm
        write(c_p(k),'(f10.3)') pres(k)
     end do
     write(c_dx,'(f10.0)') dx*float(navgmom_x)
     write(c_dy,'(f10.0)') dy*float(navgmom_y)
     write(c_time,'(f10.2)') nstep*dt/(3600.*24.)+day0
	
     write(46) long_name(1:32)
     write(46) c_time,c_dx,c_dy, (c_z(k),k=1,nzm),(c_p(k),k=1,nzm)

    end if ! savemombin

   end if ! msterproc 

end if ! masterproc.or.output_sep

nfields1=0

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment1(k,2,i,j)
    enddo
  enddo
enddo
name='THL1'
long_name='First Mom Liquid Wat Pot Tmp'
units='K'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment2(k,2,i,j)
    enddo
  enddo
enddo
name='THL2'
long_name='Second Mom Liquid Wat Pot Tmp'
units='K^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,2,i,j)
    enddo
  enddo
enddo
name='THLW'
long_name='Flux THL and W'
units='K m/s'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment4(k,2,i,j)
    enddo
  enddo
enddo
name='THL3'
long_name='Third Mom Liquid Wat Pot Tmp'
units='K^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment1(k,3,i,j)
    enddo
  enddo
enddo
name='QW1'
long_name='First Mom Total Water'
units='g/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment2(k,3,i,j)
    enddo
  enddo
enddo
name='QW2'
long_name='Second Mom Total Water'
units='g/kg ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,3,i,j)
    enddo
  enddo
enddo
name='THLQW'
long_name='Flux THL and QW'
units='g/kg K'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment4(k,3,i,j)
    enddo
  enddo
enddo
name='QW3'
long_name='Third Mom QW'
units='kg/kg ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment1(k,1,i,j)
    enddo
  enddo
enddo
name='W1'
long_name='First Mom Vert Vel'
units='m/s'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment2(k,1,i,j)
    enddo
  enddo
enddo
name='W2'
long_name='Second Mom Vert Vel'
units='m/s ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,1,i,j)
    enddo
  enddo
enddo
name='WQW'
long_name='Flux W and QW'
units='g/kg m/s'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1 
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment4(k,1,i,j)
    enddo
  enddo
enddo
name='W3'
long_name='Third Mom Vert Vel'
units='m/s ^3'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment1(k,4,i,j)
    enddo
  enddo
enddo
name='U1'
long_name='First Mom U-Wind'
units='m/s'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment2(k,4,i,j)
    enddo
  enddo
enddo
name='U2'
long_name='Second Mom U-Wind'
units='m/s ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,4,i,j)
    enddo
  enddo
enddo
name='UW'
long_name='Flux U and W'
units='m/s ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment1(k,5,i,j)
    enddo
  enddo
enddo
name='V1'
long_name='First Mom V-Wind'
units='m/s'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment2(k,5,i,j)
    enddo
  enddo
enddo
name='V2'
long_name='Second Mom V-Wind'
units='m/s ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,5,i,j)
    enddo
  enddo
enddo
name='VW'
long_name='Flux V and W'
units='m/s ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1 
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,1,i,j)
    enddo
  enddo
enddo
name='QL'
long_name='QL AVG'
units='kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains) 

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,2,i,j)
    enddo
  enddo
enddo
name='CDFRC'
long_name='Cloud Fraction'
units='-'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,6,i,j)
    enddo
  enddo
enddo
name='WQL'
long_name='Flux W and QL'
units='m/s kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,7,i,j)
    enddo
  enddo
enddo
name='W2QL'
long_name='Flux W^2 and QL'
units='m/s ^2 kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,8,i,j)
    enddo
  enddo
enddo
name='THLQL'
long_name='Flux THL and QL'
units='K kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment3(k,9,i,j)
    enddo
  enddo
enddo
name='QTQL'
long_name='Flux QT and QL'
units='kg/kg ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,1,i,j)
    enddo
  enddo
enddo
name='W4'
long_name='Fourth Moment W'
units='m/s ^4'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,2,i,j)
    enddo
  enddo
enddo
name='W2THL'
long_name='Flux W2 and THL'
units='m/s ^2 K'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,3,i,j)
    enddo
  enddo
enddo
name='W2QT'
long_name='Flux W2 and QT'
units='m/s ^2 kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,4,i,j)
    enddo
  enddo
enddo
name='WTHL2'
long_name='Flux W and THL2'
units='m/s K^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,5,i,j)
    enddo
  enddo
enddo
name='WQT2'
long_name='Flux W and QT2'
units='m/s kg/kg ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,6,i,j)
    enddo
  enddo
enddo
name='WQTTHL'
long_name='Flux W and QT and THL'
units='m/s kg/kg K'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,7,i,j)
    enddo
  enddo
enddo
name='WU2'
long_name='Flux W and U2'
units='m/s ^3'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

!write(*,*) 'here I am', nfields1

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=moment5(k,8,i,j)
    enddo
  enddo
enddo
name='WV2'
long_name='Flux W and V2'
units='m/s ^3'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,savemombin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,3,i,j)
    enddo
  enddo
enddo
name='QR1'
long_name='Rain Water Mixing Ratio'
units='kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,4,i,j)
    enddo
  enddo
enddo
name='AUTO1'
long_name='Autoconversion '
units='kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,5,i,j)
    enddo
  enddo
enddo
name='ACRE1'
long_name='Accretion '
units='kg/kg ^2'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,6,i,j)
    enddo
  enddo
enddo
name='RFLUX'
long_name='Rain Flux'
units='kg m-3'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,7,i,j)
    enddo
  enddo
enddo
name='EVAPa'
long_name='Rain Evap Prof 1'
units='kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

nfields1=nfields1+1
do k=1,nz-1
  do j=1,nym
    do i=1,nxm
      tmp(i,j,k)=mom1cld(k,8,i,j)
    enddo
  enddo
enddo
name='EVAPb'
long_name='Rain Evap Prof 2'
units='kg/kg'
call compress3D(tmp,nxm,nym,nzm,name,long_name,units,save3Dbin,dompi,rank,nsubdomains)

call task_barrier()

if (nfields .ne. nfields1) then
  if(masterproc) print*,'write_moments error: nfields'
  call task_abort()
endif
if (masterproc) then
  close(46)
endif

if(nfields.ne.nfields1) then
    if(masterproc) print*,'write_fields3D error: nfields'
    call task_abort()
end if
if(masterproc) then
    if(RUN3D.or.savemomsep) then
       if(dogzip3D) call systemf('gzip -f '//filename)
       print*, 'Writting statistical-moments data. file:'//filename
    else
       print*, 'Appending statistical-moments data. file:'//filename
    end if
  endif

end subroutine write_moments

end module stat_moments
