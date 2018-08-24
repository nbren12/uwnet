! module for production of movie (visualization) files written in raw format 
! and postprocessed into gif files.
! Original implementation by Peter Bogenschutz, U. Uta, April 2008
! 

module movies

use grid

implicit none

  ! Variables required for production of movie files
  real cldttmp_xy(nx,ny), cldttmpl_xy(nx,ny), cwppath(nx,ny)
  real sfcflux(nx,ny), sfcpflog10dat(nx,ny)
  real dtmovie, amin, amax 
  integer irecc
  logical :: files_opened=.false.

! limits defaults. Can be set in namelist MOVIES
  real :: &
  u_min=-30., u_max=30., & ! surface velocity in x
  v_min=-30. , v_max=30., & ! surface velocity in x
  cldtop_min=290., cldtop_max=310., & ! cloud-top temperature
  sst_min=285., sst_max=305., & ! surface temperature
  tasfc_min=280., tasfc_max=310., & ! surface air temperature
  qvsfc_min=0.008, qvsfc_max=0.025, & ! surface vapor mixing ratio
  prec_min=0.00000001, prec_max=0.1, & ! surface precipitation (will plot log scale)
  cwp_min=0.1, cwp_max=1000., & ! cloud water path
  iwp_min=0.1, iwp_max=1000.   ! ice water path

CONTAINS

subroutine init_movies()

  integer ierr, ios, ios_missing_namelist, place_holder

  NAMELIST /MOVIES/ &
       u_min, u_max, &
       v_min, v_max, &
       cldtop_min, cldtop_max, &
       sst_min, sst_max, &
       tasfc_min, tasfc_max, &
       qvsfc_min, qvsfc_max, &
       prec_min, prec_max, &
       cwp_min, cwp_max, & 
       iwp_min, iwp_max  

  NAMELIST /BNCUIODSBJCB/ place_holder
 !----------------------------------
  !  Read namelist for movie options from prm file:
  !------------
  open(55,file='./'//trim(case)//'/prm', status='old',form='formatted')

  read (UNIT=55,NML=BNCUIODSBJCB,IOSTAT=ios_missing_namelist)
  rewind(55) !note that one must rewind before searching for new namelists

  read (55,MOVIES,IOSTAT=ios)

  if (ios.ne.0) then
     !namelist error checking
     if(ios.ne.ios_missing_namelist) then
        write(*,*) '****** ERROR: bad specification in MOVIES namelist'
        call task_abort()
     end if
  end if
  close(55)


end subroutine init_movies

subroutine openmoviefiles()

use vars
use params
implicit none

character *4 rankchar
integer, external :: lenstr, bytes_in_rec

! Subroutine to open the files needed to create the movies.

if(files_opened) return

if(masterproc) open(79,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'.info.movie',form='formatted')

write(rankchar,'(i4)') rank

open(unit=80,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_usfc.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=81,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_vsfc.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=82,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_cldtop.raw_'//rankchar(5-lenstr(rankchar):4),& 
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=83,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_tasfc.raw_'//rankchar(5-lenstr(rankchar):4),&
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=84,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_qvsfc.raw_'//rankchar(5-lenstr(rankchar):4),&
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=85,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_sfcprec.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=86,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_cwp.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=87,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_iwp.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

open(unit=88,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'_sst.raw_'//rankchar(5-lenstr(rankchar):4), &
                                              form='unformatted',access='direct',recl=nx*ny/bytes_in_rec())

files_opened = .true.
return

end subroutine openmoviefiles

!----------------------------------------------------------------------

subroutine mvmovies()

use vars
use params
implicit none

call t_startf ('movies')
! Subroutine to call the subroutines required to make the movies.  
! Modify amin and max as you see fit.  

call openmoviefiles()

call cldtoptmp()

!  Surface U-Wind Component
amin=u_max
amax=u_min
call plotpix(u(1:nx,1:ny,1),nx,ny,nstep,irecc,amin,amax,80)

!  Surface V-Wind Component
amin=v_max
amax=v_min
call plotpix(v(1:nx,1:ny,1),nx,ny,nstep,irecc,amin,amax,81)

!  Cloud Top Temperature
amin=cldtop_max
amax=cldtop_min
call plotpix(cldttmp_xy(:,:),nx,ny,nstep,irecc,amin,amax,82)


!  Surface temperature
amin=sst_max
amax=sst_min
call plotpix(sstxy(1:nx,1:ny),nx,ny,nstep,irecc,amin,amax,88)

!  Surface air temperature
amin=tasfc_max
amax=tasfc_min
call plotpix(tabs(:,:,1),nx,ny,nstep,irecc,amin,amax,83)

!  Water vapor mixing ratio
amin=qvsfc_max
amax=qvsfc_min
call plotpix(qv(:,:,1),nx,ny,nstep,irecc,amin,amax,84)

!  Surface precip rate
call getvtrr(nx,ny,nz_gl,qpl(:,:,1),tabs(:,:,1),rho,sfcflux)
amin=alog10(prec_max)
amax=alog10(prec_min)
call convertlog10(sfcflux,nx,ny,amax,sfcpflog10dat)
call plotpix(sfcpflog10dat,nx,ny,nstep,irecc,amin,amax,85)

!  Cloud Water Path
amin=alog10(cwp_max)
amax=alog10(cwp_min)
sfcpflog10dat(:,:)=0.
call pathvar(nx,ny,nz_gl,z,qcl(:,:,:),cwppath)
call convertlog10(cwppath,nx,ny,amax,sfcpflog10dat)
call plotpix(sfcpflog10dat,nx,ny,nstep,irecc,amin,amax,86)

!  Ice Water Path
amin=alog10(iwp_max)
amax=alog10(iwp_min)
sfcpflog10dat(:,:)=0.
cwppath(:,:)=0.
call pathvar(nx,ny,nz_gl,z,qci(:,:,:)+qpi(:,:,:),cwppath)
call convertlog10(cwppath,nx,ny,amax,sfcpflog10dat)
call plotpix(sfcpflog10dat,nx,ny,nstep,irecc,amin,amax,87)
   
irecc = irecc+1
call close_files()

if(masterproc) print*, 'appending OUT_MOVIES/*.raw files. nstep=', nstep

call t_stopf ('movies')
return
  
end subroutine mvmovies

!---------------------------------------------------------------------

subroutine close_files()

! Close the movie files after last timestep of integration


!  close(unit=80)
!  close(unit=81)
!  close(unit=82)
!  close(unit=83)
!  close(unit=84)
!  close(unit=85)
!  close(unit=86)
!  close(unit=87)
!  close(unit=88)

  if(masterproc) then
!   open(79,file='./OUT_MOVIES/'//trim(case)//'_'//trim(caseid)//'.info.movie',form='formatted')
   rewind(79)
   write(79,*) nsubdomains_x,nsubdomains_y
   write(79,*) nx_gl/nsubdomains_x,ny_gl/nsubdomains_y
   write(79,*) irecc-1
   write(79,*) 'field: min, max values:'
   write(79,*) 'u:',u_min, u_max
   write(79,*) 'v:', v_min, v_max
   write(79,*) 'cldtop:', cldtop_min, cldtop_max
   write(79,*) 'sst:', sst_min, sst_max
   write(79,*) 'tasfc:', tasfc_min, tasfc_max
   write(79,*) 'qvsfc:', qvsfc_min, qvsfc_max
   write(79,*) 'log10(prec):', alog10(prec_min), alog10(prec_max)
   write(79,*) 'log10(cwp):', alog10(cwp_min), alog10(cwp_max)
   write(79,*) 'log10(iwp):', alog10(iwp_min), alog10(iwp_max)
!   close(79)
  end if

  return

end subroutine close_files


!---------------------------------------------------------------------

subroutine pixplt_raw(data,nx,ny,amin,amax,iun,irec)

  implicit none

  real amin, amax
  integer nx, ny, i, j, idata, iun, ncolor, irec, count
  parameter(ncolor = 254)
  character*1 iclrs(nx,ny)
  real data(nx,ny), rdata
  
  count=0
  do j=1,ny
    do i=1,nx
      rdata=data(i,j)
      idata=int((rdata-amin)/(amax-amin)*float(ncolor))+1
      idata=max(0,min(idata,ncolor+1))
      iclrs(i,j)=char(idata)
      if (idata .ge. 254) then
        count=count+1
      endif
!      write(*,*) idata, data(i,j), iclrs(i,j)
    enddo
  enddo

  write(iun,rec=irec) iclrs(:,:)

  return

end subroutine pixplt_raw



!----------------------------------------------------------------------

subroutine plotpix(data,nx,ny,n,irec,amin,amax,unitnum)

  implicit none

  integer nx, ny, n, unitnum, irec
  real data(nx,ny), amin, amax

    call pixplt_raw(data,nx,ny,amin,amax,unitnum,irec)

  return

end subroutine plotpix

!---------------------------------------------------------------------

subroutine convertlog10(data,nx,nz,minval,retdat)

  implicit none

  integer i, j, nx, nz
  real minval, data(nx,nz), retdat(nx,nz)

  do j=1,nz
    do i=1,nx
      if (data(i,j) .eq. 0 .or. alog10(data(i,j)) .lt. minval) then
        retdat(i,j)=minval
      else
        retdat(i,j)=alog10(data(i,j))
      endif
    enddo
  enddo

  return

end subroutine convertlog10

!----------------------------------------------------------------------

subroutine getvtrr(nx,ny,nz,qr,tabs,rhopro,flux)

  implicit none
  
  integer i, j, nx, ny, nz
  real qrho, act2, vconr, rho, vtrr, rhofac
  real qr(nx,ny), tabs(nx,ny), p(nx,ny), flux(nx,ny), rhopro(nz)

  real, parameter :: pie=3.141593
  real, parameter :: rnzr=8.e6
  real, parameter :: alin=841.99667
  real, parameter :: gam480=17.8379
  real, parameter :: rhor=1.e3

  act2=pie*rnzr*rhor
  vconr=alin*gam480/(6.*act2**(0.2))

  do j=1,ny
    do i=1,nx
      rho=rhopro(1)
      qrho=rho*qr(i,j)
      rhofac=sqrt(1.226/rho)
      vtrr=amin1(vconr*qrho**(0.2)*rhofac,10.0)
      flux(i,j)=rho*vtrr*qr(i,j)
    enddo
  enddo

  return

end subroutine getvtrr


!----------------------------------------------------------------------

subroutine pathvar(nx,ny,nz,z,var,path)

  implicit none

  integer i, j, k, nx, ny, nz
  real var(nx,ny,nz), path(nx,ny), z(nz+1), rho

  rho=1000.
  path(:,:)=0.
  do k=2,nz
    do j=1,ny
      do i=1,nx
        path(i,j)=path(i,j)+(rho*var(i,j,k))*(z(k)-z(k-1))
      enddo
    enddo
  enddo

  return

end subroutine pathvar


!----------------------------------------------------------------------

subroutine cldtoptmp()

use vars
use params
implicit none

integer chkind, chkind2, tcount, i, j, k
real coef1, downint

cldttmp_xy(:,:)=0.
cldttmpl_xy(:,:)=0.
tcount=0
do j=1,ny
  do i=1,nx
    
    downint=0.
    chkind=1.
    chkind2=1.
    do k=nzm,1,-1  ! Integrate downward
      coef1=rho(k)*dz*adz(k)*dtfactor
      downint=downint+(qcl(i,j,k)+qci(i,j,k))*coef1
      if (downint .ge. 0.1 .and. chkind .eq. 1) then 
        cldttmp_xy(i,j)=tabs(i,j,k)
        chkind=2.
      endif      
    end do

    do k=1,nzm-1
      if (qcl(i,j,k)+qci(i,j,k) .gt. 0 .and. qcl(i,j,k+1)+qci(i,j,k+1) .eq. 0 .and. chkind2 .eq. 1) then
        cldttmpl_xy(i,j)=tabs(i,j,k)
!        write(*,*) z
        chkind2=2.
      endif
    enddo

    if (chkind2 .eq. 1) then 
      cldttmpl_xy(i,j)=sstxy(i,j)
    endif

    if (chkind .eq. 1) then 
      cldttmp_xy(i,j)=sstxy(i,j)
      tcount=tcount+1
    endif
   
  end do
end do

return 

end subroutine cldtoptmp

end module movies
