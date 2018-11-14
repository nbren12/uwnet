     
subroutine write_fields3D
	
use vars
use rad, only: qrad
use params
use microphysics, only: nmicro_fields, micro_field, flag_number, &
     flag_micro3Dout, mkname, mklongname, mkunits, mkoutputscale, &
     index_water_vapor, GET_reffc, Get_reffi, &
     nfields3D_micro, micro_write_fields3D, micro_diagnose
use python_caller, only: FQTNN, FSLINN, funn, fvnn
implicit none
character *120 filename
character *80 long_name
character *8 name
character *10 timechar
character *4 rankchar
character *5 sepchar
character *6 filetype
character *10 units
character *12 c_z(nzm),c_p(nzm),c_dx, c_dy, c_time
integer i,j,k,n,nfields,nfields1
real(4) tmp(nx,ny,nzm)
integer, external :: lenstr

nfields=10 + nfields3D_micro ! number of 3D fields to save
if(.not. (docloud .or. dopython))  nfields=nfields-1
if(.not.(doprecip .or. dopython)) nfields=nfields-1

! add 3D outputs for the neural network parametrized sources
if (dopython) nfields = nfields + 4

!bloss: add 3D outputs for microphysical fields specified by flag_micro3Dout
!       except for water vapor (already output as a SAM default).
if(docloud) nfields=nfields+SUM(flag_micro3Dout)-flag_micro3Dout(index_water_vapor)
if((dolongwave.or.doshortwave).and..not.doradhomo) nfields=nfields+1
if(compute_reffc.and.(dolongwave.or.doshortwave).and.rad3Dout) nfields=nfields+1
if(compute_reffi.and.(dolongwave.or.doshortwave).and.rad3Dout) nfields=nfields+1

nfields1=0


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
    if(save3Dbin) then
      filetype = '.bin3D'
    else
      filetype = '.com3D'
    end if
    filename='./OUT_3D/'//trim(case)//'_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//'_'//timechar(1:10)//filetype//sepchar
    open(46,file=filename,status='unknown',form='unformatted')

  else
    if(save3Dbin) then
     if(save3Dsep) then
       filetype = '.bin3D'
     else
       filetype = '.bin2D'
     end if
    else
     if(save3Dsep) then
       filetype = '.com3D'
     else
       filetype = '.com2D'
     end if
    end if
    if(save3Dsep) then
      filename='./OUT_3D/'//trim(case)//'_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//'_'//timechar(1:10)//filetype//sepchar
      open(46,file=filename,status='unknown',form='unformatted')	
    else
      filename='./OUT_3D/'//trim(case)//'_'//trim(caseid)//'_'// &
        rankchar(5-lenstr(rankchar):4)//filetype//sepchar
      if(nrestart.eq.0.and.notopened3D) then
         open(46,file=filename,status='unknown',form='unformatted')	
      else
         open(46,file=filename,status='unknown', &
                              form='unformatted', position='append')
      end if
      notopened3D=.false.
    end if  

  end if

  if(masterproc) then

    if(save3Dbin) then

      write(46) nx,ny,nzm,nsubdomains,nsubdomains_x,nsubdomains_y,nfields
      do k=1,nzm
        write(46) REAL(z(k),KIND=4)
      end do
      do k=1,nzm
        write(46) REAL(pres(k),KIND=4)
      end do
      write(46) REAL(dx,KIND=4)
      write(46) REAL(dy,KIND=4)
      write(46) REAL(nstep*dt/(3600.*24.)+day0,KIND=4)

    else

      write(long_name,'(8i4)') nx,ny,nzm,nsubdomains, &
                                   nsubdomains_x,nsubdomains_y,nfields
      do k=1,nzm
         write(c_z(k),'(f12.3)') z(k)
      end do
      do k=1,nzm
         write(c_p(k),'(f12.3)') pres(k)
      end do
      write(c_dx,'(f12.5)') dx
      write(c_dy,'(f12.5)') dy
      write(c_time,'(f12.5)') nstep*dt/(3600.*24.)+day0
	
      write(46) long_name(1:32)
      write(46) c_time,c_dx,c_dy, (c_z(k),k=1,nzm),(c_p(k),k=1,nzm)

    end if ! save3Dbin

  end if ! masterproc
 
end if ! masterproc.or.output_sep

  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=u(i,j,k) + ug
    end do
   end do
  end do
  name='U'
  long_name='X Wind Component'
  units='m/s'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=v(i,j,k) + vg
    end do
   end do
  end do
  name='V'
  long_name='Y Wind Component'
  units='m/s'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=w(i,j,k)
    end do
   end do
  end do
  name='W'
  long_name='Z Wind Component'
  units='m/s'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=p(i,j,k)
    end do
   end do
  end do
  name='PP'
  long_name='Pressure Perturbation'
  units='Pa'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)


if((dolongwave.or.doshortwave).and..not.doradhomo) then
  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=qrad(i,j,k)*86400.
    end do
   end do
  end do
  name='QRAD'
  long_name='Radiative heating rate'
  units='K/day'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)
end if
if(compute_reffc.and.(dolongwave.or.doshortwave).and.rad3Dout) then
  nfields1=nfields1+1
  tmp(1:nx,1:ny,1:nzm)=Get_reffc()
  name='REL'
  long_name='Effective Radius for Cloud Liquid Water'
  units='mkm'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)
end if
if(compute_reffi.and.(dolongwave.or.doshortwave).and.rad3Dout) then
  nfields1=nfields1+1
  tmp(1:nx,1:ny,1:nzm)=Get_reffi()
  name='REI'
  long_name='Effective Radius for Cloud Ice'
  units='mkm'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)
end if


  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=tabs(i,j,k)
    end do
   end do
  end do
  name='TABS'
  long_name='Absolute Temperature'
  units='K'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
  do k=1,nzm
     do j=1,ny
        do i=1,nx
           tmp(i,j,k)=t(i,j,k)
        end do
     end do
  end do
  name='SLI'
  long_name='Liquid/Ice Static Energy'
  units='K'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
       save3Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
  do k=1,nzm
     do j=1,ny
        do i=1,nx
           tmp(i,j,k)=micro_field(i,j,k,1)*1.e3
        end do
     end do
  end do
  name='QT'
  long_name='Total non-precipitating water'
  units='g/kg'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
       save3Dbin,dompi,rank,nsubdomains)

  if (.not. docloud) call micro_diagnose()
  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=qv(i,j,k)*1.e3
    end do
   end do
  end do
  name='QV'
  long_name='Water Vapor'
  units='g/kg'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)

if(docloud .or. dopython) then
  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=(qcl(i,j,k)+qci(i,j,k))*1.e3
       ! tmp(i,j,k)=qn(i,j,k) * 1.e3
    end do
   end do
  end do
  name='QN'
  long_name='Non-precipitating Condensate (Water+Ice)'
  units='g/kg'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)
end if


if(doprecip .or. dopython) then
  nfields1=nfields1+1
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      tmp(i,j,k)=(qpl(i,j,k)+qpi(i,j,k))*1.e3
    end do
   end do
  end do
  name='QP'
  long_name='Precipitating Water (Rain+Snow)'
  units='g/kg'
  call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
                                 save3Dbin,dompi,rank,nsubdomains)
end if


do n = 1,nmicro_fields
   if(docloud.AND.flag_micro3Dout(n).gt.0.AND.n.ne.index_water_vapor) then
      nfields1=nfields1+1
      do k=1,nzm
         do j=1,ny
            do i=1,nx
               tmp(i,j,k)=micro_field(i,j,k,n)*mkoutputscale(n)
            end do
         end do
         ! remove factor of rho from number, if this field is a number concentration
         if(flag_number(n).gt.0) tmp(:,:,k) = tmp(:,:,k)*rho(k)
      end do
      name=TRIM(mkname(n))
      long_name=TRIM(mklongname(n))
      units=TRIM(mkunits(n))
      call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
           save3Dbin,dompi,rank,nsubdomains)
   end if
end do


! Output Neural network source terms
if (dopython) then
   nfields1=nfields1+1
   name='FQTNN'
   long_name='Parametrized source of QT from the neural network'
   units='g/kg/day'
   tmp(1:nx,1:ny,1:nzm) = fqtnn(1:nx,1:ny,1:nzm)
   call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
        save3Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   name='FSLINN'
   long_name='Parametrized source of SLI from the neural network'
   units='K/day'
   tmp(1:nx,1:ny,1:nzm) = fslinn(1:nx,1:ny,1:nzm)
   call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
        save3Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   name='FUNN'
   long_name='Parametrized source of U from the neural network'
   units='m/s/day'
   tmp(1:nx,1:ny,1:nzm) = funn(1:nx,1:ny,1:nzm)
   call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
        save3Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   name='FVNN'
   long_name='Parametrized source of V from the neural network'
   units='m/s/day'
   tmp(1:nx,1:ny,1:nzm) = fvnn(1:nx,1:ny,1:nzm)
   call compress3D(tmp,nx,ny,nzm,name,long_name,units, &
        save3Dbin,dompi,rank,nsubdomains)
end if

!=====================================================
! UW ADDITIONS
 
if(nfields3D_micro.gt.0) then
  call micro_write_fields3D(nfields1)
end if

! END UW ADDITIONS
  call task_barrier()

  if(nfields.ne.nfields1) then
    if(masterproc) print*,'write_fields3D error: nfields=',nfields,'  nfields1=',nfields1
    call task_abort()
  end if
  if(masterproc) then
    close (46)
    if(RUN3D.or.save3Dsep) then
       if(dogzip3D) call systemf('gzip -f '//filename)
       print*, 'Writting 3D data. file:'//filename
    else
       print*, 'Appending 3D data. file:'//filename
    end if
  endif

 
end
