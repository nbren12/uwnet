     
subroutine write_fields2D
	
use vars
use params
use microphysics, only: nfields2D_micro, micro_write_fields2D
implicit none
character *120 filename
character *80 long_name
character *8 name
character *10 timechar
character *4 rankchar
character *5 sepchar
character *6 filetype
character *10 units
character *12 c_dx, c_dy, c_time
integer i,j,nfields,nfields1,nsteplast
real(4) tmp(nx,ny,nzm)
character*7 filestatus
real coef
logical, save :: notopend2D=.true.
integer, external :: lenstr

nfields= 30 + nfields2D_micro
if(.not.dolongwave) nfields = nfields-4
if(.not.doshortwave) nfields = nfields-5
if(.not.dodynamicocean) nfields=nfields-1
if(.not.((ocean_type.ne.0.or.dodynamicocean).and..not.dossthomo)) &
                 nfields=nfields-1
if(.not.doprecip) nfields = nfields-1
if(.not.docloud) nfields = nfields-3
if(SFC_FLX_FXD) nfields = nfields-2
nfields1=0

if(dopython) nfields = nfields + 2

if(masterproc.or.output_sep) then

  if(output_sep) then
     write(rankchar,'(i4)') rank
     sepchar="_"//rankchar(5-lenstr(rankchar):4)
  else
     sepchar=""
  end if
  write(rankchar,'(i4)') nsubdomains
  write(timechar,'(i10)') nstep
  do i=1,11-lenstr(timechar)-1
    timechar(i:i)='0'
  end do

! Make sure that the new run doesn't overwrite the file from the old run 

    if(.not.save2Dsep.and.notopened2D.and.(nrestart.eq.0.or.nrestart.eq.2)) then
      filestatus='new'
    else
      filestatus='unknown'
    end if

    if(save2Dbin) then
      filetype = '.2Dbin'
    else
      filetype = '.2Dcom'
    end if
    
    if(save2Dsep) then
       filename='./OUT_2D/'//trim(case)//'_'//trim(caseid)//'_'// &
          rankchar(5-lenstr(rankchar):4)//'_'//timechar(1:10)//filetype//sepchar 
          open(46,file=filename,status='unknown',form='unformatted')
    else
       filename='./OUT_2D/'//trim(case)//'_'//trim(caseid)//'_'// &
             rankchar(5-lenstr(rankchar):4)//filetype//sepchar
       open(46,file=filename,status=filestatus,form='unformatted')	
       do while(.true.)
         read(46,end=222)  nsteplast
         if(nsteplast.ge.nstep) then
           backspace(46)
           backspace(46)  ! these two lines added because of
           read(46)       ! a bug in gfrotran compiler
           print*,'2Dcom file at nstep ',nsteplast
           goto 222   ! yeh, I know, it's bad ....
         end if
         read(46)
         read(46)
         read(46)
         read(46)
         do i=1,nfields
           if(masterproc) read(46)
           read(46)
           if(.not.output_sep) then
             do j=1,nsubdomains-1
               read(46)
             end do
           end if
         end do
       end do
222    continue
       backspace(46)
       notopened2D=.false. 
    end if

    write(46) nstep
    if(save2Dbin) then

      write(46) nx,ny,nzm,nsubdomains, nsubdomains_x,nsubdomains_y,nfields
      write(46) real(dx,4)
      write(46) real(dy,4)
      write(46) real(float(nstep)*dt/(3600.*24.)+day0,4)

    else

      write(long_name,'(8i4)') nx,ny,nzm,nsubdomains,  &
                                     nsubdomains_x,nsubdomains_y,nfields
      write(c_dx,'(f12.5)') dx
      write(c_dy,'(f12.5)') dy
      write(c_time,'(f12.5)') nstep*dt/(3600.*24.)+day0
      write(46) long_name(1:32)
      write(46) c_dx
      write(46) c_dy
      write(46) c_time

    end if ! save2Dbin

end if! masterproc .or. output_sep

if(save2Davg) then
   coef = 1./float(nsave2D)
else
   coef = 1.
end if


! 2D fields:

if(doprecip) then
   nfields1=nfields1+1
   do j=1,ny
    do i=1,nx
      tmp(i,j,1)=prec_xy(i,j)*dz/dt*86400.*coef
      prec_xy(i,j) = 0.
    end do
   end do
  name='Prec'
  long_name='Surface Precip. Rate'
  units='mm/day'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
end if

if(dopython) then
   nfields1=nfields1+1
   do j=1,ny
      do i=1,nx
         tmp(i,j,1)=prec_xy(i,j)  * coef
         prec_xy(i,j) = 0.
      end do
   end do
   name='Prec'
   long_name='Surface Precip. Rate'
   units='mm/day'
   call compress3D(tmp,nx,ny,1,name,long_name,units, &
        save2Dbin,dompi,rank,nsubdomains)


   nfields1=nfields1+1
   do j=1,ny
      do i=1,nx
         tmp(i,j,1)=lhf_xy(i,j)  * coef
         lhf_xy(i,j) = 0.
      end do
   end do
   name='LHF'
   long_name='Latent Heat Flux'
   units='W/m2'
   call compress3D(tmp,nx,ny,1,name,long_name,units, &
        save2Dbin,dompi,rank,nsubdomains)
end if

if(.not.SFC_FLX_FXD) then
   nfields1=nfields1+1
   do j=1,ny
    do i=1,nx
      tmp(i,j,1)=shf_xy(i,j)*rhow(1)*cp*coef
      shf_xy(i,j) = 0.
    end do
   end do
  name='SHF'
  long_name='Sensible Heat Flux'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
   nfields1=nfields1+1
   do j=1,ny
    do i=1,nx
      tmp(i,j,1)=lhf_xy(i,j)*rhow(1)*lcond*coef
      lhf_xy(i,j) = 0.
    end do
   end do
  name='LHF'
  long_name='Latent Heat Flux'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
        
end if

if(dolongwave) then
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=lwns_xy(i,j)*coef
       lwns_xy(i,j) = 0.
     end do
   end do
  name='LWNS'
  long_name='Net LW at the surface'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=lwnsc_xy(i,j)*coef
       lwnsc_xy(i,j) = 0.
     end do
   end do
  name='LWNSC'
  long_name='Net clear-sky LW at the surface'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=lwnt_xy(i,j)*coef
       lwnt_xy(i,j) = 0.
     end do
   end do
  name='LWNT'
  long_name='Net LW at TOA'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=lwntc_xy(i,j)*coef
       lwntc_xy(i,j) = 0.
     end do
   end do
  name='LWNTC'
  long_name='Clear-Sky Net LW at TOA'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

end if

if(doshortwave) then
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=solin_xy(i,j)*coef
       solin_xy(i,j) = 0.
     end do
   end do
  name='SOLIN'
  long_name='Solar TOA insolation'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=swns_xy(i,j)*coef
       swns_xy(i,j) = 0.
     end do
   end do
  name='SWNS'
  long_name='Net SW at the surface'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=swnsc_xy(i,j)*coef
       swnsc_xy(i,j) = 0.
     end do
   end do
  name='SWNSC'
  long_name='Net Clear-sky SW at the surface'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=swnt_xy(i,j)*coef
       swnt_xy(i,j) = 0.
     end do
   end do
  name='SWNT'
  long_name='Net SW at TOA'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=swntc_xy(i,j)*coef
       swntc_xy(i,j) = 0.
     end do
   end do
  name='SWNTC'
  long_name='Net Clear-Sky SW at TOA'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

end if

if(docloud) then
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=cw_xy(i,j)*coef
       cw_xy(i,j) = 0.
     end do
   end do
  name='CWP'
  long_name='Cloud Water Path'
  units='mm'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=iw_xy(i,j)*coef
       iw_xy(i,j) = 0.
     end do
   end do
  name='IWP'
  long_name='Ice Path'
  units='mm'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=cld_xy(i,j)*coef*100.
       cld_xy(i,j) = 0.
     end do
   end do
  name='CLD'
  long_name='Cloud Frequency'
  units='%'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

end if

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=pw_xy(i,j)*coef
     pw_xy(i,j) = 0.
     end do
   end do
  name='PW'
  long_name='Precipitable Water'
  units='mm'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=usfc_xy(i,j)*coef + ug
       usfc_xy(i,j) = 0.
     end do
   end do
  name='USFC'
  long_name='U at the surface'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)


   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=u200_xy(i,j)*coef + ug
       u200_xy(i,j) = 0.
     end do
   end do
  name='U200'
  long_name='U at 200 mb'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
      tmp(i,j,1)=vsfc_xy(i,j)*coef + vg
      vsfc_xy(i,j) = 0.
     end do
   end do
  name='VSFC'
  long_name='V at the surface'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=v200_xy(i,j)*coef + vg
       v200_xy(i,j) = 0.
     end do
   end do
  name='V200'
  long_name='V at 200 mb'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=w500_xy(i,j)*coef
       w500_xy(i,j) = 0.
     end do
   end do
  name='W500'
  long_name='W at 500 mb'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

if(dodynamicocean) then
   nfields1=nfields1+1
   do j=1,ny
    do i=1,nx
      tmp(i,j,1)=qocean_xy(i,j)*coef
      qocean_xy(i,j) = 0.
    end do
   end do
  name='QOCN'
  long_name='Deep Ocean Cooling'
  units='W/m2'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
end if

if((ocean_type.ne.0.or.dodynamicocean).and..not.dossthomo) then
   nfields1=nfields1+1
   do j=1,ny
    do i=1,nx
      tmp(i,j,1)=sstxy(i,j)+t00
    end do
   end do
  name='SST'
  long_name='Sea Surface Temperature'
  units='K'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)
end if

!=====================================================
! UW ADDITIONS
 
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=psfc_xy(i,j)*coef*0.01
       psfc_xy(i,j) = 0.
     end do
   end do
  name='PSFC'
  long_name='P at the surface'
  units='mbar'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

  nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=swvp_xy(i,j)*coef
       swvp_xy(i,j) = 0.
     end do
   end do
  name='SWVP'
  long_name='Saturated Water Vapor Path'
  units='mm'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
       save2Dbin,dompi,rank,nsubdomains)

   ! 850 mbar zonal velocity
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=u850_xy(i,j)*coef
       u850_xy(i,j) = 0.
     end do
   end do
  name='U850'
  long_name='850 mbar zonal velocity'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

   ! meridional wind at 850 mbar
   nfields1=nfields1+1
   do j=1,ny
     do i=1,nx
       tmp(i,j,1)=v850_xy(i,j)*coef
       v850_xy(i,j) = 0.
     end do
   end do
  name='V850'
  long_name='850 mbar meridional velocity'
  units='m/s'
  call compress3D(tmp,nx,ny,1,name,long_name,units, &
                               save2Dbin,dompi,rank,nsubdomains)

  ! cloud top height
   nfields1=nfields1+1
   do j=1,ny
      do i=1,nx
         tmp(i,j,1)=cloudtopheight(i,j)/1000.
         cloudtopheight(i,j) = 0.
      end do
   end do
   name='ZC'
   long_name='Cloud top height (Instantaneous)'
   units='km'
   call compress3D(tmp,nx,ny,1,name,long_name,units, &
        save2Dbin,dompi,rank,nsubdomains)

   ! cloud top temperature
   nfields1=nfields1+1
   do j=1,ny
      do i=1,nx
         tmp(i,j,1)=cloudtoptemp(i,j)
         cloudtoptemp(i,j) = 0.
      end do
   end do
   name='TB'
   long_name='Cloud top temperature (Instantaneous)'
   units='K'
   call compress3D(tmp,nx,ny,1,name,long_name,units, &
        save2Dbin,dompi,rank,nsubdomains)

   ! echo top height
   nfields1=nfields1+1
   do j=1,ny
      do i=1,nx
         tmp(i,j,1)=echotopheight(i,j)/1000.
         echotopheight(i,j) = 0.
      end do
   end do
   name='ZE'
   long_name='Echo top height (Instantaneous)'
   units='km'
   call compress3D(tmp,nx,ny,1,name,long_name,units, &
        save2Dbin,dompi,rank,nsubdomains)

! END UW ADDITIONS
!=====================================================

   !bloss: Option for separate 2D output from microphysics
   !Possible uses: surface fluxes of aerosol, water isotopologues, 
   !    max column reflectivity, etc.
   if(nfields2D_micro.gt.0) call micro_write_fields2D(nfields1)

call task_barrier()


if(nfields.ne.nfields1) then
  if(masterproc) print*,'write_fields2D: error in nfields!!',nfields,nfields1
  call task_abort()
end if

if(masterproc.or.output_sep) then
     close(46)
     if(save2Dsep.and.dogzip2D) call systemf('gzip -f '//filename)
     if(masterproc)print*, 'Writting 2D data. file:'//filename
endif


end
