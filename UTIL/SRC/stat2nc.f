c-------------------------------------------------------
	implicit none
	include 'netcdf.inc'
	include 'hbuf.inc'	
	
        integer, parameter :: nparms = 68  
	integer flag(HBUF_MAX_LENGTH)
c-------------------------------------------------------	
	character caseid*40, version*20, filename*148, filename2*148
	character name*80
	integer nzm, ntime,npar
	real time(50000),z(5000),p(5000),dz(5000)
        real f(500000), f1(500000), parms(500000)
        real tmp(5000), tmp1(5000), tmp2(5000), tmp3(5000)
        real rho(5000)
	real cape,cin

	integer i,j,k,l,m
	integer vdimids(2), ndimids
	integer ncid,err,zid,timeid,varid
        character(80) long_name
        character(10) abbr_name
        character(12) units

	integer iargc
	external iargc

	time=0.
	z=0.
	p=0.
	f=0.
c-------------------------------------------------------	
	m=COMMAND_ARGUMENT_COUNT()
!	m=iargc()
	if(m.eq.0.) then
	 print*,'you forgot to specify the name of the file.'
	 stop
	endif	
	call getarg(1,filename)
	print *,'open file: ',filename
	open(2,file=filename,status='old',form='unformatted')

	do i=1,144
	 if(filename(i:i+4).eq.'.stat') then
	  filename2=filename(1:i-1)//'.nc'
	  print*,filename2
	 endif
	end do
c-------------------------------------------------------	
c Count how many time points in a file:
	call HBUF_info(2,ntime,time,nzm,z,p,caseid,version)
	call HBUF_read(2,nzm,'RHO',1,1,rho,m)
	print*,'.......',ntime
	print*,(time(k),k=1,ntime/3)
	print*
	print*,(z(k),k=1,nzm)
	print*
	print*,(p(k),k=1,nzm)
	print*
	print*,(rho(k),k=1,nzm)

        dz(1) = 0.5*(z(1)+z(2))
        do k=2,nzm-1
           dz(k) = 0.5*(z(k+1)-z(k-1))
        end do
        dz(nzm) = dz(nzm-1)

	err = NF_CREATE(filename2, NF_CLOBBER, ncid)

	err = NF_REDEF(ncid)

	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'SAM version',
     &                           len_trim(version),version)
	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'caseid',
     &                                len_trim(caseid),caseid)

	err = NF_DEF_DIM(ncid, 'z', nzm, zid)
	err = NF_DEF_DIM(ncid, 'time', ntime, timeid)

	err = NF_DEF_VAR(ncid, 'z', NF_FLOAT, 1, zid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',1,'m')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',6,'height')

	err = NF_DEF_VAR(ncid, 'time', NF_FLOAT, 1, timeid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',3,'day')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',4,'time')

	err = NF_DEF_VAR(ncid, 'p', NF_FLOAT, 1, zid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',2,'mb')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',8,'pressure')

	err = NF_ENDDEF(ncid)

	err = NF_INQ_VARID(ncid,'time',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, time)

	err = NF_INQ_VARID(ncid,'z',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, z)


	err = NF_INQ_VARID(ncid,'p',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, p)
!
!  1-D fields now:
!

	call HBUF_parms(2,parms,nparms)
        call HBUF_read(2,nzm,'RHO',1,1,rho,m)	
c--------------------------------------------------------
        long_name = 'SST'
        abbr_name = 'SST'
        units = 'K'
	npar = 1
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Surface Pressure'
        abbr_name = 'Ps'
        units = ' '
	npar = 2
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Shaded Cloud Fraction'
        abbr_name = 'CLDSHD'
        units = ' '
	npar = 3
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Surface Precip. Fraction'
        abbr_name = 'AREAPREC'
        units = ' '
        npar = 4
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Cloud Fraction above 245K level'
        abbr_name = 'CLD245'
        units = ' '
        npar = 5
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Maximum Updraft Velocity'
        abbr_name = 'WMAX'
        units = 'm/s'
        npar = 6
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Maximum Horizontal Wind'
        abbr_name = 'UMAX'
        units = 'm/s'
        npar = 7
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c----------------------------------------------------------
        long_name = 'Surface Precipitation'
        abbr_name = 'PREC'
        units = 'mm/day'
        call HBUF_read(2,nzm,'PRECIP',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=f(1+(i-1)*nzm)
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Latent Heat Flux'
        abbr_name = 'LHF'
        units = 'W/m2'
        call HBUF_read(2,nzm,'QTFLUX',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=f(1+(i-1)*nzm)
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Sensible Heat Flux'
        abbr_name = 'SHF'
        units = 'W/m2'
        call HBUF_read(2,nzm,'TLFLUX',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=f(1+(i-1)*nzm)
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Precipitable Water'
        abbr_name = 'PW'
        units = 'mm'
        call HBUF_read(2,nzm,'QV',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
         tmp(i)=tmp(i)*1.e-3
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Observed Precipitable Water'
        abbr_name = 'PWOBS'
        units = 'mm'
        call HBUF_read(2,nzm,'QVOBS',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
         tmp(i)=tmp(i)*1.e-3
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c----------------------------------------------------------
        long_name = 'Cloud Water Path'
        abbr_name = 'CWP'
        units = 'g/m2'
        call HBUF_read(2,nzm,'QC',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Ice Water Path'
        abbr_name = 'IWP'
        units = 'g/m2'
        call HBUF_read(2,nzm,'QI',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Rain Water Path'
        abbr_name = 'RWP'
        units = 'g/m2'
        call HBUF_read(2,nzm,'QR',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Snow Water Path'
        abbr_name = 'SWP'
        units = 'g/m2'
        call HBUF_read(2,nzm,'QS',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'Grauple Water Path'
        abbr_name = 'GWP'
        units = 'g/m2'
        call HBUF_read(2,nzm,'QG',1,ntime,f,m)
        do i=1,ntime
         tmp(i)=0.
         do k=1,nzm-1
          tmp(i)=tmp(i)+rho(k)*f(k+(i-1)*nzm)*dz(k)
         end do
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c----------------------------------------------------------
        long_name = 'CAPE'
        abbr_name = 'CAPE'
        units = 'J/kg'
        call HBUF_read(2,nzm,'TABS',1,ntime,f,m)
        call HBUF_read(2,nzm,'QV',1,ntime,f1,m)
        do i=1,ntime
         tmp(1:nzm)=f(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp1(1:nzm)=f1(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp2(i) =  cape(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
         tmp3(i) =  cin(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
          print*,'CAPE: CIN:',tmp2(i), tmp3(i)
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp2)
        print*,long_name

c----------------------------------------------------------
        long_name = 'CAPEOBS'
        abbr_name = 'CAPEOBS'
        units = 'J/kg'
        call HBUF_read(2,nzm,'TABSOBS',1,ntime,f,m)
        call HBUF_read(2,nzm,'QVOBS',1,ntime,f1,m)
        do i=1,ntime
         tmp(1:nzm)=f(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp1(1:nzm)=f1(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp2(i) =  cape(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
         tmp3(i) =  cin(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
	  print*,'CAPEOBS: CINOBS:',tmp2(i), tmp3(i)
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
	err = NF_PUT_VAR_REAL(ncid, varid, tmp2)
        print*,long_name
c----------------------------------------------------------
        long_name = 'CIN'
        abbr_name = 'CIN'
        units = 'J/kg'
        call HBUF_read(2,nzm,'TABS',1,ntime,f,m)
        call HBUF_read(2,nzm,'QV',1,ntime,f1,m)
        do i=1,ntime
         tmp(1:nzm)=f(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp1(1:nzm)=f1(1+(i-1)*nzm:nzm+(i-1)*nzm) 
         tmp2(i) =  cin(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp2)
        print*,long_name
c----------------------------------------------------------
        long_name = 'CINOBS'
        abbr_name = 'CINOBS'
        units = 'J/kg'
        call HBUF_read(2,nzm,'TABSOBS',1,ntime,f,m)
        call HBUF_read(2,nzm,'QVOBS',1,ntime,f1,m)
        do i=1,ntime
         tmp(1:nzm)=f(1+(i-1)*nzm:nzm+(i-1)*nzm)
         tmp1(1:nzm)=f1(1+(i-1)*nzm:nzm+(i-1)*nzm) 
         tmp2(i) =  cin(nzm,p(1:nzm),tmp(1:nzm),tmp1(1:nzm))
        end do
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp2)
        print*,long_name




c--------------------------------------------------------
	if(nparms.gt.8) then

	print*,'nparms=',nparms

        long_name = 'Net LW flux at sfc'
        abbr_name = 'LWNS'
        units = 'W/m2'
        npar = 9
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Net LW flux at Top-of-Model)'
        abbr_name = 'LWNT'
        units = 'W/m2'
        npar = 10
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Net LW flux at TOA'
        abbr_name = 'LWNTOA'
        units = 'W/m2'
        npar = 11
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Net LW flux at sfc (Clear Sky)'
        abbr_name = 'LWNSC'
        units = 'W/m2'
        npar = 12
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Net LW flux at TOA (Clear Sky)'
        abbr_name = 'LWNTOAC'
        units = 'W/m2'
        npar = 13
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Downward LW flux at sfc'
        abbr_name = 'LWDS'
        units = 'W/m2'
        npar = 14
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Net SW flux at sfc'
        abbr_name = 'SWNS'
        units = 'W/m2'
        npar = 15
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Net SW flux at Top-of-Model'
        abbr_name = 'SWNT'
        units = 'W/m2'
        npar = 16
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Net SW flux at TOA'
        abbr_name = 'SWNTOA'
        units = 'W/m2'
        npar = 17
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Net SW flux at sfc (Clear Sky)'
        abbr_name = 'SWNSC'
        units = 'W/m2'
        npar = 18
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Net SW flux at TOA (Clear Sky)'
        abbr_name = 'SWNTOAC'
        units = 'W/m2'
        npar = 19
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Downward SW flux at sfc'
        abbr_name = 'SWDS'
        units = 'W/m2'
        npar = 20
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Incoming SW flux at TOA'
        abbr_name = 'SOLIN'
        units = 'W/m2'
        npar = 21
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Observed SST'
        abbr_name = 'SSTOBS'
        units = 'K'
        npar = 22
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Observed Latent Heat Flux'
        abbr_name = 'LHFOBS'
        units = 'W/m2'
        npar = 23
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Observed Sensible Heat Flux'
        abbr_name = 'SHFOBS'
        units = 'SHFOBS'
        npar = 24
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'Low Cloud Fraction'
        abbr_name = 'CLDLOW'
        units = ' '
	npar = 25
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'Middle Cloud Fraction'
        abbr_name = 'CLDMID'
        units = ' '
	npar = 26
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'High Cloud Fraction'
        abbr_name = 'CLDHI'
        units = ' '
	npar = 27
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Total Cloud Fraction (tau > 0.3)'
        abbr_name = 'ISCCPTOT'
        units = ' '
        npar = 28
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Low Cloud Fraction (tau > 0.3)'
        abbr_name = 'ISCCPLOW'
        units = ' '
        npar = 29
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Middle Cloud Fraction (tau > 0.3)'
        abbr_name = 'ISCCPMID'
        units = ' '
        npar =30 
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP High Cloud Fraction (tau > 0.3)'
        abbr_name = 'ISCCPHGH'
        units = ' '
        npar = 31
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'MODIS Total Cloud Fraction'
        abbr_name = 'MODISTOT'
        units = ' '
        npar = 32
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Low Cloud Fraction'
        abbr_name = 'MODISLOW'
        units = ' '
        npar = 33
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Middle Cloud Fraction'
        abbr_name = 'MODISMID'
        units = ' '
        npar = 34
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS High Cloud Fraction (tau > 0.3)'
        abbr_name = 'MODISHGH'
        units = ' '
        npar = 35
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MISR Total Cloud Fraction'
        abbr_name = 'MISRTOT'
        units = ' '
        npar = 36
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'MODIS Effective Radius (Liquid)'
        abbr_name = 'MODISREL'
        units = 'mkm'
        npar = 37
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'MODIS Effective Radius (Ice)'
        abbr_name = 'MODISREI'
        units = 'mkm'
        npar = 38
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name

c--------------------------------------------------------
        long_name = 'MODIS Liquid Water Path'
        abbr_name = 'MODISLWP'
        units = 'g/m2'
        npar = 39
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Ice Water Path'
        abbr_name = 'MODISIWP'
        units = 'g/m2'
        npar = 40
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Brightness Temperature'
        abbr_name = 'ISCCPTB'
        units = 'K'
        npar = 41
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Brightness Temperature (Clear Sky)'
        abbr_name = 'ISCCPTBCLR'
        units = 'K'
        npar = 42
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Total Fraction (Liquid)'
        abbr_name = 'MODISTOTL'
        units = ' '
        npar = 43
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Total Fraction (Ice)'
        abbr_name = 'MODISTOTI'
        units = ' '
        npar = 44            
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name                
c--------------------------------------------------------
        long_name = 'ISCCP Optical Path'
        abbr_name = 'ISCCPTAU'
        units = ' '
        npar = 45
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Cloud Albedo'
        abbr_name = 'ISCCPALB'
        units = ' '
        npar = 46
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'ISCCP Cloud-Top Pressure'
        abbr_name = 'ISCCPPTOP'
        units = 'mb'
        npar = 47
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Cloud Optical Path'
        abbr_name = 'MODISTAU'
        units = ' '
        npar = 48
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Cloud Optical Path (Liquid)'
        abbr_name = 'MODISTAUL'
        units = ' '
        npar = 49
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Cloud Optical Path (Ice)'
        abbr_name = 'MODISTAUI'
        units = ' '
        npar = 50
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MODIS Cloud-Top Pressure'
        abbr_name = 'MODISPTOP'
        units = 'mb'
        npar = 51
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'MISR Cloud-Top Height'
        abbr_name = 'MISRZTOP'
        units = 'km'
        npar = 52
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = 0.001*parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Inversion Height'
        abbr_name = 'ZINV'
        units = 'km'
        npar = 53
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Variance of the Inversion Height'
        abbr_name = 'ZINV2'
        units = 'km2'
        npar = 54
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Mean Cloud-top Height'
        abbr_name = 'ZCT'
        units = 'km'
        npar = 55
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Variance of Cloud-top Height'
        abbr_name = 'ZCT2'
        units = 'km2'
        npar = 56
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Maximum Cloud-top Height'
        abbr_name = 'ZCTMAX'
        units = 'km'
        npar = 57
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Mean Cloud-base Height'
        abbr_name = 'ZCB'
        units = 'km'
        npar = 58
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Variance of Cloud-base Height'
        abbr_name = 'ZCB2'
        units = 'km'
        npar = 59
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Minimum Cloud-base Height'
        abbr_name = 'ZCBMIN'
        units = 'km'
        npar = 60
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Liquid Water Path'
        abbr_name = 'LWP'
        units = 'g/m2'
        npar = 61
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = 1000.*parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Variance of Liquid Water Path'
        abbr_name = 'LWP2'
        units = '(g/m2)^2'
        npar = 62
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = 1000000.*parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Precipitation Rate'
        abbr_name = 'PRECMN'
        units = 'mm/d'
        npar = 63
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Variance of Precipitation Rate'
        abbr_name = 'PREC2'
        units = '(mm/d)^2'
        npar = 64
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Maximum Precipitation Rate'
        abbr_name = 'PRECMAX'
        units = 'mm/d'
        npar = 65
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
c--------------------------------------------------------
        long_name = 'GCSS Mean Drop Number Comcentration'
        abbr_name = 'NCMN'
        units = '#/cm3'
        npar = 66
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
        print*,tmp(1:ntime)
c--------------------------------------------------------
        long_name = 'GCSS Mean Rain Number Comcentration'
        abbr_name = 'NRMN'
        units = '#/cm3'
        npar = 67
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
        print*,tmp(1:ntime)
c--------------------------------------------------------
        long_name = 'GCSS Precip. over threshold Area Fraction'
        abbr_name = 'AREAPRTHR'
        units = ' '
        npar = 68
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_PUT_ATT_REAL(ncid,varid,'_FillValue',
     &                       NF_FLOAT,1,-1.)
        err = NF_ENDDEF(ncid)
        tmp(1:ntime) = parms(npar:npar+nparms*(ntime-1):nparms)
        err = NF_PUT_VAR_REAL(ncid, varid, tmp)
        print*,long_name
        print*,tmp(1:ntime)


c--------------------------------------------------------

        if(npar.ne.nparms) then
          print*,'number of parameters is not == to nparas'
          print*,'npar=',npar,'  nparms=',nparms
          stop
        end if



        end if

c--------------------------------------------------------------
        ndimids=2
        vdimids(1) = zid
        vdimids(2) = timeid

        do k=1,hbuf_length

           write(6,'(a72)') deflist(k)
           call HBUF_read(2,nzm,namelist(k),1,ntime,f,m)

           err = NF_REDEF(ncid)
           name = namelist(k)
           l=len_trim(name)
           err = NF_DEF_VAR(ncid,name(1:l),NF_FLOAT,
     &                                  ndimids,vdimids,varid)
           err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                                len_trim(deflist(k)),deflist(k))
           err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                              len_trim(unitlist(k)),unitlist(k))
           err = NF_PUT_ATT_REAL(ncid,varid,'missing_value',NF_FLOAT,
     &                              1,-9999.)
           err = NF_ENDDEF(ncid)
           err = NF_PUT_VAR_REAL(ncid, varid, f)
           if(err.ne.0) print*,'error:',err

        end do


	err = NF_CLOSE(ncid)

	end





