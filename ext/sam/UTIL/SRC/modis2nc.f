c-------------------------------------------------------
cbloss(2016-02-11): Based on isccp2nc.f
	implicit none
	include 'netcdf.inc'
	include 'hbuf.inc'	
	
	integer flag(HBUF_MAX_LENGTH)
c-------------------------------------------------------	
	character caseid*80, filename*88, filename2*88
	character name*80
	integer npres, ntime,ntau
	real time(10000)
        real, allocatable ::  fq_modis(:,:,:)  !accumulated fq_modis
        real, allocatable ::  taulim(:), preslim(:)

        real, allocatable ::  totalcldarea(:)
        real, allocatable ::  lowcldarea(:),midcldarea(:),hghcldarea(:)
        real, allocatable ::  meanptop(:),meantaucld(:)
        real, allocatable ::  totalcldarea_liq(:),totalcldarea_ice(:)
        real, allocatable ::  meantaucld_liq(:),meantaucld_ice(:)
        real, allocatable ::  meanwaterpath_liq(:),meanwaterpath_ice(:)
        real, allocatable ::  meaneffrad_liq(:),meaneffrad_ice(:)

	integer i,j,k,l,m,nparms
	integer vdimids(4), ndimids
	integer ncid,err,tid,zid,timeid,varid
        character(80) long_name
        character(30) abbr_name
        character(12) units

	integer iargc
	external iargc

	time=0.
c-------------------------------------------------------	
	m=COMMAND_ARGUMENT_COUNT()
	if(m.eq.0.) then
	 print*,'you forgot to specify the name of the file.'
	 stop
	endif	
	call getarg(1,filename)
	print *,'open file: ',filename
	open(2,file=filename,status='old',form='unformatted')

	do i=1,83
	 if(filename(i:i+5).eq.'.modis') then
	  filename2=filename(1:i-1)//'.modis.nc'
	  print*,filename2
	 endif
	end do
c-------------------------------------------------------	
c Count how many time points in a file:
	
        ntime = 0
        do while(.true.)
         read(2,end=111)  caseid
         ntime=ntime+1
         read (2) time(ntime)
         read (2) ntau,npres
         read (2) 

         read (2) 
         read (2) 
         read (2) 
         read (2) 

         read (2) 
         read (2) 
	 print*,time(ntime)
        end do
 111    continue
        rewind(2)

	print*,caseid
	print*,'ntau=',ntau
	print*,'npres=',npres
	print*,'ntime=',ntime

	allocate (preslim(npres), taulim(ntau))
	allocate (fq_modis(ntau,npres,ntime))
        allocate (totalcldarea(ntime))
        allocate (lowcldarea(ntime),midcldarea(ntime),hghcldarea(ntime))
        allocate (meantaucld(ntime),meanptop(ntime))
        allocate (totalcldarea_liq(ntime),totalcldarea_ice(ntime))
        allocate (meantaucld_liq(ntime),meantaucld_ice(ntime))
        allocate (meanwaterpath_liq(ntime),meanwaterpath_ice(ntime))
        allocate (meaneffrad_liq(ntime),meaneffrad_ice(ntime))

	do i=1,ntime
	 read(2)
	 read(2) time(i)
         read(2) ntau,npres
         read(2) taulim(1:ntau)
         read(2) preslim(1:npres)
         read(2) fq_modis(1:ntau,1:npres,i) 
         read(2) totalcldarea(i),
     &            lowcldarea(i),midcldarea(i),hghcldarea(i)
         read(2) meantaucld(i),meanptop(i)
	 read(2) totalcldarea_liq(i), meantaucld_liq(i),
     &             meanwaterpath_liq(i),meaneffrad_liq(i)
	 read(2) totalcldarea_ice(i), meantaucld_ice(i),
     &             meanwaterpath_ice(i),meaneffrad_ice(i)
        end do

	err = NF_CREATE(filename2, NF_CLOBBER, ncid)

	err = NF_REDEF(ncid)

	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'model',7,'CSU CEM')
	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'caseid',
     &                                  len_trim(caseid),caseid)
	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'MissingValue',2,'-1')

	err = NF_DEF_DIM(ncid, 'time', ntime, timeid)
	err = NF_DEF_DIM(ncid, 'pres', npres, zid)
	err = NF_DEF_DIM(ncid, 'tau', ntau, tid)

	err = NF_DEF_VAR(ncid, 'pres', NF_FLOAT, 1, zid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',3,'hPa')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',7,'pressure')

	err = NF_DEF_VAR(ncid, 'time', NF_FLOAT, 1, timeid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',3,'day')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',4,'time')

	err = NF_DEF_VAR(ncid, 'tau', NF_FLOAT, 1, tid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',1,' ')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                                 17,'Optical thickness')

	err = NF_ENDDEF(ncid)

        err = NF_INQ_VARID(ncid,'time',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, time)
        err = NF_INQ_VARID(ncid,'pres',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, preslim)
        err = NF_INQ_VARID(ncid,'tau',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, taulim)
c--------------------------------------------------------


        long_name = 'MODIS Total Cloud Area tau > 0.3'
        abbr_name = 'CLDTOT'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, totalcldarea(1:ntime))

c--------------------------------------------------------
        long_name = 'MODIS Low Cloud Area tau > 0.3'
        abbr_name = 'CLDLOW'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, lowcldarea(1:ntime))

c--------------------------------------------------------
        long_name = 'MODIS Middle Cloud Area tau > 0.3'
        abbr_name = 'CLDMID'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, midcldarea(1:ntime))

c--------------------------------------------------------
        long_name = 'MODIS High Cloud Area tau > 0.3'
        abbr_name = 'CLDHGH'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, hghcldarea(1:ntime))

c--------------------------------------------------------
        long_name = 'MODIS mean optical depth for tau > 0.3'
	abbr_name = 'TAUMEAN'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meantaucld(1:ntime))

c--------------------------------------------------------
        long_name = 'MODIS mean cloud top pressure for tau > 0.3'
	abbr_name = 'PTOPMEAN'
        units = 'hPa'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meanptop(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Total Liquid Cloud Area tau > 0.3'
        abbr_name = 'CLDTOTLIQ'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, totalcldarea_liq(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Liquid Cloud Optical Depth, tau>0.3'
        abbr_name = 'LIQTAUMEAN'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meantaucld_liq(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Liquid Cloud Water Path, tau>0.3'
        abbr_name = 'LWP_MODIS'
        units = 'g/m2'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meanwaterpath_liq(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Liquid Cloud Effective Radius, tau>0.3'
        abbr_name = 'EFFR_LIQ'
        units = 'um'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meaneffrad_liq(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Total Ice Cloud Area tau > 0.3'
        abbr_name = 'CLDTOTICE'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, totalcldarea_ice(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Ice Cloud Optical Depth, tau>0.3'
        abbr_name = 'ICETAUMEAN'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meantaucld_ice(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Ice Cloud Water Path, tau>0.3'
        abbr_name = 'IWP_MODIS'
        units = 'g/m2'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meanwaterpath_ice(1:ntime))

c--------------------------------------------------------

        long_name = 'MODIS Mean Ice Cloud Effective Radius, tau>0.3'
        abbr_name = 'EFFR_ICE'
        units = 'um'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meaneffrad_ice(1:ntime))

c--------------------------------------------------------
        ndimids=3
        vdimids(1) = tid
        vdimids(2) = zid
        vdimids(3) = timeid

        long_name = 'MODIS Table'
        abbr_name = 'modis'
        units = ' '
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  ndimids,vdimids,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                   len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                   len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, 
     &                     fq_modis(1:ntau,1:npres,1:ntime))


	err = NF_CLOSE(ncid)

	end





