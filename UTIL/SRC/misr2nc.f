c-------------------------------------------------------
cbloss(2016-02-11): Based on isccp2nc.f
	implicit none
	include 'netcdf.inc'
	include 'hbuf.inc'	
	
	integer flag(HBUF_MAX_LENGTH)
c-------------------------------------------------------	
	character caseid*80, filename*88, filename2*88
	character name*80
	integer ncth, ntime,ntau
	real time(10000)
        real, allocatable ::  fq_misr(:,:,:)  !accumulated fq_misr
        real, allocatable ::  taulim(:), heightlim(:)

        real, allocatable ::  totalcldarea(:), meanztop(:)

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
	 if(filename(i:i+4).eq.'.misr') then
	  filename2=filename(1:i-1)//'.misr.nc'
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
         read (2) ntau,ncth
c These lines will hold the optical detpth and cloud top height bin edges
         read (2) 
         read (2) 
c These lines will hold the histograms and aggregate numbers.
         read (2) 
         read (2) 

	 print*,time(ntime)
        end do
 111    continue
        rewind(2)

	print*,caseid
	print*,'ntau=',ntau
	print*,'ncth=',ncth
	print*,'ntime=',ntime
	print*,'Note: tau = cloud optical depth, cth = cloud top height'

	allocate (heightlim(ncth), taulim(ntau))
	allocate (fq_misr(ntau,ncth,ntime))
        allocate (totalcldarea(ntime))
        allocate (meanztop(ntime))

	do i=1,ntime
	 read(2)
	 read(2) time(i)
         read(2) ntau,ncth
         read(2) taulim(1:ntau)
         read(2) heightlim(1:ncth)
         read(2) fq_misr(1:ntau,1:ncth,i) 
         read(2) totalcldarea(i), meanztop(i)
        end do

	err = NF_CREATE(filename2, NF_CLOBBER, ncid)

	err = NF_REDEF(ncid)

	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'model',7,'CSU CEM')
	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'caseid',
     &                                  len_trim(caseid),caseid)
	err = NF_PUT_ATT_TEXT(ncid,NF_GLOBAL,'MissingValue',2,'-1')

	err = NF_DEF_DIM(ncid, 'time', ntime, timeid)
	err = NF_DEF_DIM(ncid, 'cth', ncth, zid)
	err = NF_DEF_DIM(ncid, 'tau', ntau, tid)

	err = NF_DEF_VAR(ncid, 'cth', NF_FLOAT, 1, zid, varid)
	err = NF_PUT_ATT_TEXT(ncid,varid,'units',3,'km ')
	err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',16,'cloud top height')

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
        err = NF_INQ_VARID(ncid,'cth',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, heightlim)
        err = NF_INQ_VARID(ncid,'tau',varid)
	err = NF_PUT_VAR_REAL(ncid, varid, taulim)
c--------------------------------------------------------


        long_name = 'MISR Total Cloud Area tau > 0.3'
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
        long_name = 'MISR mean cloud top height for tau > 0.3'
	abbr_name = 'ZTOPMEAN'
        units = 'm'
        err = NF_REDEF(ncid)
        err = NF_DEF_VAR(ncid,trim(abbr_name),NF_FLOAT,
     &                                  1, timeid,varid)
        err = NF_PUT_ATT_TEXT(ncid,varid,'long_name',
     &                  len_trim(long_name),trim(long_name))
        err = NF_PUT_ATT_TEXT(ncid,varid,'units',
     &                       len_trim(units),trim(units))
        err = NF_ENDDEF(ncid)
        err = NF_PUT_VAR_REAL(ncid, varid, meanztop(1:ntime))

c--------------------------------------------------------
        ndimids=3
        vdimids(1) = tid
        vdimids(2) = zid
        vdimids(3) = timeid

        long_name = 'MISR Table'
        abbr_name = 'misr'
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
     &                     fq_misr(1:ntau,1:ncth,1:ntime))


	err = NF_CLOSE(ncid)

	end





