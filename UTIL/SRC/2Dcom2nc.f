c
c (C) 2000 Marat Khairoutdinov
c

	implicit none
	include 'netcdf.inc'

c---------------------------------------------------------------
c variables:

	character(120) filename,filename1
	character(80) long_name
	character(10) units
	character(8)  name
	character(1)  blank
	character(10) c_min, c_max
	character (12)c_dx, c_dy, c_time	
	character(3) nm

	integer(2), allocatable :: byte(:)
	real, allocatable :: fld(:)
	real fmax,fmin,day0
	real dx,dy,x(100000),y(100000),time, time_old
	logical condition
	integer nsubs,nsubsx,nsubsy,nx,ny,nz,nfields,nstep
	integer i,j,k,k1,k2,n,i0,j0,nx_gl,ny_gl,length,ifields

	integer vdimids(3),start(3),count(3),ndimids,nfile
	integer ncid,err,yid,xid,timeid,ntime,varrr
	integer nrecords
c External functions:

	integer iargc,strlen1
	external iargc,strlen1

	real fldmin, fldmax


	nrecords = 1000! max number ot time steps in one file
c---------------------------------------------------------------
c---------------------------------------------------------------
c
c Read the file-name from the comman line:
c
	i=COMMAND_ARGUMENT_COUNT()
	if(i.eq.0) then
	  print*,'no input-file name is specified.'
	  print*,'Format: 2Dcom2nc input.2Dcom'
	  stop
	end if
	call getarg(1,filename)

c---------------------------------------------------------------
c Read files; merge data from different subdomains;
c save as a netcdf file.
c
	open(1,file=filename,status='old',form='unformatted')


	ntime=1
	nfile=0
        time_old = -1.

      do while(.true.) ! infinit loop 

c
c The output filename:

        condition=mod(ntime-1,nrecords).eq.0
        if(condition) then
	  ntime=1
          nfile=nfile+1
 	  filename1=filename
          do i=1,115
            if(filename1(i:i+5).eq.'.2Dcom') then
	      if(nfile.lt.10) then
              write(filename1(i+6:i+12),'(a1,i1,a3)') '_',nfile,'.nc'
	      else if(nfile.lt.100) then
              write(filename1(i+6:i+13),'(a1,i2,a3)') '_',nfile,'.nc'
	      else if(nfile.lt.1000) then
              write(filename1(i+6:i+14),'(a1,i3,a3)') '_',nfile,'.nc'
	      end if
	      print*,filename1
	      EXIT
	    else if(i.eq.116) then
	      print*,'wrong file name extension!'
	      stop
            endif
          end do
	  if(nfile.ne.1) err = NF_CLOSE(ncid)
        end if

	read(1,end=3333,err=3333) nstep
	read(1) long_name(1:32)
	print*,long_name(1:32)
	read(long_name,'(8i4)') nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	read(1) c_dx
	read(1) c_dy
	read(1) c_time
	print*,nstep,c_time,ntime+(nfile-1)*nrecords,c_dx,c_dy
	read(c_dx,'(f12.0)') dx
	read(c_dy,'(f12.0)') dy
	read(c_time,'(f12.5)') time

	print*,'nx,ny,nz,nsubs,nsubsx,nsubsy,nfields:'
	print*,nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	
	nx_gl=nx*nsubsx
	ny_gl=ny*nsubsy
	if(condition) then
	 print*,'nx_gl=',nx_gl, '    dx=',dx
	 print*,'ny_gl=',ny_gl, '    dy=',dy
	end if

	do i=1,nx_gl
	 x(i) = dx*(i-1)
	end do
	do j=1,ny_gl
	 y(j) = dy*(j-1)
	end do

	if(ntime.eq.1.and.nfile.eq.1) then
	  day0=time
          allocate(byte(nx*ny))
	  allocate(fld(nx_gl*ny_gl))
        end if
c
c Initialize netcdf stuff, define variables,etc.
c
       if(condition) then



	err = NF_CREATE(filename1, NF_CLOBBER, ncid)
	err = NF_REDEF(ncid)

	err = NF_DEF_DIM(ncid, 'x', nx_gl, xid)
	if(ny_gl.ne.1)err = NF_DEF_DIM(ncid, 'y', ny_gl, yid)
	err = NF_DEF_DIM(ncid, 'time', NF_UNLIMITED, timeid)

        err = NF_DEF_VAR(ncid, 'x', NF_FLOAT, 1, xid, varrr)
	err = NF_PUT_ATT_TEXT(ncid,varrr,'units',1,'m')
	if(ny_gl.ne.1) then
         err = NF_DEF_VAR(ncid, 'y', NF_FLOAT, 1, yid, varrr)
	 err = NF_PUT_ATT_TEXT(ncid,varrr,'units',1,'m')
	endif
        err = NF_DEF_VAR(ncid, 'time', NF_FLOAT, 1, timeid, varrr)
	err = NF_PUT_ATT_TEXT(ncid,varrr,'units',3,'day')
        err = NF_PUT_ATT_TEXT(ncid,varrr,'long_name',4,'time')

	err = NF_ENDDEF(ncid)

	err = NF_INQ_VARID(ncid,'x',varrr)
	err = NF_PUT_VAR_REAL(ncid, varrr, x)
	if(ny_gl.ne.1) then
	err = NF_INQ_VARID(ncid,'y',varrr)
	err = NF_PUT_VAR_REAL(ncid, varrr, y)
	endif

	end if ! condition


	if(ny_gl.ne.1) then
	 ndimids=3
	 vdimids(1) = xid
	 vdimids(2) = yid
	 vdimids(3) = timeid
	 start(1) = 1
         start(2) = 1
         start(3) = ntime 
	 count(1) = nx_gl
         count(2) = ny_gl
         count(3) = 1 
	else
	 ndimids=2
	 vdimids(1) = xid
	 vdimids(2) = timeid
	 start(1) = 1
         start(2) = ntime 
	 count(1) = nx_gl
         count(2) = 1 
	endif

	

	do ifields=1,nfields
	
         read(1) name,blank,long_name,blank,units,blank,c_max,c_min
	  print*,name,strlen1(long_name),
     &      long_name(1:strlen1(long_name)),'   ',c_max,c_min

	  do n=0,nsubs-1
	  
     	    read(1) (byte(k),k=1,nx*ny)

	    j0 = n/nsubsx 
	    i0 = n - j0*nsubsx	
	    i0 = i0 * (nx_gl/nsubsx) 
	    j0 = j0 * (ny_gl/nsubsy)  
	    length=0
	     read(c_min,*) fmin
	     read(c_max,*) fmax
	     do j=1+j0,ny+j0
	      do i=1+i0,nx+i0
		length=length+1
		fld(i+nx_gl*(j-1))= 
     &                  fmin+(byte(length)+32000)*(fmax-fmin)/64000.
	      end do
	     end do

	  end do ! n

	 if(condition) then
	  err = NF_REDEF(ncid)
          err = NF_DEF_VAR(ncid,name,NF_FLOAT,
     &                           ndimids,vdimids,varrr)
	  err = NF_PUT_ATT_TEXT(ncid,varrr,'long_name',
     &		strlen1(long_name),long_name(1:strlen1(long_name)))
	  err = NF_PUT_ATT_TEXT(ncid,varrr,'units',
     &		strlen1(units),units(1:strlen1(units)))
	  err = NF_ENDDEF(ncid)
	 end if
	  
	
         if(time.gt.time_old) then
	   err = NF_INQ_VARID(ncid,name,varrr)
           err = NF_PUT_VARA_REAL(ncid,varrr,start,count,fld)
         end if

	end do

        if(time.gt.time_old) then
	   err = NF_INQ_VARID(ncid,'time',varrr)
           err = NF_PUT_VAR1_REAL(ncid,varrr,ntime,time)
           time_old = time
           ntime=ntime+1
        end if    


      end do ! while


 3333	continue

	err = NF_CLOSE(ncid)

	end
	




	integer function strlen1(str)
	character*(*) str
	strlen1=len(str)
	do i=len(str),1,-1
	  if(str(i:i).ne.' ') then
	    strlen1=i
	    return
	  endif 
	end do
        return
	end
