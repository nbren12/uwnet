c
c (C) 2000 Marat Khairoutdinov
c

	implicit none
	include 'netcdf.inc'

c---------------------------------------------------------------
c variables:

	character(200) filename,filename_out
        character(80) long_name
	character(10) units
	character(8)  name
	character(1)  blank
	character(10) c_min, c_max
	character (12) c_dx, c_dy, c_time	
	character(3) nm
        character(4) rankchar

	integer(2), allocatable :: byte(:)
	real, allocatable :: fld(:)
	real fmax,fmin
	real dx,dy,x(100000),y(100000),time
	logical condition
	integer nsubs,nsubsx,nsubsy,nx,ny,nz,nfields,nstep
	integer i,j,k,k1,k2,n,i0,j0,nx_gl,ny_gl,length,ifields

	integer vdimids(3),start(3),count(3),ndimids
	integer ncid,err,yid,xid,timeid,ntime,varrr
c External functions:

	integer,external :: iargc
!	external iargc

	real fldmin, fldmax

c---------------------------------------------------------------
c
c Read the file-name from the comman line:
c
	i=iargc()
	if(i.eq.0) then
	  print*,'no input-file name is specified.'
	  print*,'Format: 2Dcom2nc input.2Dcom (without trailing _*)'
	  stop
	end if
	call getarg(1,filename)

c---------------------------------------------------------------
c Read files; merge data from different subdomains;
c save as a netcdf file.
c
	open(1,file=trim(filename)//"_0",status='old',form='unformatted')


	read(1,end=3333,err=3333) nstep
	read(1) long_name(1:32)
	print*,long_name(1:32)
	read(long_name,'(8i4)') nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	read(1) c_dx
	read(1) c_dy
	read(1) c_time
	print*,c_time,ntime
	read(c_dx,'(f12.0)') dx
	read(c_dy,'(f12.0)') dy
	read(c_time,'(f12.5)') time
        ntime=1

	print*,'nx,ny,nz,nsubs,nsubsx,nsubsy,nfields:'
	print*,nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	
	nx_gl=nx*nsubsx
	ny_gl=ny*nsubsy
        print*,'nx_gl=',nx_gl, '    dx=',dx
        print*,'ny_gl=',ny_gl, '    dy=',dy

	do i=1,nx_gl
	 x(i) = dx*(i-1)
	end do
	do j=1,ny_gl
	 y(j) = dy*(j-1)
	end do

        allocate(byte(nx*ny))
	allocate(fld(nx_gl*ny_gl))
c
c The output filename:

 	  filename_out=filename
          do i=1,195
            if(filename_out(i:i+5).eq.'.2Dcom') then
              filename_out(i:i+5)='.nc  '
	      EXIT
	    else if(i.eq.195) then
	      print*,'wrong file name extension!'
	      stop
            endif
          end do
c
c Initialize netcdf stuff, define variables,etc.
c



	err = NF_CREATE(filename_out, NF_CLOBBER, ncid)
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

	

	do ifields=0,nfields-1
	

	  do n=0,nsubs-1

            if(n.ne.0) then
              write(rankchar,'(i4)') n
              open(2,file=trim(filename)//"_"//trim(adjustl(rankchar)),
     &          status='old',form='unformatted')
              read(2)
              do i = 1,ifields
                read(2)
              end do
     	      read(2) (byte(k),k=1,nx*ny)
            else
             read(1) name,blank,long_name,blank,units,blank,c_max,c_min
	     print*,ifields+1,name,trim(long_name),'   ',c_max,c_min
             read(1) (byte(k),k=1,nx*ny)
            end if

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

	  err = NF_REDEF(ncid)
          err = NF_DEF_VAR(ncid,name,NF_FLOAT,
     &                           ndimids,vdimids,varrr)
	  err = NF_PUT_ATT_TEXT(ncid,varrr,'long_name',
     &		len(trim(long_name)),trim(long_name))
	  err = NF_PUT_ATT_TEXT(ncid,varrr,'units',
     &		len(trim(units)),trim(units))
	  err = NF_ENDDEF(ncid)
	  
	
	 err = NF_INQ_VARID(ncid,name,varrr)
         err = NF_PUT_VARA_REAL(ncid,varrr,start,count,fld)

	end do

	 err = NF_INQ_VARID(ncid,'time',varrr)
         err = NF_PUT_VAR1_REAL(ncid,varrr,ntime,time)


 3333	continue

	err = NF_CLOSE(ncid)

	end
	








