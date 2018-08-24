c
c
c

	implicit none

c---------------------------------------------------------------
c variables:

	character(120) filename
	character(80) long_name
	character(10) units
	character(8)  name
	character(1)  blank
	character(10) c_min(500), c_max(500)
	character (12) c_z(500),c_p(500),c_dx, c_dy, c_time	
	character(3) nm

	integer(2), allocatable :: byte(:)
	real(4), allocatable ::  fld(:)
	real(4) fmax,fmin
	real(4) dx,dy,z(500),p(500),x(100000),y(100000),time
        real tabs(1:1000),rho(1:1000)

	integer nsubs,nsubsx,nsubsy,nx,ny,nz,nfields
	integer i,j,k,k1,k2,n,i0,j0,nx_gl,ny_gl,count,ifields

c External functions:

	integer iargc,strlen1
	external iargc,strlen1

	real fldmin, fldmax
        logical condition


c---------------------------------------------------------------

c
c Read the file-name from the comman line:
c
	i=COMMAND_ARGUMENT_COUNT()
	if(i.eq.0) then
	  print*,'no input-file name is specified.'
	  print*,'Format: com3D2nc input.com3D'
	  stop
	end if
	call getarg(1,filename)
	print*,filename
c---------------------------------------------------------------
c Read files; merge data from different subdomains;
c save as a netcdf file.
c
	open(1,file=filename,status='old',form='unformatted')
	read(1) long_name(1:32)
	print*,long_name(1:32)
	read(long_name,'(7i4)') nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	read(1) c_time,c_dx,c_dy,(c_z(k),k=1,nz),(c_p(k),k=1,nz)
	print*,c_time,c_dx,c_dy,(c_z(k),k=1,nz),(c_p(k),k=1,nz)
	read(c_dx,'(f12.0)') dx
	read(c_dy,'(f12.0)') dy
	read(c_time,'(f12.5)') time
	do k=1,nz
	  read(c_z(k),'(f12.3)') z(k)
	end do
	do k=1,nz
	  read(c_p(k),'(f12.3)') p(k)
	end do

	print*,'nx,ny,nz,nsubs,nsubsx,nsubsy,nfields:'
	print*,nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
	
	nx_gl=nx*nsubsx
	ny_gl=ny*nsubsy
	print*,'nx_gl=',nx_gl
	print*,'ny_gl=',ny_gl

	do i=1,nx_gl
	 x(i) = dx*(i-1)
	end do
	do j=1,ny_gl
	 y(j) = dy*(j-1)
	end do

	allocate (byte(nx*ny*nz))
	allocate (fld(nx_gl*ny_gl*nz))

c
c The output filename:

	do i=1,115
	  if(filename(i:i+5).eq.'.com3D') then
	    filename(i:i+5)='.bin  '
	    EXIT
	  else if(i.eq.115) then
	    print*,'wrong filename extension!'
	    stop
	  endif
	end do

c
	open(2,file=filename,form='unformatted')
        write(2) time
        write(2) nx_gl
        write(2) ny_gl
        write(2) nz
        write(2) (x(2)-x(1))*0.001
        write(2) (y(2)-y(1))*0.001
        write(2) z(1:nz)*0.001
c	
	
	ifields=0

	do while(ifields.lt.nfields)
	
          ifields=ifields+1
           condition = .true.
!          condition = ifields.ne.4
!          condition = ifields.ne.1.and.ifields.ne.2.and.ifields.ne.4

	  read(1) name,blank,long_name,blank,units,blank
     &	           ,(c_max(k),k=1,nz), (c_min(k),k=1,nz)
	  print*,long_name
	  print*,(c_max(k),k=1,nz)
	  print*,(c_min(k),k=1,nz)
	  do n=0,nsubs-1
	  
     	    read(1) (byte(k),k=1,nx*ny*nz)
	    print*,n,maxval(byte),minval(byte)
	    j0 = n/nsubsx 
	    i0 = n - j0*nsubsx	
	    i0 = i0 * (nx_gl/nsubsx) 
	    j0 = j0 * (ny_gl/nsubsy)  
	    count=0
	    do k=1,nz
	     read(c_min(k),*) fmin
	     read(c_max(k),*) fmax
	     do j=1+j0,ny+j0
	      do i=1+i0,nx+i0
		count=count+1
		fld(i+nx_gl*(j-1)+nx_gl*ny_gl*(k-1))=
     &		   fmin+(byte(count)+32000.)*(fmax-fmin)/64000.
	      end do
	     end do
	    end do

	  end do ! n

	  fldmin=1.e20
	  fldmax=-1.e20
	  do k=1,nz
	   do j=1,ny_gl
	    do i=1,nx_gl
	        fldmin=min(fldmin,fld(i+nx_gl*(j-1)+nx_gl*ny_gl*(k-1)))
	        fldmax=max(fldmax,fld(i+nx_gl*(j-1)+nx_gl*ny_gl*(k-1)))
	    end do
	   end do
	  end do
	  print*,fldmax,fldmin


        if(ifields.eq.nfields-3) then
	  do k=1,nz
           tabs(k) = sum(fld(nx_gl*ny_gl*(k-1)+1:nx_gl*ny_gl*k))/float(nx_gl*ny_gl)
          end do
          print*,'tabsz:',tabs(1:nz)
        endif
        if(condition) then
         do k=1,nz
          write(2) fld(nx_gl*ny_gl*(k-1)+1:nx_gl*ny_gl*k)
         end do
        endif

        end do ! while

        rho(1:nz) = p(1:nz)*100/287./tabs(1:nz)
        write(2) tabs(1:nz)
        write(2) rho(1:nz)


	end
	




	integer function strlen1(str)
	character*(*) str
	strlen1=len(str)
	do i=1,len(str)
	  if(str(i:i).ne.' ') then
	    strlen1=strlen1-i+1
	    return
	  endif 
	end do
        return
	end
