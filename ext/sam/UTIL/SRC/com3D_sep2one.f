	implicit none

c---------------------------------------------------------------
c variables:

	character(120) filename
	character(80) long_name
	character(10) units
	character(8)  name
	character(1)  blank/" "/
        character(10) c_min(500), c_max(500)
        character(12) c_z(500), c_p(500), c_dx, c_dy, c_time
        character(4) rankchar

	integer(2), allocatable :: byte(:)
	real dx,dy,z(500),p(500),time
	integer nsubs,nsubsx,nsubsy,nx,ny,nz,nfields
	integer i,k,n,nx_gl,ny_gl,ifields

c External functions:

	integer, external :: iargc
	real fldmin, fldmax

c---------------------------------------------------------------
c---------------------------------------------------------------
c
c Read the file-name from the comman line:
c
	n=COMMAND_ARGUMENT_COUNT()
	if(n.eq.0) then
	  print*,'no input-file name is specified.'
	  print*,'Format: com3D_sep2one input.com3D (without trailing _*)'
	  stop
	end if
	call getarg(1,filename)

c---------------------------------------------------------------
c Read files; merge data from different subdomains;
c save as a netcdf file.
c
	open(1,file=trim(filename)//"_0",status='old',form='unformatted')
	open(3,file=trim(filename),status='new',form='unformatted')

        read(1,end=3333,err=3333) long_name(1:32)
        write(3) long_name(1:32)
        print*,long_name(1:32)
        read(long_name,'(8i4)') nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
        read(1)  c_time,c_dx,c_dy,(c_z(k),k=1,nz),(c_p(k),k=1,nz)
        print*,  c_time,c_dx,c_dy,(c_z(k),k=1,nz),(c_p(k),k=1,nz)
        write(3) c_time,c_dx,c_dy,(c_z(k),k=1,nz),(c_p(k),k=1,nz)

        allocate(byte(nx*ny*nz))

	do ifields=0,nfields-1
	
	  do n=0,nsubs-1

	    if(n.ne.0) then 
              write(rankchar,'(i4)') n
              open(2,file=trim(filename)//"_"//trim(adjustl(rankchar)),
     &               status='old',form='unformatted')
              do i = 1,ifields
                read(2)
              end do
              read(2) (byte(k),k=1,nx*ny*nz)
            else
             read(1) name,blank,long_name,blank,units,blank
     &                ,(c_max(k),k=1,nz), (c_min(k),k=1,nz)
             write(3) name,blank,long_name,blank,units,blank
     &                ,(c_max(k),k=1,nz), (c_min(k),k=1,nz)
	     print*,ifields+1,trim(long_name),' ',trim(units)
     	     read(1) (byte(k),k=1,nx*ny*nz)
            end if
     	    write(3) (byte(k),k=1,nx*ny*nz)

	  end do ! n

        end do ! ifields

 3333	continue

	end
	



