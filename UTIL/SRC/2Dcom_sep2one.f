c

	implicit none

c---------------------------------------------------------------
c variables:

	character(120) filename
	character(80) long_name
	character(10) units
	character(8)  name
	character(1)  blank/" "/
        character(10) c_min, c_max
        character (12)c_dx, c_dy, c_time
        character(4) rankchar

	integer(2), allocatable :: byte(:)
	real dx,dy,time
	integer nsubs,nsubsx,nsubsy,nx,ny,nz,nfields,nstep
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
	  print*,'Format: 2Dcom_sep2one input.2Dcom (without trailing _*)'
	  stop
	end if
	call getarg(1,filename)

c---------------------------------------------------------------
c Read files; merge data from different subdomains;
c save as a netcdf file.
c
	open(1,file=trim(filename)//"_0",status='old',form='unformatted')
	open(3,file=trim(filename),status='new',form='unformatted')

        read(1,end=3333,err=3333) nstep
        read(1) long_name(1:32)
        write(3) nstep
        write(3) long_name(1:32)
        print*,long_name(1:32)
        read(long_name,'(8i4)') nx,ny,nz,nsubs,nsubsx,nsubsy,nfields
        read(1) c_dx
        read(1) c_dy
        read(1) c_time
        print*, c_time,c_dx,c_dy
        write(3) c_dx
        write(3) c_dy
        write(3) c_time

        allocate(byte(nx*ny))

	do ifields=0,nfields-1
	
	  do n=0,nsubs-1

	    if(n.ne.0) then 
              write(rankchar,'(i4)') n
              open(2,file=trim(filename)//"_"//trim(adjustl(rankchar)),
     &               status='old',form='unformatted')
              read(2)
              do i = 1,ifields
                read(2)
              end do
              read(2) (byte(k),k=1,nx*ny)
            else
             read(1) name,blank,long_name,blank,units,blank,c_max,c_min
             write(3) name,blank,long_name,blank,units,blank,c_max,c_min
	     print*,ifields+1,trim(long_name),' ',trim(units), '  ',c_min,c_max
     	     read(1) (byte(k),k=1,nx*ny)
            end if
     	    write(3) (byte(k),k=1,nx*ny)

	  end do ! n

        end do ! ifields

 3333	continue

	end
	



