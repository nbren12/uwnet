	program glue_movie_raw

	implicit none

	integer i, n, k, nsubdomains_x, nsubdomains_y
	integer nx, ny, startx, starty, x, y, del, thetimes, chunk
        integer(8) nchunks

	character*120 filepath

	character(len=10) titles(9)
	character*4 rankchar

	character*1, allocatable ::  tmpdata(:,:,:), bigdata(:,:,:)
        integer, external ::  bytes_in_rec

!-------------------------------------------------------
	n=COMMAND_ARGUMENT_COUNT()
	if(n.eq.0.) then
	print*,'Command format: glue_movie_raw filepath/case_caseid'
	print*,'for example: glue_movie_raw ./OUT_MOVIES/TOGA_128x128x64_10s'
	stop
	endif
	call getarg(1,filepath)
	print *,'filepath: ',filepath
!--------------------
	open(1,file=trim(filepath)//'.info.movie',form='formatted',status='old')
	read(1,*) nsubdomains_x,nsubdomains_y
	print*, nsubdomains_x,nsubdomains_y
	read(1,*) nx, ny
	print*, nx, ny
	read(1,*) thetimes
	print*, thetimes
        
	nchunks = max(1_8,1_8*nx*nsubdomains_x*ny*nsubdomains_y*thetimes/1000000000_8)
	print*,'nchunks=',nchunks
	chunk = int(thetimes/nchunks)
	print*, 'chunk=',chunk
	allocate(tmpdata(nx,ny,chunk), bigdata(nx*nsubdomains_x,ny*nsubdomains_y,chunk))

	titles(1)='cwp'
	titles(2)='iwp'
	titles(3)='sfcprec'
	titles(4)='qvsfc'
	titles(5)='vsfc'
	titles(6)='cldtop'
	titles(7)='usfc'
	titles(8)='tasfc'
	titles(9)='sst'

	do i=1,9  !numtits

	  print*,titles(i)
	  open(2,file=trim(filepath)//'_'//trim(titles(i))//'.raw',
     &                form='unformatted', 
     &              access='direct',status='new',
     &             recl=(nx*nsubdomains_x)*(ny*nsubdomains_y)*chunk/bytes_in_rec())

	  do k = 1,nchunks

            print*,'k=',k 
	    n=0
	    del=1
	    starty=1
	    do y=0,nsubdomains_y-1
	      startx=1
	      do x=0,nsubdomains_x-1
 
!	         print*,x,y
	         write(rankchar,'(i4)') n
	         open(10+k,file=trim(filepath)//'_'//trim(titles(i))//
     &                       '.raw'//'_'//rankchar(5-del:4), 
     &                            form='unformatted',access='direct',
     &                              status='old',recl=nx*ny*chunk/bytes_in_rec())
	         read(10+k,rec=k) tmpdata
	         close(10+k)

	         bigdata(startx:(startx+nx)-1,starty:(starty+ny)-1,1:chunk)=
     &                                       tmpdata(:,:,1:chunk)

	         n=n+1
	         if (n .eq. 10) then
	           del=del+1
	         endif
	         if (n .eq. 100) then
	           del=del+1
	         endif
	         if (n .eq. 1000) then
	           del=del+1.
	         endif

	         startx=startx+nx

	      enddo
	
	      starty=starty+ny

	    enddo
	
	    write(2,rec=k) bigdata(:,:,1:chunk)
	
	 end do ! k

	 close(2)

	enddo

	end



! determine number of byte in a record in direct access files (can be anything, from 1 to 8):
! determine number of byte in a record in direct access files (can be anything, from 1 to 8):
! can't assume 1 as it is compiler and computer dependent
        integer function bytes_in_rec()
        implicit none
        character*8 str
        integer n, err
        open(11,status ='scratch',access ='direct',recl=1)
        do n = 1,8
         write(11,rec=1,iostat=err) str(1:n)
         if (err.ne.0) exit
         bytes_in_rec = n
        enddo
        close(11,status='delete')
        end

