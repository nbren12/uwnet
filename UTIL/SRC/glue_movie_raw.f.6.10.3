	program glue_movie_raw

	implicit none

	integer i, n, k, nsubdomains_x, nsubdomains_y
	integer nx, ny, startx, starty, x, y, del, thetimes, nchunks,chunk,chunky

	character*120 filepath

	character(len=10) titles(9)
	character*4 rankchar

	character*1, allocatable ::  tmpdata(:,:,:), bigdata(:,:,:)

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
	open(1,file=trim(filepath)//'.info.raw',form='formatted',status='old')
	read(1,*) nsubdomains_x,nsubdomains_y
	print*, nsubdomains_x,nsubdomains_y
	read(1,*) nx, ny
	print*, nx, ny
	read(1,*) thetimes
	print*, thetimes
	nchunks = max(1,nx*nsubdomains_x*ny*nsubdomains_y*thetimes/100000000)
	print*,'nchunks=',nchunks
	chunk = thetimes/nchunks
	allocate(tmpdata(nx,ny,chunk), bigdata(nx*nsubdomains_x,ny*nsubdomains_y,chunk))

	titles(1)='cwp'
	titles(2)='iwp'
	titles(3)='sfcprec'
	titles(4)='qvsfc'
	titles(5)='vsfc'
	titles(6)='cldtop'
	titles(7)='usfc'
	titles(8)='thsfc'
	titles(9)='mse'


	do i=1,9  !numtits

	  print*,titles(i)
	  open(2,file=trim(filepath)//'_'//trim(titles(i))//'.raw',
     &                form='unformatted', 
     &              access='direct',status='new',
     &             recl=(nx*nsubdomains_x)*(ny*nsubdomains_y)*chunk)

	  do k = 1,nchunks

	    chunky = chunk
	    if(k.eq.nchunks) chunky = thetimes-(k-1)*chunk
	    n=0
	    del=1
	    starty=1
	    do y=0,nsubdomains_y-1
	      startx=1
	      do x=0,nsubdomains_x-1
 
	         print*,x,y
	         write(rankchar,'(i4)') n
	         open(1,file=trim(filepath)//'_'//trim(titles(i))//
     &                       '.raw'//'_'//rankchar(5-del:4), 
     &                            form='unformatted',access='direct',
     &                              status='old',recl=nx*ny*chunky)
	         read(1,rec=k) tmpdata
	         close(1)

	         bigdata(startx:(startx+nx)-1,starty:(starty+ny)-1,1:chunky)=
     &                                       tmpdata(:,:,1:chunky)

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
	    print*, titles(i),k
	
	    write(2,rec=k) bigdata(:,:,1:chunky)
	
	 end do ! k

	 close(2)

	enddo

	end

