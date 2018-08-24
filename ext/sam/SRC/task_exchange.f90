! Non-blocking receives before blocking sends for efficiency on IBM SP.



subroutine task_exchange(f,dimx1,dimx2,dimy1,dimy2,dimz,i_1, i_2, j_1, j_2, id)
	
! sends and receives the boundary messages	
	
use grid
use advection, only: NADVS
implicit none
	
integer dimx1, dimx2, dimy1, dimy2, dimz
integer i_1, i_2, j_1, j_2
real f(dimx1:dimx2, dimy1:dimy2, dimz)
	
integer bufflen
parameter (bufflen = max(nx,ny)*(3+NADVS)*nz)
real buff_send(bufflen)	! buff to send data
real buff_recv(bufflen,8)	! buff to receive data
integer reqs_in(8), ids(8), count  
integer i_start(8), i_end(8), j_start(8), j_end(8), ranks(8)
logical flag(8)

integer id   ! id of the sent field	
integer i, j, k, n, rf, tag, m, mend
integer i1, i2, j1, j2, mask
	
	i1 = i_1 - 1
	i2 = i_2 - 1
	j1 = j_1 - 1
	j2 = j_2 - 1

	i_start(1) = nx-i1
	i_end  (1) = nx
	j_start(1) = 1
	j_end  (1) = ny

	i_start(2) = 1
	i_end  (2) = 1+i2
	j_start(2) = 1
	j_end  (2) = ny

	i_start(7) = 1
	i_end  (7) = nx
	j_start(7) = ny-j1
	j_end  (7) = ny

	i_start(8) = nx-i1
	i_end  (8) = nx
	j_start(8) = ny-j1
	j_end  (8) = ny

	i_start(3) = nx-i1
	i_end  (3) = nx
	j_start(3) = 1
	j_end  (3) = 1+j2

	i_start(4) = 1
	i_end  (4) = nx
	j_start(4) = 1
	j_end  (4) = 1+j2

	i_start(5) = 1
	i_end  (5) = 1+i2
	j_start(5) = 1
	j_end  (5) = 1+j2

	i_start(6) = 1
	i_end  (6) = 1+i2
	j_start(6) = ny-j1
	j_end  (6) = ny

	ranks(1) = rankee
	ranks(2) = rankww
	ranks(7) = ranknn
	ranks(8) = rankne
	ranks(3) = rankse
	ranks(4) = rankss
	ranks(5) = ranksw
	ranks(6) = ranknw

	mask=id*100000+i1*10000+i2*1000+j1*100+j2*10
	ids(1) = mask+7
	ids(2) = mask+3
	ids(7) = mask+5
	ids(8) = mask+6
	ids(3) = mask+8
	ids(4) = mask+1
	ids(5) = mask+2
	ids(6) = mask+4
!----------------------------------------------------------------------
!  Send/receive buffs to/from neighbors 
!----------------------------------------------------------------------

	if(RUN3D) then
	   mend = 8
	else
	   mend = 2
	endif


! Non-blocking receives first:

	do m = 1,mend

	 if(rank.ne.ranks(m)) then 

          call task_receive_float(buff_recv(1,m),bufflen,reqs_in(m))
	  flag(m) = .false.

	 else

          flag(m) = .true.

         end if

	end do ! m

! Blocking sends:

	do m = 1,mend

	   n=0
	   do k=1,dimz
	     do j=j_start(m),j_end(m)
	       do i=i_start(m),i_end(m)
	         n = n+1
	         buff_send(n) = f(i,j,k)
	       end do
	     end do
           end do

	if(rank.ne.ranks(m)) then 

	  call task_bsend_float(ranks(m),buff_send,n,ids(m))

	 else

          do i=1,n
            buff_recv(i,m) =  buff_send(i)
          end do

	 end if

	end do ! m


! Fill the data from the buffers that have the same rank (were not sent):

	count = 0
	do m = 1,mend
	 if(flag(m)) then
           count = count+1
           tag = ids(m)
           call task_dispatch(buff_recv(1,m),tag)
	 end if 
	end do


! Monitor the progress of receiving; fill the data:


        do while (count .lt. mend)
	  do m = 1,mend
	   if(.not.flag(m)) then
	    call task_test(reqs_in(m),flag(m),rf,tag)
	    if(flag(m)) then 
	      count=count+1
	      call task_dispatch(buff_recv(1,m),tag)
	    endif   
	   endif
	  end do
	end do

 call task_barrier()

end
