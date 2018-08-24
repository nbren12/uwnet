
	subroutine task_start(rank,numtasks)

	include 'mpif.h'	
	integer rank,numtasks,rc,ierr
	call MPI_INIT(ierr)
        if(ierr .ne. 0) then
        	print *,'Error starting MPI program. Terminating.'
        	call MPI_ABORT(MPI_COMM_WORLD, rc, ierr)
        	call MPI_FINALIZE(ierr)
        	stop
     	end if		
        call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierr)
        call MPI_COMM_SIZE(MPI_COMM_WORLD, numtasks, ierr)
	return
	end
	
!----------------------------------------------------------------------
	
	subroutine task_abort()
	
        use grid, only: dompi, nstep,nstop
	include 'mpif.h'	
	integer ierr, rc

	if(dompi) then
!          call MPI_ABORT(MPI_COMM_WORLD, rc, ierr)
!bloss          call MPI_FINALIZE(ierr)  
	endif

!bloss: call task_stop instead
!bloss        call exit(999) ! to avolid resubmission when finished
	call task_stop()

	end
!----------------------------------------------------------------------
	subroutine task_stop()
	
        use grid, only: dompi,nstep,nstop,nelapse
	include 'mpif.h'	
	integer ierr

	if(dompi) then
          call MPI_FINALIZE(ierr)	  
	endif

	if(nstep.ge.nstop) then
           call exit(9) ! avoid resubmission when finished
        elseif(nelapse.eq.0) then
           call exit(0) !bloss: clean exit condition for restart
        else
           call exit(1) !bloss: avoid resubmission if ending in error
        end if

	end
!----------------------------------------------------------------------

        subroutine task_barrier()

        use grid, only: dompi
        implicit none
	include 'mpif.h'	
	integer ierr
        
	if(dompi) then
          call MPI_BARRIER(MPI_COMM_WORLD,ierr)
        end if  

        return
        end

!----------------------------------------------------------------------

        subroutine task_bcast_float4(rank_from,buffer,length)
        implicit none
        include 'mpif.h'

        integer rank_from       ! broadcasting task's rank
        real(4) buffer(*)          ! buffer of data
        integer length          ! buffers' length
        integer ierr

        call MPI_BCAST(buffer,length,MPI_REAL,rank_from,MPI_COMM_WORLD,ierr)

        return
        end

!----------------------------------------------------------------------

        subroutine task_bcast_real8(rank_from,buffer,length)
        implicit none
        include 'mpif.h'

        integer rank_from       ! broadcasting task's rank
        real*8 buffer(*)          ! buffer of data
        integer length          ! buffers' length
        integer ierr

        call MPI_BCAST(buffer,length,mpi_double_precision,rank_from,MPI_COMM_WORLD,ierr)

        return
        end

!----------------------------------------------------------------------

        subroutine task_bcast_integer(rank_from,buffer,length)
        implicit none
        include 'mpif.h'

        integer rank_from       ! broadcasting task's rank
        integer buffer(*)          ! buffer of data
        integer length          ! buffers' length
        integer ierr

        call MPI_BCAST(buffer,length,MPI_INTEGER,rank_from,MPI_COMM_WORLD,ierr)

        return
        end

!----------------------------------------------------------------------

	subroutine task_bsend_float(rank_to,buffer,length,tag)
	implicit none
	include 'mpif.h'	
	
	integer rank_to		! receiving task's rank
	real buffer(*)		! buffer of data
	integer length		! buffers' length
	integer tag		! tag of the message
	integer ierr, real_size

        if(sizeof(buffer(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if
	 call MPI_SEND(buffer,length,real_size,rank_to,tag,MPI_COMM_WORLD,ierr)
	
	return
	end

!----------------------------------------------------------------------

	subroutine task_bsend_float4(rank_to,buffer,length,tag)
	implicit none
	include 'mpif.h'	
	
	integer rank_to		! receiving task's rank
	real(4) buffer(*)		! buffer of data
	integer length		! buffers' length
	integer tag		! tag of the message
	integer ierr

	call MPI_SEND(buffer,length,MPI_REAL,rank_to,tag,MPI_COMM_WORLD,ierr)
	
	return
	end

!----------------------------------------------------------------------

	subroutine task_send_float(rank_to,buffer,length,tag,request)
	implicit none
	include 'mpif.h'	
	
	integer rank_to		! receiving task's rank
	real buffer(*)		! buffer of data
	integer length		! buffers' length
	integer tag		! tag of the message
	integer request		! request id
	integer ierr, real_size

        if(sizeof(buffer(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

        call MPI_ISEND(buffer,length,real_size,rank_to,tag,MPI_COMM_WORLD,request,ierr)

	
	return
	end

!----------------------------------------------------------------------

	subroutine task_send_integer(rank_to,buffer,length,tag,request)

	implicit none
	include 'mpif.h'	
	
	integer rank_to		! receiving task's rank
	integer buffer(*)	! buffer of data
	integer length		! buffers' length
	integer tag		! tag of the message
	integer request
	integer ierr

	call MPI_ISEND(buffer,length,MPI_INTEGER,rank_to,tag, &
					MPI_COMM_WORLD,request,ierr)

	return
	end
	
!----------------------------------------------------------------------

	subroutine task_send_character(rank_to,buffer,length,tag,request)

	implicit none
	include 'mpif.h'	
	
	integer rank_to		! receiving task's rank
	character*1 buffer(*)	! buffer of data
	integer length		! buffers' length
	integer tag		! tag of the message
	integer request
	integer ierr

	call MPI_ISEND(buffer,length,MPI_CHARACTER,rank_to,tag, &
					MPI_COMM_WORLD,request,ierr)

	return
	end
	
!----------------------------------------------------------------------

        subroutine task_breceive_float(buffer,length,rank,tag)

	implicit none
	include 'mpif.h'	
	
	real buffer(*)		! buffer of data
	integer length		! buffers' length
	integer status(MPI_STATUS_SIZE)
	integer rank, tag
	integer ierr, real_size

        if(sizeof(buffer(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

	call MPI_RECV(buffer,length,real_size,MPI_ANY_SOURCE, &
		MPI_ANY_TAG,MPI_COMM_WORLD,status,ierr)
	rank = status(MPI_SOURCE)
	tag = status(MPI_TAG)
	return
	end

!----------------------------------------------------------------------

        subroutine task_receive_float(buffer,length,request)

	implicit none
	include 'mpif.h'	
	
	real buffer(*)		! buffer of data
	integer length		! buffers' length
	integer request
	integer ierr, real_size

        if(sizeof(buffer(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

	call MPI_IRECV(buffer,length,real_size,MPI_ANY_SOURCE, &
		MPI_ANY_TAG,MPI_COMM_WORLD,request,ierr)

	return
	end

!----------------------------------------------------------------------

        subroutine task_receive_float4(buffer,length,request)

	implicit none
	include 'mpif.h'	
	
	real(4) buffer(*)		! buffer of data
	integer length		! buffers' length
	integer request
	integer ierr

	call MPI_IRECV(buffer,length,MPI_REAL,MPI_ANY_SOURCE, &
		MPI_ANY_TAG,MPI_COMM_WORLD,request,ierr)

	return
	end

!----------------------------------------------------------------------

        subroutine task_receive_integer(buffer,length,request)

	implicit none
	include 'mpif.h'	
	
	integer buffer(*)	! buffer of data
	integer length		! buffers' length
	integer request
	integer ierr

	call MPI_IRECV(buffer,length,MPI_INTEGER,MPI_ANY_SOURCE, &
		MPI_ANY_TAG,MPI_COMM_WORLD,request,ierr)

	return
	end

!----------------------------------------------------------------------

        subroutine task_receive_character(buffer,length,request)

	implicit none
	include 'mpif.h'	
	
	character*1 buffer(*)	! buffer of data
	integer length		! buffers' length
	integer request
	integer ierr

	call MPI_IRECV(buffer,length,MPI_CHARACTER,MPI_ANY_SOURCE, &
		MPI_ANY_TAG,MPI_COMM_WORLD,request,ierr)

	return
	end

!----------------------------------------------------------------------
        subroutine task_wait(request,rank,tag)

	implicit none
	include 'mpif.h'
	integer status(MPI_STATUS_SIZE),request
	integer rank, tag
	integer ierr
	call MPI_WAIT(request,status,ierr) 
	rank = status(MPI_SOURCE)
	tag = status(MPI_TAG)

	return
	end

!----------------------------------------------------------------------
        
        subroutine task_waitall(count,reqs,ranks,tags)

	use grid, only: dompi
	implicit none
	include 'mpif.h'
 	integer count,reqs(count)
	integer stats(MPI_STATUS_SIZE,1000),ranks(count),tags(count)
	integer ierr, i
	if(dompi) then
	call MPI_WAITALL(count,reqs,stats,ierr)
        if(count.gt.1000) then
            print*,'task_waitall: count > 1000 !'
	    call task_abort()
	end if
	do i = 1,count
	  ranks(i) = stats(MPI_SOURCE,i)
	  tags(i) = stats(MPI_TAG,i)
	end do
	end if

	return
	end

!----------------------------------------------------------------------
        subroutine task_test(request,flag,rank,tag)

	implicit none
	include 'mpif.h'
	integer status(MPI_STATUS_SIZE),request
	integer rank, tag
	logical flag
	integer ierr
	call MPI_TEST(request,flag,status,ierr)
	if(flag) then 
	  rank = status(MPI_SOURCE)
	  tag = status(MPI_TAG)
	endif

	return
	end

!----------------------------------------------------------------------

        subroutine task_sum_real(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real buffer_in(*)	! buffer of data
	real buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr, real_size

        if(sizeof(buffer_in(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

	call MPI_ALLREDUCE(buffer_in,buffer_out,length, &
                           real_size,MPI_SUM,MPI_COMM_WORLD,ierr)

	return
	end

!----------------------------------------------------------------------

        subroutine task_sum_real8(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real(8) buffer_in(*)	! buffer of data
	real(8) buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out,length, &
                         MPI_REAL8,MPI_SUM, MPI_COMM_WORLD,ierr)

	return
	end
!----------------------------------------------------------------------

        subroutine task_sum_integer(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	integer buffer_in(*)	! buffer of data
	integer buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out,length, &
                        MPI_INTEGER,MPI_SUM, MPI_COMM_WORLD,ierr)

	return
	end
!----------------------------------------------------------------------

        subroutine task_max_real(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real buffer_in(*)	! buffer of data
	real buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr, real_size

        if(sizeof(buffer_in(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                          length,real_size,MPI_MAX,MPI_COMM_WORLD,ierr)

	return
        end
!----------------------------------------------------------------------

        subroutine task_max_real4(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real(4) buffer_in(*)	! buffer of data
	real(4) buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                          length,MPI_REAL,MPI_MAX,MPI_COMM_WORLD,ierr)

	return
	end
!----------------------------------------------------------------------

        subroutine task_max_integer(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	integer buffer_in(*)	! buffer of data
	integer buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                        length,MPI_INTEGER,MPI_MAX,MPI_COMM_WORLD,ierr)

	return
	end

!----------------------------------------------------------------------

        subroutine task_min_real(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real buffer_in(*)	! buffer of data
	real buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr, real_size

        if(sizeof(buffer_in(1)).eq.4) then
         real_size=MPI_REAL
        else
         real_size=MPI_REAL8
        end if

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                            length,real_size,MPI_MIN,MPI_COMM_WORLD,ierr)
	return
	end
!----------------------------------------------------------------------

        subroutine task_min_real4(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	real(4) buffer_in(*)	! buffer of data
	real(4) buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                            length,MPI_REAL,MPI_MIN,MPI_COMM_WORLD,ierr)
	return
	end
!----------------------------------------------------------------------

        subroutine task_min_integer(buffer_in,buffer_out,length)

	implicit none
	include 'mpif.h'	
	
	integer buffer_in(*)	! buffer of data
	integer buffer_out(*)	! buffer of data
	integer length		! buffers' length
	integer ierr

	call MPI_ALLREDUCE(buffer_in,buffer_out, &
                  length,MPI_INTEGER,MPI_MIN,MPI_COMM_WORLD,ierr)

	return
	end
!----------------------------------------------------------------------

