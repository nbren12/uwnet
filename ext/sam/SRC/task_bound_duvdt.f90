! Non-blocking receives before blocking sends for efficiency on IBM SP.

subroutine task_bound_duvdt()
          
!  These routine exchanges subdomain overlaping information 
!  for horizontal velocity tendencies

use vars
implicit none
                
integer i,j,k,n,m,tag(2),rf(2),bufflen,nsent
parameter (bufflen = max(nx,ny)*nzm)
real buff_send(bufflen),buff_recv(bufflen,2)
integer reqs_in(2)    


! Any messages to send?
             
nsent=0
if(rank.ne.rankww) nsent=nsent+1
if(rank.ne.rankss) nsent=nsent+1

! Non-blocking receive first:

 do m =  1,nsent
    call task_receive_float(buff_recv(1,m),bufflen,reqs_in(m))
 end do

! Blocking send second:

 n=0 
 do k=1,nzm
    do j=1,ny
     n=n+1
     buff_send(n) = dudt(1,j,k,na)
    end do
 end do 

 if(rank.ne.rankww) then
       call task_bsend_float(rankww, buff_send,n,54)
 else
       do i=1,n
          buff_recv(i,2) = buff_send(i)
       end do
       tag(2) = 54
       rf(2) = rankee
 end if
          
 if(RUN3D) then

  n=0
  do k=1,nzm
    do i=1,nx
      n=n+1
      buff_send(n) = dvdt(i,1,k,na)
    end do
  end do 


  if(rank.ne.rankss) then
     call task_bsend_float(rankss, buff_send, n, 54)
  else
     do i=1,n
       buff_recv(i,2) = buff_send(i)
     end do
     tag(2) = 54
     rf(2) = ranknn
  end if

 endif


! Wait until messages are received::

  call task_waitall(nsent,reqs_in,rf,tag)

! Fill data:

  do m =  1,1+YES3D
    if(tag(m).ne.54) then
       print*,'MPI:Wrong message tag in task_bound_duvdt.'
       print*,'    expected 54  Received:',tag(m)
       call task_abort()
    endif
    if(rf(m).eq.rankee) then
       n=0
       do k=1,nzm
         do j=1,ny
           n = n+1 
           dudt(nxp1,j,k,na) = buff_recv(n,m)
         end do
       end do 
    elseif (rf(m).eq.ranknn) then
       n=0
       do k=1,nzm
         do i=1,nx
           n = n+1 
           dvdt(i,nyp1,k,na) = buff_recv(n,m)
         end do
       end do 
    end if   
  end do      
    
end
