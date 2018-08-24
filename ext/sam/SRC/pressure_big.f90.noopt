! A version with non-blocking receive and blocking send. Doesn't need 
! EAGER_LIMIT set for communication.


subroutine pressure_big
	
!       parallel pressure-solver for 3D large domains.
!       the FFT is done on vertical slabs, first in x then in y.
!       Each processor gets its own slab.
!       This routine should be used only when the number of processors is larger
!       then the number of level. Otherwise, use pressure_orig
!       (C) 2002, Marat Khairoutdinov

use vars
use params, only: dowallx, dowally, docolumn
implicit none
	

integer, parameter :: nx_s=nx_gl/nsubdomains ! width of the x-slabs
integer, parameter :: ny_s=ny_gl/nsubdomains ! width of the y-slabs

! Slabs:
real fx(0:nx_gl+2, 0:ny_s, nzm) ! slab for x-pass Fourier coefs
real fy(nx_s+1, 0:ny_gl+2, nzm) ! slab for y-pass Fourier coefs

! Message buffers:
real bufx1(0:nx, 0:ny_s, nzm)
real bufx2(0:nx, 0:ny_s, nzm, max(1,nsubdomains_x-1))
real bufy1(nx_s+1, 0:ny_s, nzm)
real bufy2(nx_s+1, 0:ny_s, nzm, max(1,nsubdomains-1))

! FFT stuff:
real work(max((nx_gl+3)*(ny_s+1),(nx_s+1)*(ny_gl+2)))
real trigxi(3*nx_gl/2+1),trigxj(3*ny_gl/2+1) 
integer ifaxj(100),ifaxi(100)

! Tri-diagonal matrix solver coefficients:
real(8) a(nzm),b(nx_s+1,ny_gl+2),c(nzm),e	
real(8) xi,xj,xnx,xny,ddx2,ddy2,pii,fact,eign(nx_s+1,ny_gl+2)
real(8) alfa(nx_s+1,ny_gl+2,nzm),beta(nx_s+1,ny_gl+2,nzm)

integer reqs_in(nsubdomains)
integer i, j, k, id, jd, m, n, it, jt, tag
integer irank, rnk
integer n_in, count
logical flag(nsubdomains)
integer jwall

! Make sure that the grid is suitable for the solver:

if(mod(nx_gl,nsubdomains).ne.0) then
  if(masterproc) print*,'pressure_big: nx_gl/nsubdomains is not round number. STOP'
  call task_abort
endif
if(mod(ny_gl,nsubdomains).ne.0) then
  if(masterproc) print*,'pressure_big: ny_gl/nsubdomains is not round number. STOP'
  call task_abort
endif
if(dowallx) then
  if(masterproc) print*,'pressure_big: dowallx cannot be used with it. STOP'
  call task_abort
end if

!-----------------------------------------------------------------

if(docolumn) return
if(RUN2D) then
  print*,'pressure3D() cannot be called for 2D domains. Quitting...'
  call task_abort()
endif

!-----------------------------------------------------------------
!  Compute the r.h.s. of the Poisson equation for pressure

call press_rhs()

!-----------------------------------------------------------------	 
!   Form the vertical slabs (x-z) of right-hand-sides of Poisson equation 
!   for the FFT - one slab per a processor.

irank = rank-mod(rank,nsubdomains_x)  

n_in = 0
do m = irank, irank+nsubdomains_x-1


  if(m.ne.rank) then

    n_in = n_in + 1
    call task_receive_float(bufx2(0:nx,0:ny_s,1:nzm,n_in),(nx+1)*(ny_s+1)*nzm,reqs_in(n_in))
    flag(n_in) = .false.

  end if

end do ! m

do m = irank, irank+nsubdomains_x-1

  if(m.ne.rank) then

    n = m-irank

    bufx1(0:nx,0:ny_s,1:nzm) = p(0:nx,n*ny_s+0+1-YES3D:n*ny_s+ny_s+1-YES3D,1:nzm)
    call task_bsend_float(m,bufx1(0:nx,0:ny_s,1:nzm),(nx+1)*(ny_s+1)*nzm, 33) 

  endif

end do ! m


! don't send a buffer to itself, just fill directly.

n = rank-irank
call task_rank_to_index(rank,it,jt)
fx(1+it:nx+it,1:ny_s,1:nzm) = p(1:nx,n*ny_s+1:n*ny_s+ny_s,1:nzm)


! Fill slabs when receive buffers are filled:

count = n_in
do while (count .gt. 0)
  do m = 1,n_in
   if(.not.flag(m)) then
	call task_test(reqs_in(m), flag(m), rnk, tag)
        if(flag(m)) then 
	   count=count-1
           call task_rank_to_index(rnk,it,jt)	  
           fx(1+it:nx+it,1:ny_s,1:nzm) = bufx2(1:nx,1:ny_s,1:nzm,m)
	endif   
   endif
  end do
end do

! Perform Fourier transformation n x-direction for a slab:

 call fftfax_crm(nx_gl,ifaxi,trigxi)

 do k=1,nzm
  call fft991_crm(fx(1,YES3D,k),work,trigxi,ifaxi,1,nx_gl+2+1,nx_gl,ny_s,-1)
 end do 

! Synchronize all tasks:

call task_barrier()

!-----------------------------------------------------------------	 
!   Form the vertical slabs (y-z) of Fourier coefs  
!   for the FFT - in y, one slab per a processor.

n_in = 0
do m = 0, nsubdomains-1

  if(m.ne.rank) then

    n_in = n_in + 1
    call task_receive_float(bufy2(1:nx_s+1,0:ny_s,1:nzm,n_in),(nx_s+1)*(ny_s+1)*nzm,reqs_in(n_in))
    flag(n_in) = .false.

  endif

end do ! m

do m = 0, nsubdomains-1

  if(m.ne.rank) then
    bufy1(1:nx_s+1,0:ny_s,1:nzm) = fx(m*nx_s+1:m*nx_s+nx_s+1,0:ny_s,1:nzm)
    call task_bsend_float(m,bufy1(1:nx_s+1,0:ny_s,1:nzm),(nx_s+1)*(ny_s+1)*nzm, 33) 

  else

! don't sent a buffer to itself, just fill directly.

    fy(1:nx_s+1,m*ny_s+1:m*ny_s+ny_s,1:nzm) = &
                                      fx(m*nx_s+1:m*nx_s+nx_s+1,1:ny_s,1:nzm)

  endif

end do ! m


! Fill slabs when receive buffers are full:

count = n_in
do while (count .gt. 0)
  do m = 1,n_in
   if(.not.flag(m)) then
	call task_test(reqs_in(m), flag(m), rnk, tag)
        if(flag(m)) then 
	   count=count-1
           fy(1:nx_s+1,rnk*ny_s+1:rnk*ny_s+ny_s,1:nzm) = &
                                         bufy2(1:nx_s+1,1:ny_s,1:nzm,m)
	endif   
   endif
  end do
end do

! Perform Fourier transform in y-direction for a slab:

 call fftfax_crm(ny_gl,ifaxj,trigxj)

 do k=1,nzm
   if(dowally) then
     call cosft_crm(fy(1,1,k),work,trigxj,ifaxj,nx_s+1,1,ny_gl,nx_s+1,-1)
   else
    call fft991_crm(fy(1,1,k),work,trigxj,ifaxj,nx_s+1,1,ny_gl,nx_s+1,-1)
   end if
 end do 

!-------------------------------------------------
!   Solve the tri-diagonal system for Fourier coeffiecients 
!   in the vertical for each slab:

do k=1,nzm
    a(k)=rhow(k)/(adz(k)*adzw(k)*dz*dz)
    c(k)=rhow(k+1)/(adz(k)*adzw(k+1)*dz*dz)	 
end do 

if(dowally) then
  jwall=2
else
  jwall=0
end if
	

ddx2=dx*dx
ddy2=dy*dy
pii = dacos(-1.d0)
xny=ny_gl 	 
xnx=nx_gl
it=rank*nx_s
jt=0
do j=1,ny_gl+2-jwall
 if(dowally) then
    jd=j+jt-1
    fact = 1.d0
 else
    jd=(j+jt-0.1)/2.
    fact = 2.d0
 end if
 xj=jd
 do i=1,nx_s+1
  id=(i+it-0.1)/2.
  xi=id
  eign(i,j)=(2.d0*cos(2.d0*pii/xnx*xi)-2.d0)/ddx2+ & 
            (2.d0*cos(fact*pii/xny*xj)-2.d0)/ddy2
  if(id+jd.eq.0) then               
     b(i,j)=eign(i,j)*rho(1)-a(1)-c(1)
     alfa(i,j,1)=-c(1)/b(i,j)
     beta(i,j,1)=fy(i,j,1)/b(i,j)
  else
     b(i,j)=eign(i,j)*rho(1)-c(1)
     alfa(i,j,1)=-c(1)/b(i,j)
     beta(i,j,1)=fy(i,j,1)/b(i,j)
  end if
 end do 
end do 

do k=2,nzm-1
 do j=1,ny_gl+2-jwall
  do i=1,nx_s+1	 
    e=eign(i,j)*rho(k)-a(k)-c(k)+a(k)*alfa(i,j,k-1)
    alfa(i,j,k)=-c(k)/e
    beta(i,j,k)=(fy(i,j,k)-a(k)*beta(i,j,k-1))/e
  end do
 end do
end do

do j=1,ny_gl+2-jwall
  do i=1,nx_s+1	 
     fy(i,j,nzm)=(fy(i,j,nzm)-a(nzm)*beta(i,j,nzm-1))/ &
	        (eign(i,j)*rho(nzm)-a(nzm)+a(nzm)*alfa(i,j,nzm-1))
  end do
end do
	  
do k=nzm-1,1,-1
  do j=1,ny_gl+2-jwall
    do i=1,nx_s+1	 	  
       fy(i,j,k)=alfa(i,j,k)*fy(i,j,k+1)+beta(i,j,k)
    end do
  end do  
end do 

! Perform inverse Fourier transf in y-direction for a slab:

 call fftfax_crm(ny_gl,ifaxj,trigxj)

 do k=1,nzm
   if(dowally) then
     call cosft_crm(fy(1,1,k),work,trigxj,ifaxj,nx_s+1,1,ny_gl,nx_s+1,1)
   else
    call fft991_crm(fy(1,1,k),work,trigxj,ifaxj,nx_s+1,1,ny_gl,nx_s+1,1)
   end if
 end do

call task_barrier()

!-----------------------------------------------------------------
!   Form the vertical slabs (x-z) of Fourier coefs
!   for the inverse FFT - in x, one slab per a processor.

n_in = 0
do m = 0, nsubdomains-1

  if(m.ne.rank) then

    n_in = n_in + 1
    call task_receive_float(bufy2(1:nx_s+1,0:ny_s,1:nzm,n_in), &
                            (nx_s+1)*(ny_s+1)*nzm,reqs_in(n_in))
    flag(n_in) = .false.

  endif

end do ! m

fy(:,0,:) = fy(:,ny_gl,:)

do m = 0, nsubdomains-1

  if(m.ne.rank) then

    bufy1(1:nx_s+1,0:ny_s,1:nzm) = fy(1:nx_s+1,m*ny_s+0:m*ny_s+ny_s,1:nzm)
    call task_bsend_float(m,bufy1(1:nx_s+1,0:ny_s,1:nzm),(nx_s+1)*(ny_s+1)*nzm, 33)

  else

! don't sent a buffer to itself, just fill directly.

    fx(m*nx_s+1:m*nx_s+nx_s+1,0:ny_s,1:nzm) = &
                                       fy(1:nx_s+1,ny_s*m+0:m*ny_s+ny_s,1:nzm) 

  endif

end do ! m




! Fill slabs when receive buffers are full:

count = n_in
do while (count .gt. 0)
  do m = 1,n_in
   if(.not.flag(m)) then
        call task_test(reqs_in(m), flag(m), rnk, tag)
        if(flag(m)) then
           count=count-1
           fx(rnk*nx_s+1:rnk*nx_s+nx_s+1,0:ny_s,1:nzm) = &
                                                 bufy2(1:nx_s+1,0:ny_s,1:nzm,m)
        endif
   endif
  end do
end do

fx(nx_gl+2,0,:) = 0.

! Perform inverse Fourier transform n x-direction for a slab:

 call fftfax_crm(nx_gl,ifaxi,trigxi)

 do k=1,nzm
  call fft991_crm(fx(1,0,k),work,trigxi,ifaxi,1,nx_gl+2+1,nx_gl,ny_s+1,1)
 end do

! Synchronize all tasks:

call task_barrier()

!-----------------------------------------------------------------
!  Update the pressure fields in the subdomains
!
n_in = 0
do m = irank, irank+nsubdomains_x-1

  if(m.ne.rank) then

    n_in = n_in + 1
    call task_receive_float(bufx2(0:nx,0:ny_s,1:nzm,n_in),(nx+1)*(ny_s+1)*nzm,reqs_in(n_in))
    flag(n_in) = .false.

  endif

end do ! m

fx(0,:,:) = fx(nx_gl,:,:)

irank = rank-mod(rank,nsubdomains_x)

do m = irank, irank+nsubdomains_x-1

  if(m.ne.rank) then

    call task_rank_to_index(m,it,jt)
    bufx1(0:nx,0:ny_s,1:nzm) = fx(0+it:it+nx,0:ny_s,1:nzm)
    call task_bsend_float(m,bufx1(0:nx,0:ny_s,1:nzm),(nx+1)*(ny_s+1)*nzm, 33)

  endif

end do ! m

! don't sent a buffer to itself, just fill directly.

n = rank-irank
call task_rank_to_index(rank,it,jt)
p(0:nx,n*ny_s+0+1-YES3D:n*ny_s+ny_s+1-YES3D,1:nzm) = fx(0+it:nx+it,0:ny_s,1:nzm)  

! Fill slabs when receive buffers are full:

count = n_in
do while (count .gt. 0)
  do m = 1,n_in
   if(.not.flag(m)) then
        call task_test(reqs_in(m), flag(m), rnk, tag)
        if(flag(m)) then
           count=count-1
           n = rnk-irank
           p(0:nx,n*ny_s+0+1-YES3D:n*ny_s+ny_s+1-YES3D,1:nzm) = bufx2(0:nx,0:ny_s,1:nzm,m)
        endif
   endif
  end do
end do

call task_barrier()

!  Add pressure gradient term to the rhs of the momentum equation:

call press_grad()


end subroutine pressure_big



