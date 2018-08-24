
! A version with non-blocking receive and blocking send. Doesn't need 
! EAGER_LIMIT set for communication.

subroutine pressure_big
	
!   parallel pressure-solver for 3D large domains.
!   the FFT is done on vertical slabs, first in x then in y.
!   Each processor gets its own slab.
!   This routine should be used only when the number of processors is larger
!   then the number of level. Otherwise, use pressure_orig
!   (C) 2002, Marat Khairoutdinov
!   Update 2015:
!     This version is based on the version of  pressure_big.f90 code by Don Dazlich
!     It is designed to minimize MPI communication for very large number of precessors.
!     DD: The transpose form x-z slabs to y-z slabs has each process doing 
!         NSUBDOMAINS send/receive pairs. This is replaced with an inverse 
!         transpose back to grid space and a transpose to y-z slabs, each requiring
!         only SQRT(NSUBDOMAINS) send/receive pairs per process. Additionally, the transpose
!         to y-z slabs switches dimensions so that the fft is passed contiguous memory space.


use vars
use params, only: dowallx, dowally, docolumn
implicit none

integer, parameter :: nx_s=nx_gl/nsubdomains ! width of the x-slabs
integer, parameter :: ny_s=ny_gl/nsubdomains ! width of the y-slabs

! Slabs:
real fx(nx_gl, ny_s, nzm) ! slab for x-pass Fourier coefs
real fy(ny_gl, nx_s, nzm) ! slab for y-pass Fourier coefs
real gx(nx_gl+2, ny_s) ! array to perform FFT in x
real gy(ny_gl+2, nx_s) ! array to perform FFT in y
real ff(ny_gl+2, nx_s+1, nzm)

! Message buffers:
real bufx1(nx, ny_s, nzm)
real bufx2(nx, ny_s, nzm, max(1,nsubdomains_x))
real bufy1(nx_s, ny, nzm)
real bufy2(nx_s, ny, nzm, max(1,nsubdomains_y))

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

! for wrapping p
real buff_ew1(ny,nzm), buff_ns1(nx,nzm)
real buff_ew2(ny,nzm), buff_ns2(nx,nzm)
integer rf, tagrf
logical waitflag

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

!==========================================================================
!  Compute the r.h.s. of the Poisson equation for pressure

call press_rhs()

! variable p will also be used as grid decomposition placeholder between transposes
!   for the fourier coefficients

!==========================================================================
!   Form the vertical slabs (x-z) of right-hand-sides of Poisson equation 
!   for the FFT - one slab per a processor.

   call transpose_x(fx)

!==========================================================================
! Perform Fourier transformation n x-direction for a slab:

 call fftfax_crm(nx_gl,ifaxi,trigxi)

 do k=1,nzm
  gx(1:nx_gl,1:ny_s) = fx(1:nx_gl,1:ny_s,k)
  call fft991_crm(gx,work,trigxi,ifaxi,1,nx_gl+2,nx_gl,ny_s,-1)
  fx(1,1:ny_s,k) = gx(1,1:ny_s)
  fx(2:nx_gl,1:ny_s,k) = gx(3:nx_gl+1,1:ny_s)
 end do

call task_barrier()

!==========================================================================
!   Form the vertical slabs (y-z) of Fourier coefs  
!   for the FFT - in y, one slab per a processor.

call transpose_x_inv(fx)
call transpose_y(fy)

call fftfax_crm(ny_gl,ifaxj,trigxj)

ff = 0.
do k=1,nzm
   gy(1:ny_gl,1:nx_s) = fy(1:ny_gl,1:nx_s,k)
   if(dowally) then
    call cosft_crm(gy,work,trigxj,ifaxj,1,ny_gl+2,ny_gl,nx_s,-1)
   else
    call fft991_crm(gy,work,trigxj,ifaxj,1,ny_gl+2,ny_gl,nx_s,-1)
   end if
   ff(1:ny_gl+2,2:nx_s+1,k) = gy(1:ny_gl+2,1:nx_s)
end do 
if(rank.eq.0) then
  ff(:,1,:)=ff(:,2,:)
  ff(:,2,:)=0.
end if

!==========================================================================
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
     beta(i,j,1)=ff(j,i,1)/b(i,j)
  else
     b(i,j)=eign(i,j)*rho(1)-c(1)
     alfa(i,j,1)=-c(1)/b(i,j)
     beta(i,j,1)=ff(j,i,1)/b(i,j)
  end if
 end do
end do

do k=2,nzm-1
 do j=1,ny_gl+2-jwall
  do i=1,nx_s+1
    e=eign(i,j)*rho(k)-a(k)-c(k)+a(k)*alfa(i,j,k-1)
    alfa(i,j,k)=-c(k)/e
    beta(i,j,k)=(ff(j,i,k)-a(k)*beta(i,j,k-1))/e
  end do
 end do
end do

do j=1,ny_gl+2-jwall
  do i=1,nx_s+1
     ff(j,i,nzm)=(ff(j,i,nzm)-a(nzm)*beta(i,j,nzm-1))/ &
                (eign(i,j)*rho(nzm)-a(nzm)+a(nzm)*alfa(i,j,nzm-1))
  end do
end do

do k=nzm-1,1,-1
  do j=1,ny_gl+2-jwall
    do i=1,nx_s+1
       ff(j,i,k)=alfa(i,j,k)*ff(j,i,k+1)+beta(i,j,k)
    end do
  end do
end do

if(rank.eq.0) then
  ff(:,2,:)=ff(:,1,:)
  ff(:,1,:)=0.
end if

!==========================================================================
! Perform inverse Fourier transf in y-direction for a slab:

call fftfax_crm(ny_gl,ifaxj,trigxj)

 do k=1,nzm
   gy(1:ny_gl+2,1:nx_s) = ff(1:ny_gl+2,2:nx_s+1,k)
   if(dowally) then
     call cosft_crm(gy,work,trigxj,ifaxj,1,ny_gl+2,ny_gl,nx_s,1)
   else
     call fft991_crm(gy,work,trigxj,ifaxj,1,ny_gl+2,ny_gl,nx_s,1)
   end if
   fy(1:ny_gl,1:nx_s,k) = gy(1:ny_gl,1:nx_s)
 end do

call task_barrier()

!==========================================================================
!   Form the vertical slabs (x-z) of Fourier coefs
!   for the inverse FFT - in x, one slab per a processor.


call transpose_y_inv(fy)
call transpose_x(fx)

! Perform inverse Fourier transform n x-direction for a slab:

 call fftfax_crm(nx_gl,ifaxi,trigxi)

 do k=1,nzm
  gx(1,1:ny_s) = fx(1,1:ny_s,k)
  gx(2,:) = 0.
  gx(3:nx_gl+1,1:ny_s) = fx(2:nx_gl,1:ny_s,k)
  gx(nx_gl+2,:) = 0.
  call fft991_crm(gx,work,trigxi,ifaxi,1,nx_gl+2,nx_gl,ny_s,1)
  fx(1:nx_gl,1:ny_s,k) = gx(1:nx_gl,1:ny_s)
 end do

call task_barrier()

call transpose_x_inv(fx)

!==========================================================================
!  Update the pressure fields in the subdomains
!

! when we cut back on the ffts wrap p here - look to sib for the model
!DD temporary measure for dompi
  if(dompi) then
!DD custom build a wrap that sends from the north and east edges, and receives at the
!DD    south and west edges

      if(rank==rankee) then
        p(0,1:ny,1:nzm) = p(nx,1:ny,1:nzm)
      else
        call task_receive_float(buff_ew2(:,:),ny*nzm,reqs_in(1))
          buff_ew1(1:ny,1:nzm) = p(nx,1:ny,1:nzm)
          call task_bsend_float(rankee,buff_ew1(:,:),ny*nzm,1)
          waitflag = .false.
        do while (.not.waitflag)
            call task_test(reqs_in(1),waitflag,rf,tagrf)
        end do
        call task_barrier()
        p(0,1:ny,1:nzm) = buff_ew2(1:ny,1:nzm)
      endif

      if(rank==ranknn) then
         p(:,0,1:nzm) = p(:,ny,1:nzm)
      else
             call task_receive_float(buff_ns2(:,:),nx*nzm,reqs_in(1))
             buff_ns1(1:nx,1:nzm) = p(1:nx,ny,1:nzm)
             call task_bsend_float(ranknn,buff_ns1(:,:),nx*nzm,1)
             waitflag = .false.
         do while (.not.waitflag)
               call task_test(reqs_in(1),waitflag,rf,tagrf)
             end do
         call task_barrier()
         p(1:nx,0,1:nzm) = buff_ns2(1:nx,1:nzm)
      endif

  else
    p(0,:,1:nzm) = p(nx,:,1:nzm)
    p(:,0,1:nzm) = p(:,ny,1:nzm)
  endif
!DD end ugly wrap code.

!==========================================================================
!  Add pressure gradient term to the rhs of the momentum equation:

call press_grad()

!==========================================================================
!==========================================================================
!==========================================================================

contains

!==========================================================================
   subroutine transpose_x(f)

! transpose from blocks to x-z slabs

      REAL, INTENT(OUT) :: f(nx_gl, ny_s, nzm)
      
      irank = rank-mod(rank,nsubdomains_x)  

      n_in = 0
      do m = irank, irank+nsubdomains_x-1

        if(m.ne.rank) then

          n_in = n_in + 1
          call task_receive_float(bufx2(:,:,:,n_in),nx*ny_s*nzm,reqs_in(n_in))
          flag(n_in) = .false.

        end if

      end do ! m

      do m = irank, irank+nsubdomains_x-1

        if(m.ne.rank) then

          n = m-irank

          bufx1(:,:,:) = p(1:nx,n*ny_s+1:n*ny_s+ny_s,1:nzm)
          call task_bsend_float(m,bufx1(:,:,:),nx*ny_s*nzm, 33) 

        endif

      end do ! m


! don't sent a buffer to itself, just fill directly.

      n = rank-irank
      call task_rank_to_index(rank,it,jt)
      f(1+it:nx+it,1:ny_s,1:nzm) = p(1:nx,n*ny_s+1:n*ny_s+ny_s,1:nzm)


      ! Fill slabs when receive buffers are full:

      count = n_in
      do while (count .gt. 0)
        do m = 1,n_in
         if(.not.flag(m)) then
      	    call task_test(reqs_in(m), flag(m), rnk, tag)
              if(flag(m)) then 
            	 count=count-1
                 call task_rank_to_index(rnk,it,jt)	  
                 f(1+it:nx+it,1:ny_s,1:nzm) = bufx2(1:nx,1:ny_s,1:nzm,m)
              endif   
          endif
         end do
      end do
      call task_barrier()
   end subroutine transpose_x
   
!==========================================================================
   subroutine transpose_x_inv(f)

! transpose from x-z slabs to blocks

      REAL, INTENT(IN) :: f(nx_gl, ny_s, nzm)
      
      irank = rank-mod(rank,nsubdomains_x)
      n_in = 0
      do m = irank, irank+nsubdomains_x-1

        if(m.ne.rank) then

          n_in = n_in + 1
          call task_receive_float(bufx2(:,:,:,n_in),nx*ny_s*nzm,reqs_in(n_in))
          flag(n_in) = .false.

        endif

      end do ! m

      do m = irank, irank+nsubdomains_x-1

        if(m.ne.rank) then

          call task_rank_to_index(m,it,jt)
          bufx1(:,:,:) = f(1+it:it+nx,1:ny_s,1:nzm)
          call task_bsend_float(m,bufx1(:,:,:),nx*ny_s*nzm, 33)

        endif

      end do ! m

! don't sent a buffer to itself, just fill directly.

      n = rank-irank
      call task_rank_to_index(rank,it,jt)
      p(1:nx,n*ny_s+1:n*ny_s+ny_s,1:nzm) = f(1+it:nx+it,1:ny_s,1:nzm)  

! Fill slabs when receive buffers are full:

      count = n_in
      do while (count .gt. 0)
        do m = 1,n_in
         if(.not.flag(m)) then
              call task_test(reqs_in(m), flag(m), rnk, tag)
              if(flag(m)) then
                 count=count-1
                 n = rnk-irank
                 p(1:nx,n*ny_s+1:n*ny_s+ny_s,1:nzm) = bufx2(1:nx,1:ny_s,1:nzm,m)
              endif
         endif
        end do
      end do

      call task_barrier()
   end subroutine transpose_x_inv

!==========================================================================
   subroutine transpose_y(f)

! transpose from blocks to y-z slabs

      REAL, INTENT(OUT) :: f(ny_gl, nx_s, nzm)
      
      irank = rank / nsubdomains_y  

      n_in = 0
      do m = irank, nsubdomains-1, nsubdomains_x

        if(m.ne.rank) then

          n_in = n_in + 1
          call task_receive_float(bufy2(:,:,:,n_in),ny*nx_s*nzm,reqs_in(n_in))
          flag(n_in) = .false.
          
        else
! don't sent a buffer to itself, just fill directly.

          n = mod(rank,nsubdomains_y)
          call task_rank_to_index(rank,it,jt)
          do i = 1,nx_s	  
            f(1+jt:ny+jt,i,1:nzm) = p(n*nx_s+i,1:ny,1:nzm)
          enddo

        end if

      end do ! m

      irank = nsubdomains_y*mod(rank,nsubdomains_x)
      do m = irank, irank+nsubdomains_y-1

        if(m.ne.rank) then

          n = m-irank

          bufy1(:,:,:) = p(n*nx_s+1:n*nx_s+nx_s,1:ny,1:nzm)
          call task_bsend_float(m,bufy1(:,:,:),ny*nx_s*nzm, 33) 

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
                 call task_rank_to_index(rnk,it,jt)
                 do i = 1,nx_s	  
                   f(1+jt:ny+jt,i,1:nzm) = bufy2(i,1:ny,1:nzm,m)
                 enddo
              endif   
          endif
         end do
      end do

      call task_barrier()
   end subroutine transpose_y
   
!==========================================================================
   subroutine transpose_y_inv(f)

! transpose from y-z slabs to blocks

      REAL, INTENT(IN) :: f(ny_gl, nx_s, nzm)
      
      n_in = 0
      irank = nsubdomains_y*mod(rank,nsubdomains_x)
      do m = irank, irank+nsubdomains_y-1

        if(m.ne.rank) then

          n_in = n_in + 1
          call task_receive_float(bufy2(:,:,:,n_in),ny*nx_s*nzm,reqs_in(n_in))
          flag(n_in) = .false.
          
        else

! don't sent a buffer to itself, just fill directly.

          n = rank-irank
          call task_rank_to_index(rank,it,jt)
          do i = 1,nx_s
            p(n*nx_s+i,1:ny,1:nzm) = f(1+jt:ny+jt,i,1:nzm) 
          enddo 

        endif

      end do ! m

      irank = rank / nsubdomains_y  
      do m = irank, nsubdomains-1, nsubdomains_x

        if(m.ne.rank) then

          call task_rank_to_index(m,it,jt)
          do i = 1,nx_s
            bufy1(i,:,:) = f(1+jt:jt+ny,i,1:nzm)
          enddo
          call task_bsend_float(m,bufy1(:,:,:),ny*nx_s*nzm, 33)

        endif

      end do ! m

! Fill slabs when receive buffers are full:

      irank = nsubdomains_y*mod(rank,nsubdomains_x)
      count = n_in
      do while (count .gt. 0)
        do m = 1,n_in
         if(.not.flag(m)) then
              call task_test(reqs_in(m), flag(m), rnk, tag)
              if(flag(m)) then
                 count=count-1
                 n = rnk-irank
                 p(n*nx_s+1:n*nx_s+nx_s,1:ny,1:nzm) = bufy2(1:nx_s,1:ny,1:nzm,m)
              endif
         endif
        end do
      end do

      call task_barrier()
   end subroutine transpose_y_inv

end subroutine pressure_big
