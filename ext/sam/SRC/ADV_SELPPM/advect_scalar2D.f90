subroutine advect_scalar2D( f, u, w, rho, rhow, flux, f_ref, do_poslimit )

  !	Two dimentional 3rd order PPM scheme with selective flux correction 
  !     Optional flux correction for positivity as well.

  use grid
  use advect_lib
  use params, only: dowallx
  implicit none

  !	input & output
  real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: f
  real, dimension(dimx1_u:dimx2_u, dimy1_u:dimy2_u, nzm), intent(inout) :: u
  real, dimension(dimx1_w:dimx2_w, dimy1_w:dimy2_w, nz ), intent(in) :: w
  real, dimension(nzm), intent(in) :: rho, f_ref
  real, dimension(nz), intent(in) :: rhow
  real, dimension(nz), intent(out) :: flux
  logical, intent(in) :: do_poslimit

  !	local
  real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm) :: rhoprime
  integer, parameter :: j = 1
  integer :: dim_ordering, i, k
  
  !--------------------------------------------------------------------------
  if(.NOT.is_initialized_SELECTIVE_PPM) then
    call initialize_SELECTIVE_PPM()
    is_initialized_SELECTIVE_PPM = .true.
  end if

  adv_dopositive = do_poslimit

  if (dowallx) then
    if ( mod(rank,nsubdomains_x) == 0 ) then
      do k = 1, nzm
        do i = dimx1_u, 1
          u(i,j,k) = 0.
        enddo
      enddo
    endif
    if ( mod(rank,nsubdomains_x) == nsubdomains_x-1 ) then
      do k = 1, nzm
        do i = nx+1, dimx2_u
          u(i,j,k) = 0.
        enddo
      enddo
    endif
  endif
  !--------------------------------------------------------------------------

  ! set up rhoprime, a synthetic density that absorbs the non-divergence
  !   of the one-dimensional velocity fields in this dimension-split
  !  advection scheme.  At the end of the time step, rhoprime will
  !  equal the density with an error on the order of the time step
  !  times the divergence of the mass flux (typically rounding
  !  error for SAM).
  do k = 1,nzm
    do i = dimx1_s,dimx2_s
      rhoprime(i,j,k) = rho(k)
    end do
  end do
    
  adz_padded(-1:0) = adz(1)
  iadz(-1:0) = 1./adz(1)
  irho(-1:0) = 1./rho(1)
  do k = 1,nzm
    adz_padded(k) = adz(k)
    irho(k) = 1./rho(k)
    iadz(k) = 1./adz_padded(k)
  end do
  adz_padded(nzm+1:nzm+2) = adz(nzm)
  iadz(nzm+1:nzm+2) = 1./adz(nzm)
  irho(nzm+1:nzm+2) = 1./rho(nzm)

  irhow(0) = 1./rhow(1)
  irhow(1:nz) = 1./rhow(1:nz)
  irhow(nz+1) = 1./rhow(nz)


  flux(:) = 0.
  
  ! shuffle ordering of dimensions in this dimension-split scheme
  !   The following amounts to Strang splitting.
  dim_ordering = mod(nstep,2)
  select case (dim_ordering)
  case(0)
    call x_update_sel_ppm( 1, nx, 1, 1, f, u, rhoprime ) ! x-direction
    call z_update_sel_ppm( 1, nx, 1, 1, f, w, rhoprime, flux, f_ref ) ! z-direction
  case(1)
    call z_update_sel_ppm( 1-npad_s, nx+npad_s, 1, 1, f, w, rhoprime, flux, f_ref ) ! z-direction
    call x_update_sel_ppm( 1, nx, 1, 1, f, u, rhoprime ) ! x-direction
  end select

end subroutine advect_scalar2D
