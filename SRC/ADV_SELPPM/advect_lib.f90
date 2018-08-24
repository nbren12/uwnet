module advect_lib

  ! function & subroutine library for ppm implementation with
  !   selective monotonicity preservation through FCT.
  !   Follows Blossey & Durran (2008, doi:10.1016/j.jcp.2008.01.043).
  !   Equation numbers corresponding to the paper are supplied below.
  !   Flux correction for positivity preservation and fully monotonic
  !   flux correction are also available as options.

  ! Written by Peter Blossey, pblossey@uw.edu.
  ! Modifies an interface written for 5th order ULTIMATE MACHO by Tak Yamaguchi.

  use grid, only: adz, nx, ny, nzm, &
       dimx1_s, dimx2_s, dimy1_s, dimy2_s, &
       dimx1_u, dimx2_u, dimy1_u, dimy2_u, &
       dimx1_v, dimx2_v, dimy1_v, dimy2_v, &
       dimx1_w, dimx2_w, dimy1_w, dimy2_w !, &
       !firststep, nstep, nsave3D
  use advection, only: npad_s, npad_uv
  implicit none

  logical :: adv_doselective = .true. ! apply FCT for monotonicity where lambda>lambda_max
  logical :: adv_dopositive = .true.  ! apply FCT for positivity everywhere
  logical :: adv_domonotonic = .true.  ! apply FCT for monotonicity everywhere
  logical, parameter :: use_uniform_grid_method_for_z = .false.
  logical, parameter :: debug = .false.
  ! brnr use alternate definition to set FCT trigger threshold

  real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm) :: &
       lambda_x, lambda_y
  real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, -1:nzm+2) :: &
       lambda_z
!!$  real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm) :: &
!!$       rhoprime

  real, parameter :: max_lambda = 20. ! 100. !
  real, parameter :: eps2 = 1.e-16
  real, parameter :: eps_hybrid = 1.e-8

  integer :: ipad_limit = 0

  real :: tmpeps
  real, dimension(0:nzm+2) :: irhow
  real, dimension(-1:nzm+2) :: irho, iadz, adz_padded

  logical, save :: is_initialized_SELECTIVE_PPM = .false.
  real, dimension(4,-1:nzm+3) :: ppm_interp_coef

contains
  !------------------------------------------------------------------
  real function compute_gamma_BD08( f_left, f_center, f_right )

    !	Returns value of gamma, according to equation 27 in BD08
    implicit none
    real, intent(in) :: f_left, f_center, f_right 

    compute_gamma_BD08 = (f_center - f_left)**2 &
         + (f_right - f_center)**2

  end function compute_gamma_BD08
  !
  !------------------------------------------------------------------
  real function ppm_face_uniformgrid( f_left2, f_left, f_right, f_right2 )

    !	Returns value of f interpolated to face, equation 3 in BD08
    implicit none
    real, intent(in) :: f_left2, f_left, f_right, f_right2
    real, parameter :: fac1 = 7./12., fac2 = -1./12.

    ppm_face_uniformgrid = fac1*(f_left + f_right) + fac2*(f_left2 + f_right2)

  end function ppm_face_uniformgrid
  !
  !------------------------------------------------------------------
  real function ppm_face_nonuniform( b0, f_left2, f_left, f_right, f_right2 )

    !	Returns value of f interpolated to face, similar to
    !     equation 3 in BD08, except that interpolation is done to
    !     preserve order of accuracy on a non-uniform grid.
    implicit none
    real, intent(in) :: b0(4)
    real, intent(in) :: f_left2, f_left, f_right, f_right2

    ppm_face_nonuniform = &
         b0(1)*f_left2 + b0(2)*f_left + b0(3)*f_right + b0(4)*f_right2

  end function ppm_face_nonuniform
  !
  !------------------------------------------------------------------
  real function ppm_corr_scalar( &
       f_face_left, f_face_center, f_face_right, &
       f_upwind, cr, crpos, crneg )

    !	Returns value of f interpolated to face, equation 3 in BD08
    implicit none
    real, intent(in) :: f_face_left, f_face_center, f_face_right
    real, intent(in) :: f_upwind, cr, crpos, crneg

    real :: f_face_upwind

    f_face_upwind = crpos*f_face_left + crneg*f_face_right

    ! equations 4 and A.1 in BD08.  
    !   Adapted so that no logic is required.
    ppm_corr_scalar = &
         (1.-ABS(cr))*( f_face_center - f_upwind &
         - ABS(cr)*(f_face_center - 2.*f_upwind + f_face_upwind) )

  end function ppm_corr_scalar
  !
  !------------------------------------------------------------------
  real function compute_fct( &
       f_left2, f_left, f_right, f_right2, &
       fx_left, fx_center, fx_right, &
       rhof_td_left, rhof_td_right, rhoprime_left, rhoprime_right )

    !	Returns value of f interpolated to face, equation 3 in BD08
    implicit none
    real, intent(in) :: f_left2, f_left, f_right, f_right2
    real, intent(in) :: fx_left, fx_center, fx_right
    real, intent(in) :: rhof_td_left, rhof_td_right
    real, intent(in) :: rhoprime_left, rhoprime_right

    real :: f_max_right, f_max_left, f_min_right, f_min_left

    if(fx_center.ge.0.) then

      ! correction flux is to right --> we need to limit overshoots
      !  on right and undershoots on left.
      f_max_right = MAX(f_left, f_right, f_right2)
      f_min_left  = MIN(f_left2, f_left, f_right)

      ! equation 16 in BD08, except factor of dt/dx is absorbed into fx's
      compute_fct = MAX(0., MIN(1., &
           ( rhof_td_left - rhoprime_left*f_min_left ) &
           / ( eps2 + fx_center - MIN(0., fx_left) ), &
           (  rhoprime_right*f_max_right - rhof_td_right ) &
           / ( eps2 + fx_center - MIN(0., fx_right) ) ) )

    else

      ! correction flux is to left --> we need to limit overshoots
      !  on left and undershoots on right.
      f_min_right = MIN(f_left, f_right, f_right2)
      f_max_left  = MAX(f_left2, f_left, f_right)

      ! equation A4 in BD08, except factor of dt/dx is absorbed into fx's
      compute_fct = MAX(0., MIN(1., &
           ( rhof_td_right - rhoprime_right*f_min_right ) &
           / ( eps2 - fx_center + MAX(0., fx_right) ), &
           (  rhoprime_left*f_max_left - rhof_td_left ) &
           / ( eps2 - fx_center + MAX(0., fx_left) ) ) )

    end if

  end function compute_fct
  !
  !------------------------------------------------------------------
  subroutine initialize_SELECTIVE_PPM()
    use grid, only: zi
    implicit none

    real :: tmp_zi(-4:nzm+6), zi_local(5)
    real :: b0_coeff(4), b1_coeff(4), b2_coeff(4), b3_coeff(4)
    integer :: k

    do k = 1,nzm+1
      tmp_zi(k) = zi(k)
    end do
    do k = 1,5
      tmp_zi(1-k) = zi(1) - float(k)*(zi(2) - zi(1))
      tmp_zi(nzm+1+k) = zi(nzm+1) + float(k)*(zi(nzm+1) - zi(nzm))
    end do

    do k = -1,nzm+3
      ! define coordinate relative to this face
      zi_local(1:5) = tmp_zi(k-2:k+2) - tmp_zi(k)

      ! get coefficients for interpolating a cubic polynomial
      !  given cell-averaged values over these four cells.
      call compute_ppm_nonuniform_interp_coeff( &
           zi_local, b0_coeff, b1_coeff, b2_coeff, b3_coeff)
    
      ppm_interp_coef(1:4,k) = b0_coeff(1:4)
    end do

  end subroutine initialize_SELECTIVE_PPM
  !
  !------------------------------------------------------------------
  subroutine compute_ppm_nonuniform_interp_coeff(zlocal, b0, b1, b2, b3)
    implicit none

    real, intent(in) :: zlocal(5)
    real, intent(out) :: b0(4), b1(4), b2(4), b3(4)

    integer, parameter :: lwork = 20
    real*8 :: zright(4), zleft(4), work(lwork), A(4,4), B(4,4), C(4,4)
    real*8, parameter :: zero = 0.d0, one = 1.d0
    integer :: ipiv(4), info, i, j

    zright(1:4) = zlocal(2:5)
    zleft(1:4) = zlocal(1:4)

    A(:,1) = zright-zleft
    A(:,2) = (zright**2 - zleft**2)/2.
    A(:,3) = (zright**3 - zleft**3)/3.
    A(:,4) = (zright**4 - zleft**4)/4.

    B(:,:) = 0.
    do i = 1,4
      B(i,i) = zright(i)-zleft(i)
    end do

    !bloss: Note that it is not necessary to use the machine-optimized
    !         LAPACK routines, because this routine in only called at 
    !         the beginning of each simulation, not every time step.
    !       The LAPACK and BLAS utilities have been included in the
    !         new directory SRC/LAPACK_BLAS_UTILS.  If you compile 
    !         against the LAPACK and BLAS libraries, comment this directory
    !         out in the Build script.
    call DGETRF(4,4,A,4,ipiv,info)
    if(info.ne.0) STOP 'error in dgetrf'
    call DGETRI(4,A,4,ipiv,work,lwork,info)
    if(info.ne.0) STOP 'error in dgetri'

    C(:,:) = 0.
    call DGEMM('N','N',4,4,4,one,A,4,B,4,zero,C,4)

    b0(1:4) = C(1,1:4)
    b1(1:4) = C(2,1:4)
    b2(1:4) = C(3,1:4)
    b3(1:4) = C(4,1:4)

  end subroutine compute_ppm_nonuniform_interp_coeff
  !
  !------------------------------------------------------------------
  subroutine x_update_sel_ppm( x1, x2, y1, y2, f, u, rhoprime)

    implicit none

    !	input
    integer, intent(in) :: x1, x2, y1, y2
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: f
    real, dimension(dimx1_u:dimx2_u, dimy1_u:dimy2_u, nzm), intent(in) :: u
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: &
         rhoprime

    !	local
    integer :: i, j, k
    integer :: itmp, imonlimit(x1-2:x2+3,y1:y2)
    real :: gamma1, gamma2, gamma3, tmp_sign
    real, dimension(x1-3:x2+3,y1:y2) :: gamma_f
    real, dimension(x1-2:x2+3,y1:y2) :: crpos, crneg, f_upwind, fx_upwind, f_face
    real, dimension(x1-2:x2+2,y1:y2) :: rhof_td
    real, dimension(x1-1:x2+1,y1:y2) :: r_pos_fct_cell
    real, dimension(x1-1:x2+2,y1:y2) :: fx_corr, r_fct
    real :: olambda_max

    olambda_max = 1./max_lambda

    do k = 1,nzm
      do j = y1,y2 
       do i = x1-2,x2+3
          crpos(i,j) = MAX(0.,SIGN(1.,u(i,j,k)))
          crneg(i,j) = 1. - crpos(i,j)
          f_upwind(i,j) = crpos(i,j)*f(i-1,j,k) + crneg(i,j)*f(i,j,k)
          fx_upwind(i,j) = u(i,j,k)*f_upwind(i,j)
          f_face(i,j) = ppm_face_uniformgrid( &
               f(i-2,j,k), f(i-1,j,k), f(i,j,k), f(i+1,j,k))
        end do

        ! update scalar mass with upwind fluxes.
        i = x1-2
        rhof_td(i,j) = rhoprime(i,j,k)*f(i,j,k) &
             + fx_upwind(i,j) - fx_upwind(i+1,j)

        do i = x1-1,x2+2
          ! compute the PPM correction flux, scaled by dt/dx.
          !   This is the difference between full PPM flux and the upwind flux.
          fx_corr(i,j) = u(i,j,k)*ppm_corr_scalar( &
               f_face(i-1,j), f_face(i,j), f_face(i+1,j), &
               f_upwind(i,j), u(i,j,k)*irho(k), crpos(i,j), crneg(i,j))

          ! update the scalar mass (rho*f) using the upwind flux.
          !   Note that the flux already includes the factor of dt/dx.
          rhof_td(i,j) = rhoprime(i,j,k)*f(i,j,k) &
               + fx_upwind(i,j) - fx_upwind(i+1,j)
        end do

        do i = x1-1,x2+1
          ! update rhoprime, equation 9 (first line) in BD08
          rhoprime(i,j,k) = rhoprime(i,j,k) + u(i,j,k) - u(i+1,j,k)
        end do
      end do

      if(adv_domonotonic) then
        do j = y1,y2
          do i = x1,x2+1
            ! apply monotonic flux correction at this face.
            r_fct(i,j) = compute_fct( &
                 f(i-2,j,k), f(i-1,j,k), f(i,j,k), f(i+1,j,k), &
                 fx_corr(i-1,j), fx_corr(i,j), fx_corr(i+1,j), &
                 rhof_td(i-1,j), rhof_td(i,j), &
                 rhoprime(i-1,j,k), rhoprime(i,j,k) )
          end do
        end do

      else ! not monotonic

        ! Initialize the flux correction normalization factor to one.
        r_fct(:,:) = 1.

        if(adv_doselective) then

          ! compute smoothness indicators
          do j = y1,y2 
            do i = x1-3,x2+3
              gamma_f(i,j) = (f(i-1,j,k)-f(i,j,k))**2 + (f(i,j,k)-f(i+1,j,k))**2
            end do

            ! figure out where limiting needs to be applied (e.g., where lambda > lambda_max)
            imonlimit(x1-2:x1-1,j) = 0
            do i = x1-2,x2+2
              ! modified form for epsilon (in denominator of eqn 28 in BD08)
              !   that uses local values of gamma
              tmpeps = TINY(1.) + 1.e-14*(gamma_f(i-1,j)+gamma_f(i,j)+gamma_f(i+1,j)) !brnr 

              ! itmp=1 if lambda>= lambda_max, 0 otherwise
              itmp = MAX(0, &
                   FLOOR(MIN(10.,MAX(gamma_f(i-1,j),gamma_f(i,j),gamma_f(i+1,j)) &
                     /( tmpeps + MIN(gamma_f(i-1,j),gamma_f(i,j),gamma_f(i+1,j)) ) &
                     *olambda_max) ) )

              ! imonlimit>0 at faces where lambda>=lambda_max in one of the neighboring cells.
              imonlimit(i,j) = imonlimit(i,j) + itmp
              imonlimit(i+1,j) = itmp
            end do

            do i = x1,x2+1
              if(imonlimit(i,j).gt.0) then
                ! apply monotonic flux correction at this face.
                r_fct(i,j) = compute_fct( &
                     f(i-2,j,k), f(i-1,j,k), f(i,j,k), f(i+1,j,k), &
                     fx_corr(i-1,j), fx_corr(i,j), fx_corr(i+1,j), &
                     rhof_td(i-1,j), rhof_td(i,j), &
                     rhoprime(i-1,j,k), rhoprime(i,j,k) )
              end if
            end do
          end do
        end if

        if(adv_dopositive) then
          do j = y1,y2
            do i = x1-1,x2+1
              ! compute the ratio of scalar mass in the cell to the
              !  outflow of scalar mass from the cell during this
              !   timestep, roughly equation 21 in BD08
              r_pos_fct_cell(i,j) = rhof_td(i,j) / (eps2 &
                   + MAX(0.,fx_corr(i+1,j)) - MIN(0.,fx_corr(i,j)) )
            end do
          end do


          do j = y1,y2
            do i = x1,x2+1
              ! renormalize the correction flux at this face if it 
              !   will remove all scalar mass from one of the
              !    neighboring cells.
              tmp_sign = MAX(0.,SIGN(1.,fx_corr(i,j)))
              r_fct(i,j) = MAX(0.,MIN(r_fct(i,j), &
                   tmp_sign*r_pos_fct_cell(i-1,j) &
                   + (1.-tmp_sign)*r_pos_fct_cell(i,j) ) )
            end do
          end do

        end if ! if(adv_dopositive)

      end if ! if(adv_domonotonic)

      ! update the scalar
      do j = y1,y2
        do i = x1,x2
          ! equation 17 in BD08, with division by rhoprime (updated
          !  above) to recover the scalar value after x-advection.
          f(i,j,k) = ( rhof_td(i,j) + r_fct(i,j)*fx_corr(i,j) &
               - r_fct(i+1,j)*fx_corr(i+1,j) ) / rhoprime(i,j,k)
        end do
      end do

    end do

  end subroutine x_update_sel_ppm
  !
  !------------------------------------------------------------------
  subroutine y_update_sel_ppm( x1, x2, y1, y2, f, v, rhoprime)

    implicit none

    !	input
    integer, intent(in) :: x1, x2, y1, y2
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: f
    real, dimension(dimx1_v:dimx2_v, dimy1_v:dimy2_v, nzm), intent(in) :: v
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: &
         rhoprime

    !	local
    integer :: i, j, k
    integer :: itmp, imonlimit(x1:x2,y1-2:y2+3)
    real :: gamma1, gamma2, gamma3, tmp_sign
    real, dimension(x1:x2,y1-3:y2+3) :: gamma_f
    real, dimension(x1:x2,y1-2:y2+3) :: crpos, crneg, f_upwind, gy_upwind, f_face
    real, dimension(x1:x2,y1-2:y2+2) :: rhof_td
    real, dimension(x1:x2,y1-1:y2+1) :: r_pos_fct_cell
    real, dimension(x1:x2,y1-1:y2+2) :: gy_corr, r_fct
    real :: olambda_max

    olambda_max = 1./max_lambda

    do k = 1, nzm
      do j = y1-2,y2+3
        do i = x1,x2
          crpos(i,j) = MAX(0.,SIGN(1.,v(i,j,k)))
          crneg(i,j) = 1. - crpos(i,j)
          f_upwind(i,j) = crpos(i,j)*f(i,j-1,k) + crneg(i,j)*f(i,j,k)
          gy_upwind(i,j) = v(i,j,k)*f_upwind(i,j)
          f_face(i,j) = ppm_face_uniformgrid( &
               f(i,j-2,k), f(i,j-1,k), f(i,j,k), f(i,j+1,k))
        end do
      end do


      ! update scalar mass with upwind fluxes.
      j = y1-2
      do i = x1,x2
        rhof_td(i,j) = rhoprime(i,j,k)*f(i,j,k) &
             + gy_upwind(i,j) - gy_upwind(i,j+1)
      end do

      do j = y1-1,y2+2
        do i = x1,x2
          ! compute the PPM correction flux, scaled by dt/dx.
          !   This is the difference between full PPM flux and the upwind flux.
          gy_corr(i,j) = v(i,j,k)*ppm_corr_scalar( &
               f_face(i,j-1), f_face(i,j), f_face(i,j+1), &
               f_upwind(i,j), v(i,j,k)*irho(k), crpos(i,j), crneg(i,j))

          ! update the scalar mass (rho*f) using the upwind flux.
          !   Note that the flux already includes the factor of dt/dx.
          rhof_td(i,j) = rhoprime(i,j,k)*f(i,j,k) &
               + gy_upwind(i,j) - gy_upwind(i,j+1)
        end do
      end do

      do j = y1-1,y2+1
        do i = x1,x2
          ! update rhoprime, equation 9 (first line) in BD08
          rhoprime(i,j,k) = rhoprime(i,j,k) + v(i,j,k) - v(i,j+1,k)
        end do
      end do

      if(adv_domonotonic) then

        do j = y1,y2+1
          do i = x1,x2
            ! apply monotonic flux correction at this face.
            r_fct(i,j) = compute_fct( &
                 f(i,j-2,k), f(i,j-1,k), f(i,j,k), f(i,j+1,k), &
                 gy_corr(i,j-1), gy_corr(i,j), gy_corr(i,j+1), &
                 rhof_td(i,j-1), rhof_td(i,j), &
                 rhoprime(i,j-1,k), rhoprime(i,j,k) )
          end do
        end do

      else ! not monotonic

        ! Initialize the flux correction normalization factor to one.
        r_fct(:,:) = 1.

        if(adv_doselective) then

          do j = y1-3,y2+3
            do i = x1,x2
              gamma_f(i,j) = (f(i,j-1,k)-f(i,j,k))**2 + (f(i,j,k)-f(i,j+1,k))**2
            end do
          end do

          ! choose locations for limiting based on equation 29 in BD08.
          !   slight modification here: modify at both faces of a cell 
          !   where lambda>lambda_max.
          do i = x1,x2
            imonlimit(i,y1-2) = 0
          end do

          do j = y1-2,y2+2
            do i = x1,x2
              tmpeps = TINY(1.) + 1.e-14*(gamma_f(i,j-1)+gamma_f(i,j)+gamma_f(i,j+1)) !brnr
              itmp = MAX(0, &
                   FLOOR(MIN(10.,MAX(gamma_f(i,j-1),gamma_f(i,j),gamma_f(i,j+1)) &
                   /( tmpeps + MIN(gamma_f(i,j-1),gamma_f(i,j),gamma_f(i,j+1)) ) &
                   *olambda_max) ) )
              imonlimit(i,j) = imonlimit(i,j) + itmp
              imonlimit(i,j+1) = itmp
            end do
          end do

          do j = y1,y2+1
            do i = x1,x2
              if(imonlimit(i,j).gt.0) then
                ! apply monotonic flux correction at this face.
                r_fct(i,j) = compute_fct( &
                     f(i,j-2,k), f(i,j-1,k), f(i,j,k), f(i,j+1,k), &
                     gy_corr(i,j-1), gy_corr(i,j), gy_corr(i,j+1), &
                     rhof_td(i,j-1), rhof_td(i,j), &
                     rhoprime(i,j-1,k), rhoprime(i,j,k) )
              end if
            end do
          end do

        end if

        if(adv_dopositive) then
          do j = y1-1,y2+1
            do i = x1,x2
              ! compute the ratio of scalar mass in the cell to the
              !  outflow of scalar mass from the cell during this
              !   timestep, roughly equation 21 in BD08
              r_pos_fct_cell(i,j) = rhof_td(i,j) / (eps2 &
                   + MAX(0.,gy_corr(i,j+1)) - MIN(0.,gy_corr(i,j)) )
            end do
          end do

          do j = y1,y2+1
            do i = x1,x2
              ! renormalize the correction flux at this face if it 
              !   will remove all scalar mass from one of the
              !    neighboring cells.
              tmp_sign = MAX(0.,SIGN(1.,gy_corr(i,j)))
              r_fct(i,j) = MAX(0.,MIN(r_fct(i,j), &
                   tmp_sign*r_pos_fct_cell(i,j-1) &
                   + (1.-tmp_sign)*r_pos_fct_cell(i,j) ) )
            end do
          end do

        end if ! if(adv_dopositive)


      end if ! if(adv_domonotonic)

      ! update the scalar
      do j = y1,y2
        do i = x1,x2
          ! equation 17 in BD08, with division by rhoprime (updated
          !  above) to recover the scalar value after x-advection.
          f(i,j,k) = ( rhof_td(i,j) + r_fct(i,j)*gy_corr(i,j) &
               - r_fct(i,j+1)*gy_corr(i,j+1) ) / rhoprime(i,j,k)
        end do
      end do

    end do

  end subroutine y_update_sel_ppm
  !
  !------------------------------------------------------------------
  subroutine z_update_sel_ppm( x1, x2, y1, y2, f, w, rhoprime, flux, f_ref)

    implicit none

    !	input
    integer, intent(in) :: x1, x2, y1, y2
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: f
    real, dimension(dimx1_w:dimx2_w, dimy1_w:dimy2_w, nzm+1), intent(in) :: w
    real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: &
         rhoprime
    real, dimension(nzm+1), intent(inout) :: flux ! horizontally-summed scalar mass flux, rho f'w'
    real, dimension(nzm), intent(in) :: f_ref

    !	local
    integer :: i, j, k
    integer, parameter :: npad = 4
    real :: gamma1, gamma2, gamma3, tmp_sign, cr_z
    integer :: itmp, imonlimit(x1:x2,-1:nzm+4)
    real, dimension(x1:x2,1-npad:nzm+npad) :: f_slice
    real, dimension(x1:x2,1-npad:nzm+npad) :: rhoprime_slice
    real, dimension(x1:x2,-2:nzm+3) :: gamma_f
    real, dimension(x1:x2,-1:nzm+3) :: crpos, crneg, f_upwind, hz_upwind, f_face, w_slice
    real, dimension(x1:x2,-1:nzm+2) :: rhof_td
    real, dimension(x1:x2,0:nzm+1) :: r_pos_fct_cell
    real, dimension(x1:x2,0:nzm+2) :: hz_corr, r_fct
    real, dimension(x1:x2) :: winflow
    real :: olambda_max
    real :: tmpvar1,tmpvar2,tmpvar3,tmpvar4,tmpvar5,tmpvar6

    olambda_max = 1./max_lambda


    do j = y1,y2
      
      ! copy f into local array for current slice
      do k = 1,nzm
        do i = x1,x2
          f_slice(i,k) = f(i,j,k)
          rhoprime_slice(i,k) = rhoprime(i,j,k)
          w_slice(i,k) = w(i,j,k)
        end do
      end do
      k = nzm+1
      do i = x1,x2
        w_slice(i,k) = w(i,j,k)
      end do

      ! fill ghost cells below bottom boundary
      winflow(x1:x2) = 0.
      do i = x1,x2
        if(w(i,j,1).gt.0.) winflow(i) = 1.
        w_slice(i,-1:0) = w_slice(i,1)
      end do
      do k = 1,npad
        do i = x1,x2
          f_slice(i,1-k) = f_slice(i,1) &
               + winflow(i)*float(k)*(f_ref(1) - f_ref(2)) & ! extrapolate with ref gradient at inflow
               + (1.-winflow(i))*float(k)*(f_slice(i,1) - f_slice(i,2)) ! extrapolate linearly at outflow
          rhoprime_slice(i,1-k) = rhoprime_slice(i,1)
        end do
      end do
      
      ! fill ghost cells above top boundary
      winflow(x1:x2) = 0.
      do i = x1,x2
        if(w(i,j,nzm+1).lt.0.) winflow(i) = 1.
        w_slice(i,nzm+2:nzm+3) = w_slice(i,nzm+1)
      end do
      do k = 1,npad
        do i = x1,x2
          f_slice(i,nzm+k) = f_slice(i,nzm) &
               + winflow(i)*float(k)*(f_ref(nzm) - f_ref(nzm-1)) & ! extrapolate with ref gradient at inflow
               + (1.-winflow(i))*float(k)*(f_slice(i,nzm) - f_slice(i,nzm-1)) ! extrapolate linearly at outflow
          rhoprime_slice(i,nzm+k) = rhoprime_slice(i,nzm)
        end do
      end do
      
!!$      k = -2
!!$      do i = x1,x2
!!$        gamma_f(i,k) = (f_slice(i,k-1)-f_slice(i,k))**2 + (f_slice(i,k)-f_slice(i,k+1))**2
!!$      end do

      if(use_uniform_grid_method_for_z) then
        do k = -1,nzm+3
          do i = x1,x2
            ! Note that w isn't a full CFL, it is missing the factor of
            !    1/adz(k), which reflects the variable grid stretching
            !    in the vertical and adz = (zi(k+1)-zi(k))/(2*z(1))
            !    where the bottom of the domain is assumed to be the surface.
            crpos(i,k) = MAX(0.,SIGN(1.,w_slice(i,k)))
            crneg(i,k) = 1. - crpos(i,k)
            f_upwind(i,k) = crpos(i,k)*f_slice(i,k-1) + crneg(i,k)*f_slice(i,k)
            hz_upwind(i,k) = w_slice(i,k)*f_upwind(i,k)
            f_face(i,k) = ppm_face_uniformgrid( &
                 f_slice(i,k-2), f_slice(i,k-1), f_slice(i,k), f_slice(i,k+1))
!!$            gamma_f(i,k) = (f_slice(i,k-1)-f_slice(i,k))**2 + (f_slice(i,k)-f_slice(i,k+1))**2
          end do
        end do
      else
        do k = -1,nzm+3
          do i = x1,x2
            ! Note that w isn't a full CFL, it is missing the factor of
            !    1/adz(k), which reflects the variable grid stretching
            !    in the vertical and adz = (zi(k+1)-zi(k))/(2*z(1))
            !    where the bottom of the domain is assumed to be the surface.
            crpos(i,k) = MAX(0.,SIGN(1.,w_slice(i,k)))
            crneg(i,k) = 1. - crpos(i,k)
            f_upwind(i,k) = crpos(i,k)*f_slice(i,k-1) + crneg(i,k)*f_slice(i,k)
            hz_upwind(i,k) = w_slice(i,k)*f_upwind(i,k)
            f_face(i,k) = ppm_face_nonuniform( ppm_interp_coef(1:4,k), &
                 f_slice(i,k-2), f_slice(i,k-1), f_slice(i,k), f_slice(i,k+1))
!!$            gamma_f(i,k) = (f_slice(i,k-1)-f_slice(i,k))**2 + (f_slice(i,k)-f_slice(i,k+1))**2
          end do
        end do
      end if


!!$      do i = x1,x2
!!$        imonlimit(i,0) = 0
!!$      end do
!!$
!!$      do k = 0,nzm+2
!!$        do i = x1,x2
!!$          tmpeps = TINY(1.) + 1.e-14*(gamma_f(i,k-1)+gamma_f(i,k)+gamma_f(i,k+1))
!!$          itmp = MAX(0, &
!!$               FLOOR(MIN(10.,MAX(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) &
!!$               /( tmpeps + MIN(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) ) &
!!$               *olambda_max) ) )
!!$          imonlimit(i,k) = imonlimit(i,k) + itmp
!!$          imonlimit(i,k+1) = itmp
!!$        end do
!!$      end do
            
      !if(debug.AND.(firststep.OR.mod(nstep,nsave3D).eq.0)) then
      !  do k = 1,nzm
      !    do i = x1,x2
      !      ! compute lambda, equation 28 in BD08
      !      lambda_z(i,j,k) = MAX(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) &
      !           /( tmpeps + MIN(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) )
      !    end do
      !  end do
      !end if

      ! update scalar mass with upwind fluxes.
      k = -1
      do i = x1,x2
        ! Scale fluxes by 1/adz since w scaling in adams.f90 did
        !   not include this factor. 
        rhof_td(i,k) = rhoprime_slice(i,k)*f_slice(i,k) &
             + iadz(k)*(hz_upwind(i,k) - hz_upwind(i,k+1))
      end do

!!$      write(*,992) (iadz(k),k=-1,nzm+2)
!!$      write(*,992) (irhow(k),k=0,nzm+2)
!!$      992 format(40f7.3)

      do k = 0,nzm+2
        do i = x1,x2
!!$          ! choose gamma appropriate for this face using equation 29 in BD08
!!$          gamma1 = gamma_f(i,k)
!!$          gamma2 = gamma_f(i,k-1)
!!$          gamma3 = crpos(i,k)*gamma_f(i,k-2) + crneg(i,k)*gamma_f(i,k+1)
!!$
!!$          ! compute lambda, equation 28 in BD08
!!$          lambda_z(i,j,k) = MAX(gamma1,gamma2,gamma3) &
!!$               /(tmpeps + MIN(gamma1,gamma2,gamma3)) 
!!$
          ! compute the PPM correction flux, scaled by dt/dx.
          !   This is the difference between full PPM flux and the upwind flux.
          cr_z = w_slice(i,k)*irhow(k)*(crpos(i,k)*iadz(k-1) + crneg(i,k)*iadz(k))
          hz_corr(i,k) = w_slice(i,k)*ppm_corr_scalar( &
               f_face(i,k-1), f_face(i,k), f_face(i,k+1), &
               f_upwind(i,k), cr_z, crpos(i,k), crneg(i,k))

          ! update the scalar mass (rho*f) using the upwind flux.
          !   Note that the flux already includes the factor of dt/dx.
          ! Scale fluxes by 1/adz since w scaling in adams.f90 did
          !   not include this factor. 
          rhof_td(i,k) = rhoprime_slice(i,k)*f_slice(i,k) &
               + iadz(k)*(hz_upwind(i,k) - hz_upwind(i,k+1))
        end do
      end do

      do k = 0,nzm+1
        do i = x1,x2
          ! update rhoprime, equation 9 (first line) in BD08
          ! Scale fluxes by 1/adz since w scaling in adams.f90 did
          !   not include this factor. 
          rhoprime_slice(i,k) = rhoprime_slice(i,k) + iadz(k)*(w_slice(i,k) - w_slice(i,k+1))
        end do
      end do

      if(adv_domonotonic) then
        do k = 1,nzm+1
          do i = x1,x2
            ! apply monotonic flux correction at this face.
            ! scale rho-terms by adz to account for grid stretching
            !   in vertical.  Results from w not being scaled by 1/adz.
            r_fct(i,k) = compute_fct( &
                 f_slice(i,k-2), f_slice(i,k-1), f_slice(i,k), f_slice(i,k+1), &
                 hz_corr(i,k-1), hz_corr(i,k), hz_corr(i,k+1), &
                 adz_padded(k-1)*rhof_td(i,k-1), adz_padded(k)*rhof_td(i,k), &
                 adz_padded(k-1)*rhoprime_slice(i,k-1), &
                 adz_padded(k)*rhoprime_slice(i,k) )
          end do
        end do
        
      else ! not monotonic

        ! Initialize the flux correction normalization factor to one.
        r_fct(:,:) = 1.


        if(adv_doselective) then
          do k = -2,nzm+3
            do i = x1,x2
              gamma_f(i,k) = (f_slice(i,k-1)-f_slice(i,k))**2 + (f_slice(i,k)-f_slice(i,k+1))**2
            end do
          end do

          do i = x1,x2
            imonlimit(i,0) = 0
          end do

          do k = 0,nzm+2
            do i = x1,x2
              tmpeps = TINY(1.) + 1.e-14*(gamma_f(i,k-1)+gamma_f(i,k)+gamma_f(i,k+1))
              itmp = MAX(0, &
                   FLOOR(MIN(10.,MAX(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) &
                   /( tmpeps + MIN(gamma_f(i,k-1),gamma_f(i,k),gamma_f(i,k+1)) ) &
                   *olambda_max) ) )
              imonlimit(i,k) = imonlimit(i,k) + itmp
              imonlimit(i,k+1) = itmp
            end do
          end do

          do k = 1,nzm+1
            do i = x1,x2
!!$            if(lambda_z(i,j,k).gt.max_lambda) then
              if(imonlimit(i,k).gt.0) then
                ! apply monotonic flux correction at this face.
                ! scale rho-terms by adz to account for grid stretching
                !   in vertical.  Results from w not being scaled by 1/adz.
                r_fct(i,k) = compute_fct( &
                     f_slice(i,k-2), f_slice(i,k-1), f_slice(i,k), f_slice(i,k+1), &
                     hz_corr(i,k-1), hz_corr(i,k), hz_corr(i,k+1), &
                     adz_padded(k-1)*rhof_td(i,k-1), adz_padded(k)*rhof_td(i,k), &
                     adz_padded(k-1)*rhoprime_slice(i,k-1), &
                     adz_padded(k)*rhoprime_slice(i,k) )
              end if
            end do
          end do
        end if

        if(adv_dopositive) then
          do k = 0,nzm+1
            do i = x1,x2
              ! compute the ratio of scalar mass in the cell to the
              !  outflow of scalar mass from the cell during this
              !   timestep, roughly equation 21 in BD08
              ! scale rho*f_td by adz to account for grid stretching
              !   in vertical.
              r_pos_fct_cell(i,k) = adz_padded(k)*rhof_td(i,k) / (eps2 &
                   + MAX(0.,hz_corr(i,k+1)) - MIN(0.,hz_corr(i,k)) )
            end do
          end do


          do k = 1,nzm+1
            do i = x1,x2
              ! renormalize the correction flux at this face if it 
              !   will remove all scalar mass from one of the
              !    neighboring cells.
              tmp_sign = MAX(0.,SIGN(1.,hz_corr(i,k)))
              r_fct(i,k) = MAX(0.,MIN(r_fct(i,k), &
                   tmp_sign*r_pos_fct_cell(i,k-1) &
                   + (1.-tmp_sign)*r_pos_fct_cell(i,k) ) )
            end do
          end do

        end if ! if(adv_dopositive)

      end if ! if(adv_domonotonic)

      ! update the scalar
      do k = 1,nzm
        do i = x1,x2
          ! equation 17 in BD08, with division by rhoprime (updated
          !  above) to recover the scalar value after x-advection.
          ! Scale fluxes by 1/adz since w scaling in adams.f90 did
          !   not include this factor. 
          f(i,j,k) = ( rhof_td(i,k) &
                      + iadz(k)*( r_fct(i,k)*hz_corr(i,k) &
                                - r_fct(i,k+1)*hz_corr(i,k+1) ) &
                       ) / rhoprime_slice(i,k)
          rhoprime(i,j,k) = rhoprime_slice(i,k)
        end do
      end do

      if((j.ge.1).AND.(j.le.ny)) then
        do k = 1,nzm+1
          do i = 1,nx
            flux(k) = flux(k) + hz_upwind(i,k) + r_fct(i,k)*hz_corr(i,k)
          end do
        end do
      end if

    end do

  end subroutine z_update_sel_ppm
  !------------------------------------------------------------------

end module advect_lib
