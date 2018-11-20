module held_suarez_mod
  implicit none
  public :: held_suarez_driver, hs_damp_velocity
  private
  real, parameter :: dtabs_y = 60.0, dtheta_z = 10.0, p0=1000.,&
    kappa=2./7., sig_b = 0.7
contains
  subroutine held_suarez_driver()
    use vars
    use grid
    integer it, jt

    call t_startf('held_squarez')

    call task_rank_to_index(rank,it,jt)
    call held_suarez_t(t(1:nx,1:ny,1:nzm), tabs, pres, gamaz, dy, ny_gl, it, jt,&
      presi(1), presi(nz), dtn)
    call hs_damp_velocity()

    call t_stopf('held_squarez')
  end subroutine held_suarez_driver


  subroutine hs_damp_velocity()
    use vars
    use grid

    call held_suarez_vel(u(1:nx,1:ny,1:nzm), dudt(1:nx,1:ny,1:nzm,na),&
         pres, presi(1), presi(nz))
    call held_suarez_vel(v(1:nx,1:ny,1:nzm), dvdt(1:nx,1:ny,1:nzm,na),&
         pres, presi(1), presi(nz))

  end subroutine hs_damp_velocity

  subroutine held_suarez_vel(u, dudt, pres, pbot, ptop)
    real, intent(in) :: u(:,:,:), pres(:)
    real, intent(inout) ::  dudt(:,:,:)
    real, intent(in) :: pbot, ptop

    ! locals
    real :: k_f, k_v
    integer :: i, j, k

    k_f = 1.0/86400.

    do k=1,size(pres, 1)
      k_v =  k_f * coef(sigma(pres(k), pbot, ptop))
      do j=lbound(dudt,2),ubound(dudt,2)
        do i=lbound(dudt,1),ubound(dudt,1)
          dudt(i,j,k) = dudt(i,j,k) - k_v * u(i,j,k)
        end do 
      end do
    end do

  end subroutine held_suarez_vel

  subroutine held_suarez_t(t, tabs, pres, gamaz, dy, ny_gl, it, jt,&
      pbot, ptop, dt)
    real, intent(inout) :: t(:,:,:), tabs(:,:,:), pres(:), gamaz(:)
    real, intent(in) :: dt, dy, pbot, ptop
    integer, intent(in) :: ny_gl, it, jt
    ! locals
    real :: y, lat, pi 
    real :: k_a, k_s
    real :: k_t
    integer :: nx, ny, nzm
    integer :: i, j, k

    nx = size(tabs, 1)
    ny = size(tabs, 2)
    nzm = size(tabs, 3)

    pi = atan(1.0)*4.0

    k_a = 1./40./86400.
    k_s = 1./4./86400.
    
    do k=1,nzm
      do j=1,ny
        y =  dy*(j+jt-(ny_gl-1.0)/2-1)
        lat = y*2.5e-8*2*pi
        ! print *, i,j,k, y, lat/2/pi*360, pres(k), t_eq(lat, pres(k))
        do i=1,nx
          k_t = k_a + cos(lat)**4&
            *(k_s - k_a) &
            * coef(sigma(pres(k), pbot, ptop))
          t(i,j,k) = t(i,j,k) - dt * k_t &
            * (tabs(i,j,k) - t_eq(lat, pres(k)))
        end do
      end do
    end do
  end subroutine held_suarez_t

  function t_eq(phi, p)
    real t_eq
    real, intent(in) :: phi, p
    ! locals
    real s, c
    s = sin(phi)
    c = cos(phi)

    t_eq = (315 - dtabs_y * s * s - dtheta_z * log(p/p0) * c * c) &
      * (p/p0)**kappa

    t_eq = max(200.0, t_eq)
    
  end function t_eq

  function sigma(p, pbot, ptop)
    real, intent(in) :: p, pbot, ptop
    real :: sigma
    sigma =  (p - ptop)/(pbot-ptop)
  end function sigma

  function coef(sig)
    real :: coef
    real, intent(in) :: sig
    coef = max(0.0, (sig-sig_b)/(1-sig_b))
  end function coef
end
