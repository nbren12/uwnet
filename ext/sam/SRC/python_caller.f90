module python_caller
  ! only works with CAM radiation scheme
  use callpy_mod
  implicit none

  public :: initialize_from_target, initialize_python_caller, state_to_python, push_state


  character(len=256) :: module_name, function_name

  real :: last_time_called
  real, allocatable, dimension(:,:,:) :: sl_last, qt_last, FQTNN, FSLINN

  integer ntop
contains

  subroutine initialize_python_caller()
    use vars, only: nx, ny, nz, nzm, rho, pres, presi, caseid, case, adz, dz, t, time
    use microphysics, only: micro_field

    ! locals
    real(4) :: tmp(1:nzm), tmpw(1:nz)

    allocate(sl_last(nx, ny, nzm))
    allocate(qt_last(nx, ny, nzm))
    allocate(FQTNN(nx, ny, nzm))
    allocate(FSLINN(nx, ny, nzm))

    sl_last = t(1:nx, 1:ny, 1:nzm)
    qt_last = micro_field(1:nx, 1:ny, 1:nzm, 1)
    last_time_called = time
    print *, 'python_caller.f90::initialize_python_caller: storing sl and qt to sl_last and qt_last at time', time

    tmp = rho * adz * dz
    call set_state_1d("layer_mass", tmp)

    tmp = pres
    call set_state_1d("p", tmp)

    tmpw = presi
    call set_state_1d("pi", tmpw)

    call set_state_char("caseid", caseid)
    call set_state_char("case", case)
  end subroutine initialize_python_caller

  subroutine initialize_from_target
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    ! use rad, only: solinxy
    ! use grid, only: day, caseid, case
    use microphysics, only: micro_field
    real(4) :: tmp(1:nx, 1:ny, 1:nzm)
    ! locals

    print *, 'Initializing state from target'

    ! call check(compute_next_state())

    ! read in state from python
    call get_state("sl", tmp, nx * ny * nzm)
    t(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("qt", tmp, nx * ny * nzm)
    micro_field(1:nx, 1:ny, 1:nzm, 1) = tmp

    call get_state("U", tmp, nx*ny*nzm)
    u(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("V", tmp, nx*ny*nzm)
    v(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("W", tmp, nx*ny*nzm)
    w(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("Prec", tmp, nx*ny)
    prec_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("LHF", tmp, nx * ny)
    lhf_xy(1:nx, 1:ny) = tmp(:,:,1)

    last_time_called = time
  end subroutine initialize_from_target

  subroutine push_state()
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep, day, caseid, case, nz
    use microphysics, only: micro_field
    use rad, only: solinxy
    ! compute derivatives
    real, dimension(nx,ny,nzm) :: fqt, fsl
    real(4) :: tmp(1:nx, 1:ny, 1:nzm), tmp_vert(1:nzm), tmp_vertw(1:nz), tmp_scalar
    real :: dt

    ntop = nzm
    dt = time - last_time_called
    print *, 'python_caller.f90::push_state computing FSL and FQT with dt=', dt
    fsl = (t(1:nx, 1:ny, 1:nzm) - sl_last)/dt
    fqt = (micro_field(1:nx, 1:ny, 1:nzm, 1) - qt_last)/dt

    ! ! Send the arrays to a global array in the python module
    ! ! This is a much easier architecture than trying to write functions
    ! ! which take all of these arguments
    tmp = t(1:nx,1:ny,1:nzm)
    call set_state("SLI", tmp)

    tmp = micro_field(1:nx,1:ny,1:nzm, 1) * 1.e3
    call set_state("QT", tmp)

    tmp =  tabs(1:nx,1:ny,1:nzm)
    call set_state("TABS", tmp)
    tmp =  w(1:nx,1:ny,1:nzm)
    call set_state("W", tmp)

    tmp =  fqt
    call set_state("FQT", tmp)

    tmp =  fsl
    call set_state("FSLI", tmp)

    tmp =  u(1:nx,1:ny,1:nzm)
    call set_state("U", tmp)

    tmp =  v(1:nx,1:ny,1:nzm)
    call set_state("V", tmp)

    ! call set_state2d("lat", latitude)
    ! call set_state2d("lon", longitude)

    ! for some reason set_state2d has some extremee side ffects
    ! that can cause the model to crash
    tmp(:,:,1) = sstxy(1:nx, 1:ny) + t00
    call set_state2d("SST", tmp(:,:,1))


    tmp(:,:,1) = solinxy(1:nx, 1:ny)
    call set_state2d("SOLIN", tmp(:,:,1))


    tmp_vert = rho * adz * dz
    call set_state_1d("layer_mass", tmp_vert)

    tmp_vert = pres
    call set_state_1d("p", tmp_vert)

    tmp_vertw = presi
    call set_state_1d("pi", tmp_vertw)


    tmp_scalar = dt
    call set_state_scalar("p0", tmp_scalar)
    call set_state_scalar("dt", tmp_scalar)

    tmp_scalar = time
    call set_state_scalar("time", tmp_scalar)

    tmp_scalar = day
    call set_state_scalar("day", tmp_scalar)

    tmp_scalar = real(nstep)
    call set_state_scalar("nstep", tmp_scalar)
    call set_state_char("caseid", caseid)
    call set_state_char("case", case)

  end subroutine push_state

  subroutine get_state_from_python()
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    use microphysics, only: micro_field, qn, qp, micro_diagnose
    ! locals
    real(4) :: tmp(1:nx, 1:ny, 1:nzm)
    ! read in state from python
    print *, 'python_caller.f90::state_to_python retreiving state from python module'
    call get_state("SLI", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    t(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("QT", tmp, nx * ny * nzm)
    tmp = tmp / 1.e3
    ! tmp(:,:,ntop:nzm) = micro_field(1:nx,1:ny, ntop:nzm, 1)
    micro_field(1:nx, 1:ny, 1:nzm, 1) = tmp


    call get_state("FQTNN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    fqtnn(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("FSLINN", tmp, nx * ny * nzm)
    ! tmp(:,:,ntop:nzm) = t(1:nx,1:ny, ntop:nzm)
    fslinn(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("QN", tmp, nx*ny*nzm)
    tmp = tmp / 1.e3
    qn(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("QP", tmp, nx*ny*nzm)
    tmp = tmp / 1.e3
    qp(1:nx, 1:ny, 1:nzm) = tmp

    call get_state("Prec", tmp, nx*ny)
    prec_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("LHF", tmp, nx * ny)
    lhf_xy(1:nx, 1:ny) = tmp(:,:,1)

    call get_state("SHF", tmp, nx * ny)
    shf_xy(1:nx, 1:ny) = tmp(:,:,1)

    ! call get_state("U", tmp, nx*ny*nzm)
    ! u(1:nx, 1:ny, 1:nzm) = tmp

    ! call get_state("V", tmp, nx*ny*nzm)
    ! v(1:nx, 1:ny, 1:nzm) = tmp

    ! call get_state("W", tmp, nx*ny*nzm)
    ! w(1:nx, 1:ny, 1:nzm) = tmp

    ! diagnose the phase partitioning of water
    ! this is needed to compute the correct temperature
    ! diagnose is called in main.f90, so we don't need to call that here
    ! ...I hate all these global variables
    call micro_diagnose()

  end subroutine get_state_from_python

  subroutine state_to_python(dtn)
    use vars, only: t, u, v,w, tabs,&
         shf_xy, lhf_xy, sstxy, t00, prec_xy,&
         latitude, longitude,&
         nx, ny, nzm, rho, adz, dz, pres, presi, time, nstep
    use params, only: npython, usepython
    use grid, only: day, caseid, case
    use microphysics, only: micro_field

    ! arguments
    real :: dtn


    integer k

    call push_state()

    ! Compute the necessary variables
    print *, 'python_caller.f90::state_to_python calling python code'
    call call_function(module_name, function_name)
    if (usepython) call get_state_from_python()

    print *, 'python_caller.f90::state_to_python storing state to sl_last and qt_last at time', time
    sl_last = t(1:nx, 1:ny, 1:nzm)
    qt_last = micro_field(1:nx, 1:ny, 1:nzm, 1)
    last_time_called = time
  end subroutine state_to_python

end module python_caller
