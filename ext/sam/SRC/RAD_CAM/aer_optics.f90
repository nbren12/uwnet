!bloss #include <misc.h>
!bloss ##include <params.h>

module aer_optics

  ! Purpose:
  !   initialize aerosol spectral optical properties
  !   for use in radiation subroutines

  ! Author: D. Fillmore

  ! dust properties are from C. Zender's Mie calculations
  ! all other properties are from OPAC project
  ! Optical Properties of Aerosols and Clouds
  ! Hess M., P. Koepke and I. Schult
  ! Bull. Amer. Met. Soc. 79, 831 - 844, 1998

  use shr_kind_mod, only: r4 => shr_kind_r4
  use ppgrid, only: aeroptics, dompi, rundatadir
!bloss #  use filenames, only: aeroptics

  implicit none

  integer, parameter, public :: idxVIS = 8     ! index to visible band
  integer, parameter, public :: nrh = 1000   ! number of relative humidity values for look-up-table
  integer, parameter, public :: nspint = 19   ! number of spectral intervals
  integer, parameter, public :: ndstsz = 4    ! number of dust size bins

  real(r4), public :: ksul(nrh, nspint)    ! sulfate specific extinction  ( m^2 g-1 )
  real(r4), public :: wsul(nrh, nspint)    ! sulfate single scattering albedo
  real(r4), public :: gsul(nrh, nspint)    ! sulfate asymmetry parameter
  real(r4), public :: kbg(nspint)          ! background specific extinction  ( m^2 g-1 )
  real(r4), public :: wbg(nspint)          ! background single scattering albedo
  real(r4), public :: gbg(nspint)          ! background asymmetry parameter
  real(r4), public :: ksslt(nrh, nspint)   ! sea-salt specific extinction  ( m^2 g-1 )
  real(r4), public :: wsslt(nrh, nspint)   ! sea-salt single scattering albedo
  real(r4), public :: gsslt(nrh, nspint)   ! sea-salt asymmetry parameter
  real(r4), public :: kcphil(nrh, nspint)  ! hydrophilic carbon specific extinction  ( m^2 g-1 )
  real(r4), public :: wcphil(nrh, nspint)  ! hydrophilic carbon single scattering albedo
  real(r4), public :: gcphil(nrh, nspint)  ! hydrophilic carbon asymmetry parameter
  real(r4), public :: kcphob(nspint)       ! hydrophobic carbon specific extinction  ( m^2 g-1 )
  real(r4), public :: wcphob(nspint)       ! hydrophobic carbon single scattering albedo
  real(r4), public :: gcphob(nspint)       ! hydrophobic carbon asymmetry parameter
  real(r4), public :: kcb(nspint)          ! black carbon specific extinction  ( m^2 g-1 )
  real(r4), public :: wcb(nspint)          ! black carbon single scattering albedo
  real(r4), public :: gcb(nspint)          ! black carbon asymmetry parameter
  real(r4), public :: kvolc(nspint)        ! volcanic specific extinction  ( m^2 g-1)
  real(r4), public :: wvolc(nspint)        ! volcanic single scattering albedo
  real(r4), public :: gvolc(nspint)        ! volcanic asymmetry parameter
  real(r4), public :: kdst(ndstsz, nspint) ! dust specific extinction  ( m^2 g-1 )
  real(r4), public :: wdst(ndstsz, nspint) ! dust single scattering albedo
  real(r4), public :: gdst(ndstsz, nspint) ! dust asymmetry parameter

  public :: aer_optics_initialize
  public :: aer_optics_log
  private :: exp_interpol

  save

contains

  subroutine aer_optics_initialize

!    use shr_kind_mod, only: r4 => shr_kind_r4
    use ppgrid, only: masterproc
!bloss    use pmgrid  ! masterproc is here
!bloss    use ioFileMod, only: getfil

!bloss#if ( defined SPMD )
!bloss    use mpishorthand
!bloss#endif
    implicit none

    include 'netcdf.inc'


    integer :: nrh_opac  ! number of relative humidity values for OPAC data
    integer :: nbnd      ! number of spectral bands, should be identical to nspint
    real(r4), parameter :: wgt_sscm = 6.0 / 7.0

    real(r4), allocatable :: rh_opac(:)        ! OPAC relative humidity grid
    real(r4), allocatable :: ksul_opac(:,:)    ! sulfate  extinction
    real(r4), allocatable :: wsul_opac(:,:)    !          single scattering albedo
    real(r4), allocatable :: gsul_opac(:,:)    !          asymmetry parameter
    real(r4), allocatable :: ksslt_opac(:,:)   ! sea-salt
    real(r4), allocatable :: wsslt_opac(:,:)
    real(r4), allocatable :: gsslt_opac(:,:)
    real(r4), allocatable :: kssam_opac(:,:)   ! sea-salt accumulation mode
    real(r4), allocatable :: wssam_opac(:,:)
    real(r4), allocatable :: gssam_opac(:,:)
    real(r4), allocatable :: ksscm_opac(:,:)   ! sea-salt coarse mode
    real(r4), allocatable :: wsscm_opac(:,:)
    real(r4), allocatable :: gsscm_opac(:,:)
    real(r4), allocatable :: kcphil_opac(:,:)  ! hydrophilic organic carbon
    real(r4), allocatable :: wcphil_opac(:,:)
    real(r4), allocatable :: gcphil_opac(:,:)

    character(len=256) :: locfn            ! local file

    integer :: krh_opac  ! rh index for OPAC rh grid
    integer :: krh       ! another rh index
    integer :: ksz       ! dust size bin index
    integer :: kbnd      ! band index

    real(r4) :: rh   ! local relative humidity variable

    integer :: nc_id
    integer :: dst_id, rh_id, bnd_id
    integer :: rh_opac_id
    integer :: ksul_opac_id, wsul_opac_id, gsul_opac_id
    integer :: kssam_opac_id, wssam_opac_id, gssam_opac_id
    integer :: ksscm_opac_id, wsscm_opac_id, gsscm_opac_id
    integer :: kcphil_opac_id, wcphil_opac_id, gcphil_opac_id
    integer :: kcb_id, wcb_id, gcb_id
    integer :: kvolc_id, wvolc_id, gvolc_id
    integer :: kdst_id, wdst_id, gdst_id
    integer :: kbg_id, wbg_id, gbg_id

    if ( masterproc ) then

    write (6, '(2x, a)') '_______________________________________________________'
    write (6, '(2x, a)') '_______ initializing aerosol optical properties _______'
    write (6, '(2x, a)') '_______________________________________________________'

!bloss    call getfil(aeroptics, locfn, 0)
    locfn = trim(rundatadir)//trim(aeroptics)

    call wrap_open(locfn, 0, nc_id)
    call wrap_inq_dimid(nc_id, 'dst', dst_id)
    call wrap_inq_dimid(nc_id, 'rh', rh_id)
    call wrap_inq_dimid(nc_id, 'wvl_cam', bnd_id)

    call wrap_inq_dimlen(nc_id, rh_id, nrh_opac)
    call wrap_inq_dimlen(nc_id, bnd_id, nbnd)

    write (*, '(2x, a, 2x, i2)') 'aer_optics_initialize : nrh_opac =', nrh_opac
    write (*, '(2x, a, 2x, i2)') 'aer_optics_initialize : nbnd =', nbnd

    ! allocate local arrays
    allocate(rh_opac(nrh_opac))
    allocate(ksul_opac(nrh_opac, nbnd))
    allocate(wsul_opac(nrh_opac, nbnd))
    allocate(gsul_opac(nrh_opac, nbnd))
    allocate(ksslt_opac(nrh_opac, nbnd))
    allocate(wsslt_opac(nrh_opac, nbnd))
    allocate(gsslt_opac(nrh_opac, nbnd))
    allocate(kssam_opac(nrh_opac, nbnd))
    allocate(wssam_opac(nrh_opac, nbnd))
    allocate(gssam_opac(nrh_opac, nbnd))
    allocate(ksscm_opac(nrh_opac, nbnd))
    allocate(wsscm_opac(nrh_opac, nbnd))
    allocate(gsscm_opac(nrh_opac, nbnd))
    allocate(kcphil_opac(nrh_opac, nbnd))
    allocate(wcphil_opac(nrh_opac, nbnd))
    allocate(gcphil_opac(nrh_opac, nbnd))

    call wrap_inq_varid(nc_id, 'rh', rh_opac_id)
    call wrap_inq_varid(nc_id, 'ext_cam_sul', ksul_opac_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_sul', wsul_opac_id)
    call wrap_inq_varid(nc_id, 'asm_cam_sul', gsul_opac_id)
    call wrap_inq_varid(nc_id, 'ext_cam_ssam', kssam_opac_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_ssam', wssam_opac_id)
    call wrap_inq_varid(nc_id, 'asm_cam_ssam', gssam_opac_id)
    call wrap_inq_varid(nc_id, 'ext_cam_sscm', ksscm_opac_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_sscm', wsscm_opac_id)
    call wrap_inq_varid(nc_id, 'asm_cam_sscm', gsscm_opac_id)
    call wrap_inq_varid(nc_id, 'ext_cam_waso', kcphil_opac_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_waso', wcphil_opac_id)
    call wrap_inq_varid(nc_id, 'asm_cam_waso', gcphil_opac_id)
    call wrap_inq_varid(nc_id, 'ext_cam_soot', kcb_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_soot', wcb_id)
    call wrap_inq_varid(nc_id, 'asm_cam_soot', gcb_id)
    call wrap_inq_varid(nc_id, 'ext_cam_volc', kvolc_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_volc', wvolc_id)
    call wrap_inq_varid(nc_id, 'asm_cam_volc', gvolc_id)
    call wrap_inq_varid(nc_id, 'ext_cam_dust', kdst_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_dust', wdst_id)
    call wrap_inq_varid(nc_id, 'asm_cam_dust', gdst_id)
    call wrap_inq_varid(nc_id, 'ext_cam_bg', kbg_id)
    call wrap_inq_varid(nc_id, 'ssa_cam_bg', wbg_id)
    call wrap_inq_varid(nc_id, 'asm_cam_bg', gbg_id)

    call wrap_get_var_realx(nc_id, rh_opac_id, rh_opac)
    call wrap_get_var_realx(nc_id, ksul_opac_id, ksul_opac)
    call wrap_get_var_realx(nc_id, wsul_opac_id, wsul_opac)
    call wrap_get_var_realx(nc_id, gsul_opac_id, gsul_opac)
    call wrap_get_var_realx(nc_id, kssam_opac_id, kssam_opac)
    call wrap_get_var_realx(nc_id, wssam_opac_id, wssam_opac)
    call wrap_get_var_realx(nc_id, gssam_opac_id, gssam_opac)
    call wrap_get_var_realx(nc_id, ksscm_opac_id, ksscm_opac)
    call wrap_get_var_realx(nc_id, wsscm_opac_id, wsscm_opac)
    call wrap_get_var_realx(nc_id, gsscm_opac_id, gsscm_opac)
    call wrap_get_var_realx(nc_id, kcphil_opac_id, kcphil_opac)
    call wrap_get_var_realx(nc_id, wcphil_opac_id, wcphil_opac)
    call wrap_get_var_realx(nc_id, gcphil_opac_id, gcphil_opac)
    call wrap_get_var_realx(nc_id, kcb_id, kcb)
    call wrap_get_var_realx(nc_id, wcb_id, wcb)
    call wrap_get_var_realx(nc_id, gcb_id, gcb)
    call wrap_get_var_realx(nc_id, kvolc_id, kvolc)
    call wrap_get_var_realx(nc_id, wvolc_id, wvolc)
    call wrap_get_var_realx(nc_id, gvolc_id, gvolc)
    call wrap_get_var_realx(nc_id, kdst_id, kdst)
    call wrap_get_var_realx(nc_id, wdst_id, wdst)
    call wrap_get_var_realx(nc_id, gdst_id, gdst)
    call wrap_get_var_realx(nc_id, kbg_id, kbg)
    call wrap_get_var_realx(nc_id, wbg_id, wbg)
    call wrap_get_var_realx(nc_id, gbg_id, gbg)

    ! map OPAC aerosol species onto CAM aerosol species
    ! CAM name             OPAC name
    ! sul   or SO4         = suso                  sulfate soluble
    ! sslt  or SSLT        = 1/7 ssam + 6/7 sscm   sea-salt accumulation/coagulation mode
    ! cphil or CPHI        = waso                  water soluble (carbon)
    ! cphob or CPHO        = waso @ rh = 0
    ! cb    or BCPHI/BCPHO = soot

    ksslt_opac(:,:) = (1.0 - wgt_sscm) * kssam_opac(:,:) + wgt_sscm * ksscm_opac(:,:)

    wsslt_opac(:,:) = ( (1.0 - wgt_sscm) * kssam_opac(:,:) * wssam_opac(:,:) &
                  + wgt_sscm * ksscm_opac(:,:) * wsscm_opac(:,:) ) &
                  / ksslt_opac(:,:)

    gsslt_opac(:,:) = ( (1.0 - wgt_sscm) * kssam_opac(:,:) * wssam_opac(:,:) * gssam_opac(:,:) &
                  + wgt_sscm * ksscm_opac(:,:) * wsscm_opac(:,:) * gsscm_opac(:,:) ) &
                   / ( ksslt_opac(:,:) * wsslt_opac(:,:) )

    kcphob(:) = kcphil_opac(1,:)
    wcphob(:) = wcphil_opac(1,:)
    gcphob(:) = gcphil_opac(1,:)

    ! interpolate optical properties of hygrospopic aerosol species
    !   onto a uniform relative humidity grid

    do krh = 1, nrh
      rh = 1.0_r4 / nrh * (krh - 1)
      do kbnd = 1, nbnd
        ksul(krh, kbnd) = exp_interpol( rh_opac, &
          ksul_opac(:, kbnd) / ksul_opac(1, kbnd), rh ) * ksul_opac(1, kbnd)
        wsul(krh, kbnd) = lin_interpol( rh_opac, &
          wsul_opac(:, kbnd) / wsul_opac(1, kbnd), rh ) * wsul_opac(1, kbnd)
        gsul(krh, kbnd) = lin_interpol( rh_opac, &
          gsul_opac(:, kbnd) / gsul_opac(1, kbnd), rh ) * gsul_opac(1, kbnd)
        ksslt(krh, kbnd) = exp_interpol( rh_opac, &
          ksslt_opac(:, kbnd) / ksslt_opac(1, kbnd), rh ) * ksslt_opac(1, kbnd)
        wsslt(krh, kbnd) = lin_interpol( rh_opac, &
          wsslt_opac(:, kbnd) / wsslt_opac(1, kbnd), rh ) * wsslt_opac(1, kbnd)
        gsslt(krh, kbnd) = lin_interpol( rh_opac, &
          gsslt_opac(:, kbnd) / gsslt_opac(1, kbnd), rh ) * gsslt_opac(1, kbnd)
        kcphil(krh, kbnd) = exp_interpol( rh_opac, &
          kcphil_opac(:, kbnd) / kcphil_opac(1, kbnd), rh ) * kcphil_opac(1, kbnd)
        wcphil(krh, kbnd) = lin_interpol( rh_opac, &
          wcphil_opac(:, kbnd) / wcphil_opac(1, kbnd), rh ) * wcphil_opac(1, kbnd)
        gcphil(krh, kbnd) = lin_interpol( rh_opac, &
          gcphil_opac(:, kbnd) / gcphil_opac(1, kbnd), rh )  * gcphil_opac(1, kbnd)
      end do
    end do

    deallocate(rh_opac)
    deallocate(ksul_opac)
    deallocate(wsul_opac)
    deallocate(gsul_opac)
    deallocate(ksslt_opac)
    deallocate(wsslt_opac)
    deallocate(gsslt_opac)
    deallocate(kssam_opac)
    deallocate(wssam_opac)
    deallocate(gssam_opac)
    deallocate(ksscm_opac)
    deallocate(wsscm_opac)
    deallocate(gsscm_opac)
    deallocate(kcphil_opac)
    deallocate(wcphil_opac)
    deallocate(gcphil_opac)

    ! write optical constants to log file for debugging

    write (6, '(2x, a)') '_______ hygroscopic growth in visible band _______'
    call aer_optics_log_rh( 'SO4', ksul(:, 8), wsul(:, 8), gsul(:, 8) )
    call aer_optics_log_rh( 'SSLT', ksslt(:, 8), wsslt(:, 8), gsslt(:, 8) )
    call aer_optics_log_rh( 'OCPHI', kcphil(:, 8), wcphil(:, 8), gcphil(:, 8) )

    end if ! end if ( masterproc )

    ! broadcast aerosol optical constants to all nodes
    if (dompi) then
       call task_bcast_float4(0, ksul, nrh * nspint)
       call task_bcast_float4(0, wsul, nrh * nspint)
       call task_bcast_float4(0, gsul, nrh * nspint)
       call task_bcast_float4(0, ksslt, nrh * nspint)
       call task_bcast_float4(0, wsslt, nrh * nspint)
       call task_bcast_float4(0, gsslt, nrh * nspint)
       call task_bcast_float4(0, kcphil, nrh * nspint)
       call task_bcast_float4(0, wcphil, nrh * nspint)
       call task_bcast_float4(0, gcphil, nrh * nspint)
       call task_bcast_float4(0, kcphob, nspint)
       call task_bcast_float4(0, wcphob, nspint)
       call task_bcast_float4(0, gcphob, nspint)
       call task_bcast_float4(0, kcb, nspint)
       call task_bcast_float4(0, wcb, nspint)
       call task_bcast_float4(0, gcb, nspint)
       call task_bcast_float4(0, kvolc, nspint)
       call task_bcast_float4(0, wvolc, nspint)
       call task_bcast_float4(0, gvolc, nspint)
       call task_bcast_float4(0, kdst, ndstsz * nspint)
       call task_bcast_float4(0, wdst, ndstsz * nspint)
       call task_bcast_float4(0, gdst, ndstsz * nspint)
       call task_bcast_float4(0, kbg, nspint)
       call task_bcast_float4(0, wbg, nspint)
       call task_bcast_float4(0, gbg, nspint)
    end if

!bloss#if ( defined SPMD )
!bloss    call mpibcast(ksul, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wsul, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gsul, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(ksslt, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wsslt, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gsslt, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kcphil, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wcphil, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gcphil, nrh * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kcphob, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wcphob, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gcphob, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kcb, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wcb, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gcb, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kvolc, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wvolc, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gvolc, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kdst, ndstsz * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wdst, ndstsz * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gdst, ndstsz * nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(kbg, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(wbg, nspint, mpir8, 0, mpicom)
!bloss    call mpibcast(gbg, nspint, mpir8, 0, mpicom)
!bloss#endif

  end subroutine aer_optics_initialize

  subroutine aer_optics_log(name, ext, ssa, asm)

    ! Purpose:
    !   write aerosol optical constants to log file

    ! Author: D. Fillmore

!    use shr_kind_mod, only: r4 => shr_kind_r4

    implicit none

    character(len=*), intent(in) :: name
    real(r4), intent(in) :: ext(nspint)
    real(r4), intent(in) :: ssa(nspint)
    real(r4), intent(in) :: asm(nspint)

    integer :: kbnd

    write (6, '(2x, a)') name
    write (6, '(2x, a, 4x, a, 4x, a, 4x, a)') 'SW band', 'ext (m^2 g-1)', ' ssa', ' asm'
    do kbnd = 1, nspint
      write (6, '(2x, i7, 4x, f13.2, 4x, f4.2, 4x, f4.2)') kbnd, ext(kbnd), ssa(kbnd), asm(kbnd)
    end do

  end subroutine aer_optics_log


  subroutine aer_optics_log_rh(name, ext, ssa, asm)

    ! Purpose:
    !   write out aerosol optical properties
    !   for a set of test rh values
    !   to test hygroscopic growth interpolation

    ! Author: D. Fillmore

!    use shr_kind_mod, only: r4 => shr_kind_r4

    implicit none

    character(len=*), intent(in) :: name
    real(r4), intent(in) :: ext(nrh)
    real(r4), intent(in) :: ssa(nrh)
    real(r4), intent(in) :: asm(nrh)

    integer :: krh_test
    integer, parameter :: nrh_test = 36
    integer :: krh
    real(r4) :: rh
    real(r4) :: rh_test(nrh_test)
    real(r4) :: exti
    real(r4) :: ssai
    real(r4) :: asmi
    real(r4) :: wrh

    do krh_test = 1, nrh_test
      rh_test(krh_test) = sqrt(sqrt(sqrt(sqrt(((krh_test - 1.0) / (nrh_test - 1))))))
    enddo
    write (6, '(2x, a)') name
    write (6, '(2x, a, 4x, a, 4x, a, 4x, a)') '   rh', 'ext (m^2 g-1)', '  ssa', '  asm'

    ! loop through test rh values
    do krh_test = 1, nrh_test
      ! find corresponding rh index
      rh = rh_test(krh_test)
      krh = min(floor( (rh) * nrh ) + 1, nrh - 1)
      wrh = (rh) *nrh - krh
      exti = ext(krh + 1) * (wrh + 1) - ext(krh) * wrh
      ssai = ssa(krh + 1) * (wrh + 1) - ssa(krh) * wrh
      asmi = asm(krh + 1) * (wrh + 1) - asm(krh) * wrh
      write (6, '(2x, f5.3, 4x, f13.3, 4x, f5.3, 4x, f5.3)') rh_test(krh_test), exti, ssai, asmi
      ! write (6, '(2x, f5.3, 4x, f13.3, 4x, f5.3, 4x, f5.3)') rh_test(krh_test), ext(krh), ssa(krh), asm(krh)
    end do

  end subroutine aer_optics_log_rh


  function exp_interpol(x, f, y) result(g)

    ! Purpose:
    !   interpolates f(x) to point y
    !   assuming f(x) = f(x0) exp a(x - x0)
    !   where a = ( ln f(x1) - ln f(x0) ) / (x1 - x0)
    !   x0 <= x <= x1
    !   assumes x is monotonically increasing

    ! Author: D. Fillmore

!    use shr_kind_mod, only: r4 => shr_kind_r4

    implicit none

    real(r4), intent(in), dimension(:) :: x  ! grid points
    real(r4), intent(in), dimension(:) :: f  ! grid function values
    real(r4), intent(in) :: y                ! interpolation point
    real(r4) :: g                            ! interpolated function value

    integer :: k  ! interpolation point index
    integer :: n  ! length of x
    real(r4) :: a

    n = size(x)

    ! find k such that x(k) < y =< x(k+1)
    ! set k = 1 if y <= x(1)  and  k = n-1 if y > x(n)

    if (y <= x(1)) then
      k = 1
    else if (y >= x(n)) then
      k = n - 1
    else
      k = 1
      do while (y > x(k+1) .and. k < n)
        k = k + 1
      end do
    end if

    ! interpolate
    a = (  log( f(k+1) / f(k) )  ) / ( x(k+1) - x(k) )
    g = f(k) * exp( a * (y - x(k)) )

  end function exp_interpol

  function lin_interpol(x, f, y) result(g)

    ! Purpose:
    !   interpolates f(x) to point y
    !   assuming f(x) = f(x0) + a * (x - x0)
    !   where a = ( f(x1) - f(x0) ) / (x1 - x0)
    !   x0 <= x <= x1
    !   assumes x is monotonically increasing

    ! Author: D. Fillmore

!    use shr_kind_mod, only: r4 => shr_kind_r4

    implicit none

    real(r4), intent(in), dimension(:) :: x  ! grid points
    real(r4), intent(in), dimension(:) :: f  ! grid function values
    real(r4), intent(in) :: y                ! interpolation point
    real(r4) :: g                            ! interpolated function value

    integer :: k  ! interpolation point index
    integer :: n  ! length of x
    real(r4) :: a

    n = size(x)

    ! find k such that x(k) < y =< x(k+1)
    ! set k = 1 if y <= x(1)  and  k = n-1 if y > x(n)

    if (y <= x(1)) then
      k = 1
    else if (y >= x(n)) then
      k = n - 1
    else
      k = 1
      do while (y > x(k+1) .and. k < n)
        k = k + 1
      end do
    end if

    ! interpolate
    a = (  f(k+1) - f(k) ) / ( x(k+1) - x(k) )
    g = f(k) + a * (y - x(k))

  end function lin_interpol

end module aer_optics

