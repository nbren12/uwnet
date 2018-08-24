module read_netcdf_3d
  implicit none
  public :: set_field_from_nc, ncreadvarxyz

  private

  character(len=*), parameter :: file_name = 'read_netcdf_3d.f90'
  include 'netcdf.inc'

contains
  subroutine set_field_from_nc(path, micro_field, index_water_vapor)
    ! Arguments
    use grid
    use vars, only: u, v, w, t, gamaz, rho, rhow, qcl, qpl, qci, qpi
    use microphysics, only: micro_diagnose, micro_scheme_name, micro_init, qn

    character(len=*), intent(in) :: path
    real, intent(out)          :: micro_field(:,:,:,:)
    integer, intent(in)          :: index_water_vapor

    ! This is the formula used for TABS
    ! tabs(i,j,k) = t(i,j,k)-gamaz(k)+ fac_cond * (qcl(i,j,k)+qpl(i,j,k)) +&
    !      fac_sub *(qci(i,j,k) + qpi(i,j,k))

    ! Locals
    real, dimension(nx, ny, nzm) :: tabs, qv, qp, qn_local, tmp, tmp1
    real :: p0
    integer :: nmicro
    integer :: ncid, status
    logical :: use_nf_real, required

    if (micro_scheme_name() /= "sam1mom" ) then
       print *, file_name, ': NetCDF initialization only works with SAM1MOM microphysics'
       call task_abort()
    end if

    if (nsubdomains > 1) then
       print *, 'NetCDF initialization only works with a single processor'
       call task_abort()
    end if

    ! initialize microphysics
    call micro_init()

    use_nf_real = .true.
    required = .true.

    nmicro = size(micro_field, 4)

    print *, file_name, ': Initializing fields from ', path
    status = nf_open(path, NF_NOWRITE, ncid)

    ! read in data
    call ncreadvarxyz(ncid, "U", tmp, use_nf_real, status, required)
    u(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarxyz(ncid, "V", tmp, use_nf_real, status, required)
    v(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarxyz(ncid, "W", tmp, use_nf_real, status, required)
    w(1:nx, 1:ny, 1:nzm) = tmp(:,:,1:nzm)

    call ncreadvarxyz(ncid, "TABS", tmp, use_nf_real, status, required)
    tabs(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarxyz(ncid, "QV", tmp, use_nf_real, status, required)
    qv(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarxyz(ncid, "QN", tmp, use_nf_real, status, required)
    ! qn(1:nx, 1:ny, 1:nzm) = 0.0
    qn_local(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarxyz(ncid, "QP", tmp, use_nf_real, status, required)
    ! qp(1:nx, 1:ny, 1:nzm) = 0.0
    qp(1:nx, 1:ny, 1:nzm) = tmp

    call ncreadvarz(ncid, "RHO", rho, use_nf_real, status, required)
    call ncreadscalar(ncid, "Ps", p0, use_nf_real, status, required)

    ! compute the interface and cell centered pressures
    call compute_presi(rho, p0, zi, presi)
    call loginterp(presi, zi, z, pres)
    call calculate_rhow(pres, z, rhow)

    ! initialize microphsyical fields
    call calculate_micro_field_sam1mom(qv, qn_local, qp, micro_field, dimx1_s, dimy1_s)
    qn(1:nx, 1:ny, 1:nzm) = qn_local/1.e3

    ! initialize all the t0, q0, and everything else using diagnose
    call micro_diagnose()
    call calculate_static_energy(tabs, qcl, qpl, qci, qpi, gamaz, tmp)

    ! initialize the thermodynamic variables
    t(1:nx, 1:ny, 1:nzm) = tmp
    call diagnose()

    ! call pressure solver to ensure the input fields are divergence free
    print *, 'read_netdf_3d.f90::calling pressure solver'
    call pressure_correct_velocities()

    ! save output
    print *, 'read_netdf_3d.f90::saving output to disk'
    call write_fields3D()

  end subroutine set_field_from_nc

  subroutine pressure_correct_velocities()
    use vars

    dtn = 1.0
    dt3(na) = dtn
    dtfactor = dtn/dt

    call abcoefs()
    call zero()
    call boundaries(0)
    call pressure()
    call adams()
    call uvw()
  end subroutine pressure_correct_velocities

  subroutine calculate_micro_field_sam1mom(qv, qn, qp, micro_field, xs, ys)
    ! Compute the SAM1MOM microphysics fields from qv, qn and qp
    ! These fields are the total non-precipitation water Q, and the
    ! precipitating water QP
    use microphysics, only: dimx1_s, dimy1_s
    real, intent(in), dimension(:, :, :) :: qv, qn, qp
    integer, intent(in) :: xs, ys
    real, intent(out)          :: micro_field(xs:,ys:,:,:)
    !locals
    integer i,j,k

    do k=lbound(qv,3),ubound(qv,3)
       do j=lbound(qv,2),ubound(qv,2)
          do i=lbound(qv,1),ubound(qv,1)
             ! first field is the total non-precipitating water
             micro_field(i,j,k,1) = (qv(i,j,k) + qn(i,j,k))/1.e3
             ! the second field is the precipitating water
             micro_field(i,j,k,2) = qp(i,j,k)/1.e3
          end do
       end do
    end do
  end subroutine calculate_micro_field_sam1mom

  subroutine calculate_static_energy(tabs, qcl, qpl, qci, qpi, gamaz, t)
    ! calculate the dry static energy
    use params, only: fac_cond, fac_sub
    real, intent(in) :: tabs(:, :, :), gamaz(:)
    real, intent(in), dimension(:,:,:) :: qcl, qpl, qci, qpi
    real, intent(out) :: t(:,:,:)
    !locals
    integer i,j,k

    do k=lbound(tabs,3),ubound(tabs,3)
       do j=lbound(tabs,2),ubound(tabs,2)
          do i=lbound(tabs,1),ubound(tabs,1)
             t(i,j,k) = tabs(i,j,k) + gamaz(k) - fac_cond * (qcl(i,j,k)+qpl(i,j,k)) -&
                  fac_sub *(qci(i,j,k) + qpi(i,j,k))
          end do
       end do
    end do
  end subroutine calculate_static_energy

  subroutine compute_presi(rho, p0, zi, presi)
    use params, only: ggr
    real, intent(in) :: rho(:), p0, zi(:)
    real, intent(out) :: presi(:)

    integer :: nzm, nz, k
    nz = size(presi, 1)
    presi(1) = p0
    do k=1,size(presi, 1)-1
       presi(k+1) = presi(k) - rho(k) * ggr * (zi(k+1) - zi(k))/100.
    end do

  end subroutine compute_presi

  subroutine calculate_rhow(pres, z, rhow)
    ! interpolation function taken from pressz.f90
    ! interpolates in log-pressure space
    use params, only: ggr
    real, intent(in), dimension(:) :: pres, z
    real, intent(out), dimension(:) :: rhow
    integer k, nzm, nz

    nz = size(z)
    nzm  = nz - 1
    do k=2,nzm
       rhow(k) =  (pres(k-1)-pres(k))/(z(k)-z(k-1))/ggr*100.
    end do
    rhow(1) = 2*rhow(2) - rhow(3)
    rhow(nz)= 2*rhow(nzm) - rhow(nzm-1)
  end subroutine calculate_rhow

  subroutine loginterp(presi, zi, z, pres)
    ! interpolation function taken from pressz.f90
    ! interpolates in log-pressure space
    real, intent(in), dimension(:) :: presi, zi
    real, intent(out), dimension(:) :: pres, z
    integer k

    do k=1, size(pres)
       pres(k) = exp(log(presi(k))+log(presi(k+1)/presi(k))* &
            (z(k)-zi(k))/(zi(k+1)-zi(k)))
    end do
  end subroutine loginterp

  subroutine ncreadvarxyz( NCID, varName, var, use_nf_real, status, required)
    ! subroutine ncreadvarxyz( NCID, varName, ntime, nlev, var,&
    !      use_nf_real, status, required)
    !based on John Truesdale's getncdata_real_1d

    ! input/output variables
    integer, intent(in)   :: NCID
    character, intent(in) :: varName*(*)
    logical, intent(in) :: required, USE_NF_REAL

    integer, intent(out) :: status
    real, intent(inout) :: var(:,:,:)

    ! local variables
    integer :: nx, ny, nz
    integer :: varID
    character     dim_name*( NF_MAX_NAME )
    integer     var_dimIDs( NF_MAX_VAR_DIMS )
    integer     start(3), count(3)
    integer     var_ndims, dim_size, dims_set, i, n, var_type
    logical usable_var
    logical masterproc
    masterproc = .true.

    nx = size(var, 1)
    ny = size(var, 2)
    nz = size(var, 3)

    ! get variable ID
    STATUS = NF_INQ_VARID( NCID, varName, varID )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ID for ',&
               varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note(readiopdata.f90): Optional variable ', varName,&
               ' not found in file'
          return
       endif
    endif

    !
    ! Check the var variable's information with what we are expecting
    ! it to be.
    !
    STATUS = NF_INQ_VARNDIMS( NCID, varID, var_ndims )
    if ( var_ndims .GT. 4 ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: The input var',varName, &
            'has', var_ndims, 'dimensions'
       STATUS = -1
       return
    endif

    STATUS =  NF_INQ_VARTYPE(NCID, varID, var_type)
    if ( var_type .NE. NF_FLOAT .and. var_type .NE. NF_DOUBLE .and. &
         var_type .NE. NF_INT ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: The input var',varName, &
            'has unknown type', var_type
       STATUS = -1
       return
    endif

    STATUS = NF_INQ_VARDIMID( NCID, varID, var_dimIDs )
    if ( STATUS .NE. NF_NOERR ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90:Cant get dimension IDs for', varName
       return
    endif
    !
    !     Initialize the start and count arrays
    !

    dims_set = 0
    do i =  var_ndims, 1, -1

       usable_var = .false.
       STATUS = NF_INQ_DIMNAME( NCID, var_dimIDs( i ), dim_name )
       if ( STATUS .NE. NF_NOERR ) then
          if(masterproc) write(6,*) &
               'Error: getncdata.f90 - can''t get dim name', &
               'var_ndims = ', var_ndims, ' i = ',i
          return
       endif

       ! extract the single latitude in the iop file
       if ( dim_name .EQ. 'x' ) then
          start( i ) = 1
          count( i ) = nx
          dims_set = dims_set + 1
          usable_var = .true.
       endif

       ! extract the single longitude in the iop file
       if ( dim_name .EQ. 'y' ) then
          start( i ) = 1
          count( i ) = ny
          dims_set = dims_set + 1
          usable_var = .true.
       endif

       ! extract one level with each call
       if ( dim_name .EQ. 'z' ) then
          start( i ) = 1
          count( i ) = nz
          dims_set = dims_set + 1
          usable_var = .true.
       endif

       if ( usable_var .EQV. .false. ) then
          if(masterproc) write(6,*) &
               'ERROR - getncdata.f90: The input var ', &
               varName, ' has an unusable dimension ', dim_name
          STATUS = -1
       endif
    end do
    if ( dims_set .NE. var_ndims ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: Could not find all the', &
            'dimensions for input var ', varName
       if(masterproc) write(6,*) &
            'Found ',dims_set, ' of ',var_ndims
       STATUS = -1
    endif

    if (use_nf_real) then
       STATUS = NF_GET_VARA_REAL( NCID, varID, start, count, var)
    else
       STATUS = NF_GET_VARA_DOUBLE( NCID, varID, start, count, var)
    endif

    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ', &
               varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note (readiopdata.f90) : Could not find ', varName
       endif
    endif


  end subroutine ncreadvarxyz

  subroutine ncreadvarz( NCID, varName, var, use_nf_real, status, required)
    ! subroutine ncreadvarxyz( NCID, varName, ntime, nlev, var,&
    !      use_nf_real, status, required)
    !based on John Truesdale's getncdata_real_1d

    ! input/output variables
    integer, intent(in)   :: NCID
    character, intent(in) :: varName*(*)
    logical, intent(in) :: required, USE_NF_REAL

    integer, intent(out) :: status
    real, intent(inout) :: var(:)

    ! local variables
    integer :: nx, ny, nz
    integer :: varID
    character     dim_name*( NF_MAX_NAME )
    integer     var_dimIDs( NF_MAX_VAR_DIMS )
    integer     start, count
    integer     var_ndims, dim_size, dims_set, i, n, var_type
    logical masterproc
    masterproc = .true.


    ! get variable ID
    STATUS = NF_INQ_VARID( NCID, varName, varID )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ID for ',&
               varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note(readiopdata.f90): Optional variable ', varName,&
               ' not found in file'
          return
       endif
    endif

    !
    ! Check the var variable's information with what we are expecting
    ! it to be.
    !
    STATUS = NF_INQ_VARNDIMS( NCID, varID, var_ndims )
    if ( var_ndims .GT. 1 ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: The input var',varName, &
            'has', var_ndims, 'dimensions'
       STATUS = -1
       return
    endif

    STATUS =  NF_INQ_VARTYPE(NCID, varID, var_type)
    if ( var_type .NE. NF_FLOAT .and. var_type .NE. NF_DOUBLE .and. &
         var_type .NE. NF_INT ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: The input var',varName, &
            'has unknown type', var_type
       STATUS = -1
       return
    endif

    STATUS = NF_INQ_VARDIMID( NCID, varID, var_dimIDs )
    if ( STATUS .NE. NF_NOERR ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90:Cant get dimension IDs for', varName
       return
    endif

    !
    !     Initialize the start and count arrays
    !
      STATUS = NF_INQ_DIMNAME( NCID, var_dimIDs( 1 ), dim_name )
      if ( STATUS .NE. NF_NOERR ) then
        if(masterproc) write(6,*) &
              'Error: getncdata.f90 - can''t get dim name', &
              'var_ndims = ', var_ndims, ' i = ',i
        return
       endif

      ! extract one level with each call
      if ( dim_name .EQ. 'z' ) then
        start = 1
        count = size(var, 1)
        dims_set = dims_set + 1
     else
        print*, file_name, ': variable does not have dimension z'
      endif


     if (use_nf_real) then
        STATUS = NF_GET_VARA_REAL( NCID, varID, start, count, var)
     else
        STATUS = NF_GET_VARA_DOUBLE( NCID, varID, start, count, var)
     endif

     if (STATUS .NE. NF_NOERR ) then
        if(required) then
           if(masterproc) write(6,*) &
                'ERROR(readiopdata.f90):Could not find variable ', &
                varName
           STATUS = NF_CLOSE( NCID )
           call task_abort()
        else
           if(masterproc) write(6,*) &
                'Note (readiopdata.f90) : Could not find ', varName
        endif
     endif


   end subroutine ncreadvarz

  subroutine ncreadscalar( NCID, varName, var, use_nf_real, status, required)
    ! subroutine ncreadvarxyz( NCID, varName, ntime, nlev, var,&
    !      use_nf_real, status, required)
    !based on John Truesdale's getncdata_real_1d

    ! input/output variables
    integer, intent(in)   :: NCID
    character, intent(in) :: varName*(*)
    logical, intent(in) :: required, USE_NF_REAL

    integer, intent(out) :: status
    real, intent(inout) :: var

    ! local variables
    integer :: nx, ny, nz
    integer :: varID
    character     dim_name*( NF_MAX_NAME )
    integer     var_dimIDs( NF_MAX_VAR_DIMS )
    integer     start, count
    integer     var_ndims, dim_size, dims_set, i, n, var_type
    logical masterproc
    masterproc = .true.


    ! get variable ID
    STATUS = NF_INQ_VARID( NCID, varName, varID )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ID for ',&
               varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note(readiopdata.f90): Optional variable ', varName,&
               ' not found in file'
          return
       endif
    endif

    !
    ! Check the var variable's information with what we are expecting
    ! it to be.
    !
    STATUS =  NF_INQ_VARTYPE(NCID, varID, var_type)
    if ( var_type .NE. NF_FLOAT .and. var_type .NE. NF_DOUBLE .and. &
         var_type .NE. NF_INT ) then
       if(masterproc) write(6,*) &
            'ERROR - getncdata.f90: The input var',varName, &
            'has unknown type', var_type
       STATUS = -1
       return
    endif

    !
    !     Initialize the start and count arrays
    !
    start = 1
    count = 1

     if (use_nf_real) then
        STATUS = NF_GET_VARA_REAL( NCID, varID, start, count, var)
     else
        STATUS = NF_GET_VARA_DOUBLE( NCID, varID, start, count, var)
     endif

     if (STATUS .NE. NF_NOERR ) then
        if(required) then
           if(masterproc) write(6,*) &
                'ERROR(readiopdata.f90):Could not find variable ', &
                varName
           STATUS = NF_CLOSE( NCID )
           call task_abort()
        else
           if(masterproc) write(6,*) &
                'Note (readiopdata.f90) : Could not find ', varName
        endif
     endif


   end subroutine ncreadscalar
 end module read_netcdf_3d
