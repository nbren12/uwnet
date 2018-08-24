program statToNetcdf 
  use netcdf
  use hbuffer
  implicit none 
  real,    external :: cin, cape
  integer, external :: iargc
    
  integer, parameter :: max_num_times = 24 * 365 * 3, &
                        max_num_heights = 100
  integer :: num_times, num_heights, num_params
  integer :: var, time
  real, dimension(max_num_times)   :: times
  real, dimension(max_num_heights) :: z, p
  real, dimension(:), allocatable  :: rho
  
  integer, parameter :: input_file_unit = 2
  integer            :: num_read, num_parms

  character (len = 256) :: filename
  character (len = 40)  :: caseid
  integer               :: nc_file_id, nc_var_id, &
                           time_dim_id, height_dim_id, &
                           time_var_id, height_var_id, p_var_id
                          
  real, dimension(:),    allocatable :: buffer_1d
  real, dimension(:, :), allocatable :: buffer_2d, buffer2_2d

  integer, dimension(HBUF_MAX_LENGTH) :: status 
  ! --------------------------------------------------------------------------
  ! Names, long names, and units for each class of variables
  ! --------------------------------------------------------------------------
  integer, parameter :: max_long_name_length = 40, &
                        max_abbr_name_length = 8,  & 
                        max_units_length     = 6
                        
  
  !
  ! One-dimensional vars 
  !
  integer, parameter :: num_vars = 29
  character(len = max_long_name_length), dimension(num_vars), parameter :: &
    long_names  = (/                            &
     'SST                                    ', &
     'Surface Pressure                       ', &
     'Shaded Cloud Fraction                  ', & 
     'Surface Precip. Fraction               ', & 
     'Cloud Fraction above 245K level        ', &
     'Maximum Updraft Velocity               ', &
     'Convective Precipitation Rate          ', &
     'Stratiform Precipitation Rate          ', &
     'Net LW flux at sfc                     ', &
     'Net LW flux at TOA                     ', &
     'Net LW flux at sfc (Clear Sky)         ', &
     'Net LW flux at TOA (Clear Sky)         ', &
     'Downward LW flux at sfc                ', &
     'Net SW flux at sfc                     ', &
     'Net SW flux at TOA                     ', &
     'Net SW flux at sfc (Clear Sky)         ', &
     'Net SW flux at TOA (Clear Sky)         ', &
     'Downward SW flux at sfc                ', &
     'Incoming SW flux at TOA                ', &
     'Observed SST                           ', &
     'Observed Latent Heat Flux              ', &
     'Observed Sensible Heat Flux            ', &
     'Low Cloud Fraction                     ', &
     'Middle Cloud Fraction                  ', &
     'High Cloud Fraction                    ', &
     'ISCCP Total Cloud Fraction (tau > 0.3) ', &
     'ISCCP Low Cloud Fraction (tau > 0.3)   ', &
     'ISCCP Middle Cloud Fraction (tau > 0.3)', &
     'ISCCP High Cloud Fraction (tau > 0.3)  ' /)

  character(len = max_abbr_name_length), dimension(num_vars), parameter  :: &
    abbr_names= (/ &
       'SST     ', 'Ps      ', 'CLDSHD  ', 'AREAPREC',             &
       'CLD245  ', 'WMAX    ', 'PRECC   ', 'PRECS   ',             &
       'LWNS    ', 'LWNT    ', 'LWNSC   ', 'LWNTC   ', 'LWDS    ', &
       'SWNS    ', 'SWNT    ', 'SWNSC   ', 'SWNTC   ', 'SWDS    ', &
       'SOLIN   ', 'SSTOBS  ', 'LHFOBS  ', 'SHFOBS  ',             &
                   'CLDLOW  ', 'CLDMID  ', 'CLDHI   ',             &
       'ISCCPTOT', 'ISCCPLOW', 'ISCCPMID', 'ISCCPHGH' /)
       
  character(len = max_units_length),     dimension(num_vars), parameter :: &
    units = (/ &
       'K     ', 'mb    ', '      ', '      ',  &
       '      ', 'm/s   ', 'mm/day', 'mm/day',           &
       'W/m2  ', 'W/m2  ', 'W/m2  ', 'W/m2  ', 'W/m2  ', &
       'W/m2  ', 'W/m2  ', 'W/m2  ', 'W/m2  ', 'W/m2  ', &
       'W/m2  ', 'W/m2  ', 'K     ', 'W/m2  ',           &
                 '      ', '      ', '      ',           &
       '      ', '      ', '      ', '      ' /) 
  integer, dimension(num_vars) :: nc_varIds

  !
  ! Derived variables 
  !
  integer, parameter :: num_derived_vars = 14
  character(len = max_long_name_length), dimension(num_derived_vars), parameter :: &
    long_names_derived  = (/                      &
       'Surface Precipitation                  ', &
       'Latent Heat Flux                       ', &
       'Sensible Heat Flux                     ', &
       'Precipitable Water                     ', &
       'Observed Precipitable Water            ', &
       'Cloud Water Path                       ', &
       'Ice Water Path                         ', &
       'Rain Water Path                        ', &
       'Snow Water Path                        ', &
       'Graupel Water Path                     ', &
       'CAPE                                   ', &
       'CIN                                    ', &
       'CAPEOBS                                ', &
       'CINOBS                                 ' /) 
  character(len = max_abbr_name_length), dimension(num_derived_vars), parameter  :: &
    abbr_names_derived = (/ &
       'PREC    ', 'LHF     ', 'SHF     ', 'PW      ', 'PWOBS   ', &
       'CWP     ', 'IWP     ', 'RWP     ', 'SWP     ', 'GWP     ', &
       'CAPE    ', 'CIN     ', 'CAPEOBS ', 'CINOBS  ' /)
       
  character(len = max_units_length),     dimension(num_derived_vars), parameter :: &
    units_derived = (/ &
        'mm/day',   'W/m2  ',   'W/m2  ',  'mm    ',   'mm    ',    &
        'g/m2  ',   'g/m2  ',   'g/m2  ',  'g/m2  ',   'g/m2  ',    &
        'J/kg  ',   'J/kg  ',   'J/kg  ',  'J/kg  ' /) 
  integer, dimension(num_vars) :: nc_varIds_derived
  
  !
  ! Two D variables
  ! Names, long names, and units are read directly from the binary file
  !
  integer, dimension(HBUF_MAX_LENGTH) :: nc_varIds_2D

  ! --------------------------------------------------------------------------
  ! Code begins   
  ! --------------------------------------------------------------------------
  ! Open input file, read number and values for time and height  
  !
  if(iargc()  == 0) stop "You must specify an input file" 
  call getarg(1, filename)
  print *,'open file: ', trim(filename)
  open(input_file_unit, file= trim(filename), status='old', form='unformatted')

  call HBUF_info(input_file_unit, num_times, times, num_heights, z, p, caseid)
  if(num_times   > max_num_times) then 
    print *,   "Need space for more times: ",   num_times, max_num_times
    stop
  end if 
  if(num_heights > max_num_heights) then
    print *,   "Need space for more heights: ", num_heights, max_num_heights
    stop
  end if
  allocate(rho(num_heights)) 
  call HBUF_read(input_file_unit ,num_heights,'RHO', 1, 1, rho, num_read)
  print *,'.......', num_times
  print *, times(:num_times:3); print *
  print *, z    (:num_heights); print *
  print *, p    (:num_heights); print *
  print *, rho  (:num_heights)

  ! --------------------------------------------------------------------------
  ! Read the number of 1D variables and all the values at once
  !
  allocate(buffer_1d(num_times * num_vars),  &
           buffer_2d(num_heights, num_times), stat = status(1)) 
  if(status(1) /= 0) stop "Can't allocate enough memory to read variables" 
  
  call HBUF_parms(2, buffer_1d, num_params)
  if(num_params /= 8 .and. num_params /= num_vars) then
    print *,  "Expected either 8 or", num_vars, " variables but there are ", num_params
    stop
  end if 
  
  ! --------------------------------------------------------------------------
  ! Create netcdf file, define dimensions (time and height) 
  !
  status(:) = nf90_NoErr
  status(1) = nf90_create(filename(1:index(filename, '.stat')) // "nc", nf90_clobber, nc_file_id) 
  status(2) = nf90_put_att(nc_file_id, nf90_Global, 'model',  'CSU CEM version 1.0')
  status(3) = nf90_put_att(nc_file_id, nf90_Global, 'caseid', trim(caseid))
  
  status(4) = nf90_def_dim(nc_file_id, 'z',    num_heights, height_dim_id)
  status(5) = nf90_def_dim(nc_file_id, 'time', num_times,   time_dim_id)
  
  ! --------------------------------------------------------------------------
  ! Define each set of variables in turn: dimension, variables, derived variables, 2D
  !
  call define_1d_var(nc_file_id, 'z',    height_dim_id, 'height',   'm',   height_var_id)
  call define_1d_var(nc_file_id, 'p',    height_dim_id, 'pressure', 'mb',  p_var_id)
  call define_1d_var(nc_file_id, 'time', time_dim_id,   'time',     'day', time_var_id)
  
  !
  ! 1D variables are a function of time
  !
  do var = 1, num_vars 
    call define_1d_var(nc_file_id, abbr_names(var), time_dim_id, &
                       long_names(var), units(var),  nc_varIds(var))
  end do 
  do var = 1, num_derived_vars 
    call define_1d_var(nc_file_id, abbr_names_derived(var), time_dim_id, &
                       long_names_derived(var), units_derived(var),  nc_varIds_derived(var))
  end do 

  !
  ! Two D variable definitions are copied right from the binary file 
  ! 
  do var = 1, hbuf_length
    call define_MultiD_var(nc_file_id, name_list(var), (/ height_dim_id, time_dim_id /) , &
                           deflist(var), unitlist(var),  nc_varIds_2D(var))
 
  end do 
  status(6) = nf90_endDef(nc_file_id) 
  if(any(status(:6) /= nf90_NoErr)) stop "Error setting up netcdf file"
  print *, "Done defining netcdf file." 
  !
  ! At this point the file has been defined and we can start writing the variables
  ! --------------------------------------------------------------------------
  
  ! --------------------------------------------------------------------------
  ! Write dimension variables (two for height) 
  !
  status(1) = nf90_put_var(nc_file_id, height_var_id, z(      :num_heights))
  status(2) = nf90_put_var(nc_file_id, time_var_id,   times(  :num_times))
  status(3) = nf90_put_var(nc_file_id, p_var_id,      p(      :num_heights))
  if(any(status(:3) /= nf90_NoErr)) stop  "Error writing dimension variables (time, height)" 
  
  
  !
  ! 1D variables
  !
  do var = 1, num_params 
    status(var) = nf90_put_var(nc_file_id, nc_varIds(var), &
                               buffer_1d(var:var+num_vars*(num_times-1):num_vars))
  end do 
  if(any(status(:num_vars) /= nf90_NoErr)) stop  "Error writing 1-D variables" 
  print *, "Done writing 1D variables." 
  
  !
  ! Now the derived variables, which we'll calculate on the fly
  !
  do var = 1, num_derived_vars
    print *, "Writing derived variable " // trim(long_names_derived(var)) // "."
    select case(var)
      !
      ! Surface fluxes 
      !   For these we pull a 2D field and save the bottom level as a function of time
      !
      case(1) ! Surface precip
        call HBUF_read(input_file_unit, num_heights,'PRECIP', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_2d(1, :))
      case(2) ! Latent heat flux
        call HBUF_read(input_file_unit, num_heights,'QTFLUX', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_2d(1, :))
      case(3) ! Sensible heat flux
        call HBUF_read(input_file_unit, num_heights,'TLFLUX', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_2d(1, :))
      !
      ! Vertically-integrated water measures
      !   For these we vertically integrate air density times the mixiing ratio for each time
      !
      case(4) ! Precipitable water 
        call HBUF_read(input_file_unit, num_heights,'QV', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)) * 1.e-3)
      case(5) ! Observed precipitable water
        call HBUF_read(input_file_unit, num_heights,'QVOBS', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)) * 1.e-3)
      case(6) ! Cloud water path 
        call HBUF_read(input_file_unit, num_heights,'QC', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)))
      case(7) ! Ice water path 
        call HBUF_read(input_file_unit, num_heights,'QI', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)))
      case(8) ! Rain water path 
        call HBUF_read(input_file_unit, num_heights,'QR', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)))
      case(9) ! Snow water path 
        call HBUF_read(input_file_unit, num_heights,'QS', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)))
      case(10) ! Graupel water path 
        call HBUF_read(input_file_unit, num_heights,'QG', 1, num_times, buffer_2d, num_read)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var),                                        &
                                   matmul(rho(:num_heights-1) * (z(2:) - z(:num_heights-1)), &
                                          buffer_2d(:num_heights-1, :)))
      !
      ! Thermodynamic quantities - we call subroutines on the mean profiles one time step at a time
      !
      case(11) ! CAPE 
        allocate(buffer2_2d(num_heights, num_times))
        call HBUF_read(input_file_unit, num_heights,'TABS', 1, num_times, buffer_2d,  num_read)
        call HBUF_read(input_file_unit, num_heights,'QV',   1, num_times, buffer2_2d, num_read)
        do time = 1, num_times
          buffer_1d(time) = cape(num_heights, p(:num_heights), buffer_2d(:, time), buffer2_2d(:, time))
        end do 
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_1d(:num_times))
      case(12) ! CIN
               ! Use the temperature and water vapor profiles just read in 
        do time = 1, num_times
          buffer_1d(time) = cin(num_heights, p(:num_heights), buffer_2d(:, time), buffer2_2d(:, time))
        end do 
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_1d(:num_times))
      case(13) ! Observed CAPE
        call HBUF_read(input_file_unit, num_heights,'TABSOBS', 1, num_times, buffer_2d,  num_read)
        call HBUF_read(input_file_unit, num_heights,'QVOBS',   1, num_times, buffer2_2d, num_read)
        do time = 1, num_times
          buffer_1d(time) = cape(num_heights, p(:num_heights), buffer_2d(:, time), buffer2_2d(:, time))
        end do 
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_1d(:num_times))
      case(14) ! Observed CIN
               ! Use the temperature and water vapor profiles just read in 
        do time = 1, num_times
          buffer_1d(time) = cin(num_heights, p(:num_heights), buffer_2d(:, time), buffer2_2d(:, time))
        end do 
        deallocate(buffer2_2d)
        status(var) = nf90_put_var(nc_file_id, nc_varids_derived(var), buffer_1d(:num_times))
      case default 
        print *, "Don't know how to process derived variable " // trim(long_names_derived(var))
     end select
  end do 
  deallocate(rho, buffer_1d)
  if(any(status(:num_derived_vars) /= nf90_NoErr)) stop "Error writing derived variables" 
  print *, "Done writing derived variables." 

  !
  ! 2D variables, straight from the file
  !
  do var = 1, hbuf_length
    print *, "Writing ", trim(deflist(var)) // ' (' // trim(name_list(var)) // ')'
    call HBUF_read(input_file_unit, num_heights, trim(name_list(var)), 1, num_times, buffer_2d, num_read)
    status(var) = nf90_put_var(nc_file_id, nc_varIds_2d(var), buffer_2d) 
  end do 
  if(any(status(:hbuf_length) /= nf90_NoErr)) stop "Error writing 2-D variables" 
  print *, "Done writing 2D variables" 
  
  deallocate(buffer_2d)
  
  status(1) = nf90_close(nc_file_id)
  ! 
  ! --------------------------------------------------------------------------
contains
  ! --------------------------------------------------------------------------
  subroutine define_1d_var(nc_file_id, name, dim_id, long_name, units, var_id)
    integer,          intent( in) :: nc_file_id, dim_id
    character(len=*), intent( in) :: name, long_name, units
    integer,          intent(out) :: var_id 
    
    integer, dimension(3) :: status
    
    status(:) = 0 
    status(1) = nf90_def_var(nc_file_id, trim(name), nf90_float, dim_id, var_id)
    status(2) = nf90_put_att(nc_file_id, var_id, 'long_name', trim(long_name)) 
    status(3) = nf90_put_att(nc_file_id, var_id, 'units',     trim(units)) 
    if(any(status(:) /= nf90_NoErr))  print *,  "Error creating variable " // trim(name) 
    
  end subroutine define_1d_var
  ! --------------------------------------------------------------------------
  subroutine define_MultiD_var(nc_file_id, name, dim_ids, long_name, units, var_id)
    integer,               intent( in) :: nc_file_id
    integer, dimension(:), intent( in) :: dim_ids
    character(len=*),      intent( in) :: name, long_name, units
    integer,               intent(out) :: var_id 
    
    integer, dimension(3) :: status
    
    status(:) = 0 
    status(1) = nf90_def_var(nc_file_id, trim(name), nf90_float, dim_ids, var_id)
    status(2) = nf90_put_att(nc_file_id, var_id, 'long_name', trim(long_name)) 
    status(3) = nf90_put_att(nc_file_id, var_id, 'units',     trim(units)) 
    if(any(status(:) /= nf90_NoErr))  print *,  "Error creating variable " // trim(name) 
    
  end subroutine define_MultiD_var
end program statToNetcdf