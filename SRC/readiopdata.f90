!bloss #include <params.h>
!bloss #include <max.h>
!------------------------------------------------------------------------
! File: readiopdata.F 
! Author: John Truesdale (jet@ucar.edu) 
! $Id$
!
!------------------------------------------------------------------------
subroutine readiopdata( error_code )
  use vars, only: nsfc, daysfc, sstsfc, shsfc, lhsfc, tausfc, &
                  nsnd, nzsnd, daysnd, usnd, vsnd, tsnd, qsnd, psnd, zsnd, &
                  nlsf, nzlsf, dayls,  ugls, vgls, wgls, dqls, dtls, pls, zls, &
                  pres0ls
  use grid, only: nzm, z, case, masterproc, rank, & 
       dozero_out_day0, iopfile, wgls_holds_omega
  use params, only: rgas, cp, fac_cond, fac_sub, longitude0, latitude0, &
       SFC_FLX_FXD, dosubsidence
!-----------------------------------------------------------------------
!     
!     Open and read netCDF file containing initial IOP  conditions
!     
!---------------------------Code history--------------------------------
!     
!     Written by J.  Truesdale    August, 1996, revised January, 1998
!     
!     Modified by Peter Blossey (pblossey@u.washington.edu) July 2006
!       Reorganization and some changes that enable SAM to read 
!       initial soundings and forcings from SCAM IOP input datasets.
!
!     Modified by Peter Blossey (pblossey@u.washington.edu) March 2008
!       With SAM6.5, sounding and forcing data no longer need to be
!       interpolated to SAM vertical grid.  These changes should not
!       make much effect on results, but seem to clean up the code a bit.
!
!     Modified by Peter Blossey (pblossey@u.washington.edu) April 2008
!       For release in SAM6.7, have made further changes:
!         - added logical get_add_surface_data to eliminate
!             the appending of surface data to soundings/forcings.
!             If there is surface data in the netcdf file, it is read
!             in.  Otherwise, it is extrapolated.
!
!         - added input of CLDLIQ and CLDICE variables, in case cloud is
!              explicitly included in SCAM IOP initial condition.  Note
!              that the cloud liquid and ice is immediately added to water
!              vapor and the temperature is modified to account for the
!              latent heat release.  The cloud will arise with the
!              initial calls to the microphysics.
!
!     ===================================================================
!     NOTE: If the initial sounding has cloud, I (Peter) recommend using:
!             - Tli = Tabs - (L/Cp)*ql - (Ls/Cp)*qi in place of Tabs, and
!             - qtot = qv + ql + qi in place of qv
!       in the netcdf SCAM input file.  SAM will perform a saturation
!       adjustment of the initial profile that will convert the excess
!       vapor to cloud in the initial condition and give the correct
!       initial absolute temperature profile.
!     ===================================================================
!
!-----------------------------------------------------------------------
   implicit none

   include 'netcdf.inc'

!------------------------------Inputs-----------------------------------

   integer error_code        ! returns netcdf errors

!------------------------------Locals-----------------------------------
!     
   integer nlev, nlat, nlon, ntime, bdate, ierr, n

   ! dimensions
   integer, allocatable    :: tsec(:)
   real(4), allocatable       :: lon_in(:), lat_in(:), dplevs(:), & ! dimensions
                                 shf_in(:), lhf_in(:), Tg_in(:), &
                                 Ts_in(:), Ps_in(:), tmp_srf(:), &
                                 tausrf_in(:)  ! vars

   ! soundings, omega, advective tendencies (only function of time, lev here)
   real(4), allocatable :: u_in(:,:), v_in(:,:), omega_in(:,:), &
        T_in(:,:), divT_in(:,:), vertdivT_in(:,:), divT3d_in(:,:), &
        q_in(:,:), divq_in(:,:), vertdivq_in(:,:), divq3d_in(:,:), &
        ug_in(:,:), vg_in(:,:), cldliq_in(:,:), cldice_in(:,:)

   integer NCID, STATUS
   integer time_dimID, lev_dimID,  lev_varID, lat_dimID, lat_varID
   integer tsec_varID, bdate_varID, lon_dimID, lon_varID, varID

   integer k, m, kk, i,j, idummy, idummy2
   integer icldliq,icldice

   logical have_srf              ! value at surface is available
   logical use_nf_real           ! nctype for 4byte real
   logical fill_ends             ! 
   logical have_dcldliq,have_dcldice
   logical have_Tg, have_Tsair, have_shflx, have_lhflx, have_tausrf
   logical have_divq, have_vertdivq, have_divq3d, have_omega
   logical have_divT, have_vertdivT, have_divT3d, have_geostrophic_wind

   character(len=8) lowername
   character(len=120) iopfilepath

   real(4)    tmp_coef, coef, ratio_t1, ratio_t2, ratio1, ratio2
   real(4), parameter :: missing_value = -99999.

   logical :: get_add_surface_data
   integer :: nlev_in

! USE_4BYTE_REAL
   use_nf_real = .true.

   fill_ends= .false.
!     
!     Open IOP dataset
!     
   iopfilepath = './'//trim(case)//'/'//trim(iopfile) 
   if(masterproc) write(*,*) 'Opening ', iopfilepath
   STATUS = NF_OPEN( iopfilepath, NF_NOWRITE, NCID )
   if ( STATUS .NE. NF_NOERR ) then
      if(masterproc) write( 6,* ) &
           'ERROR(readiopdata.f90):Cant open iop dataset: ' ,iopfilepath
      call task_abort() 
   end if

!=====================================================================
!     Read time variables     
!
   call get_netcdf_dimlength(ncid, 'time', ntime, status, .false.)
   if ( STATUS .NE. NF_NOERR )  then
      call get_netcdf_dimlength(ncid, 'tsec', ntime, status, .true.)
      STATUS = NF_INQ_VARID( NCID, 'tsec', tsec_varID )
   else
      STATUS = NF_INQ_VARID( NCID, 'time', tsec_varID )
      if ( STATUS .NE. NF_NOERR ) then
         ! you might end up here if the dimension is called time, but the variable is tsec.
         STATUS = NF_INQ_VARID( NCID, 'tsec', tsec_varID )
      end if
   end if
   
   if ( STATUS .NE. NF_NOERR ) then
     write( 6,* )'ERROR(readiopdata.f90):Could not get variable ID for tsec'
     STATUS = NF_CLOSE ( NCID )
     call task_abort()
   end if

   ALLOCATE(tsec(ntime),STAT=status)
   if(status.ne.0) then
      write(6,*) 'Could not allocate tsec in readiopdata'
      call task_abort()
   end if

   STATUS = NF_GET_VAR_INT( NCID, tsec_varID, tsec )
   if ( STATUS .NE. NF_NOERR ) then
     write( 6,* )'ERROR(readiopdata.f90):Could not read variable tsec'
     STATUS = NF_CLOSE ( NCID )
     call task_abort()
   end if

   STATUS = NF_INQ_VARID( NCID, 'nbdate', bdate_varID )
   if ( STATUS .NE. NF_NOERR ) then
      STATUS = NF_INQ_VARID( NCID, 'bdate', bdate_varID )
      if ( STATUS .NE. NF_NOERR ) then
         write( 6,* )'ERROR(readiopdata.f90):Could not find variable ID for bdate'
         STATUS = NF_CLOSE ( NCID )
         call task_abort()
      end if
   end if

   STATUS = NF_GET_VAR_INT( NCID, bdate_varID, bdate )
   if ( STATUS .NE. NF_NOERR )then
     write( 6,* )'ERROR(readiopdata.f90):Could not find variable bdate'
     STATUS = NF_CLOSE ( NCID )
     call task_abort()
   end if

!     
!======================================================
!     read level data
!     
   call get_netcdf_dimlength(ncid, 'lev', nlev, status, .true.)

   ALLOCATE(dplevs(nlev+1),STAT=status)
   if(status.ne.0) then
      write(6,*) 'Could not allocate dplevs in readiopdata'
      call task_abort()
   end if

   ! get pressure levels (in Pascal)
   call get_netcdf_var1d_real( NCID, 'lev', dplevs, use_nf_real, status,.true. )

!     
!======================================================
!     read lat/lon data
!     
   call get_netcdf_dimlength(ncid, 'lat', nlat, status, .true.)
   call get_netcdf_dimlength(ncid, 'lon', nlon, status, .true.)

   ALLOCATE(lat_in(nlat),lon_in(nlon),STAT=status)
   if(status.ne.0) then
      write(6,*) 'Could not allocate lat/lon in readiopdata'
      call task_abort()
   end if

   ! get latitude
   call get_netcdf_var1d_real( NCID, 'lat',lat_in,use_nf_real,status,.false. )

   ! get longitude
   call get_netcdf_var1d_real( NCID, 'lon',lon_in,use_nf_real,status,.false. )

! fill SAM's lat and lon variables
   longitude0 = lon_in(1)
   latitude0 = lat_in(1)
!
!======================================================
!     allocate surface variables
!     
   ALLOCATE(shf_in(ntime), lhf_in(ntime), Tg_in(ntime), Ts_in(ntime), &
        Ps_in(ntime), tmp_srf(ntime), tausrf_in(ntime), &
        STAT=status)

   shf_in(:) = missing_value
   lhf_in(:) = missing_value
   Tg_in(:) = missing_value
   Ts_in(:) = missing_value
   Ps_in(:) = missing_value
   tausrf_in(:) = missing_value

   if(status.ne.0) then
      write(6,*) 'Could not allocate surface variables in readiopdata'
      call task_abort()
   end if
!
!======================================================
!     read surface variables
!     
! note that the last argument is whether the run should die if the variable
!   is not present in the netcdf file.
!
   ! surface air temperature
   call get_netcdf_var1d_real( ncid, 'Tsair', Ts_in,use_nf_real,status,.false.)
   have_tsair = .true.
   if ( STATUS .NE. NF_NOERR ) have_tsair = .false.

   ! ground/sea surface temperature
   call get_netcdf_var1d_real( ncid, 'Tg', Tg_in, use_nf_real, status,.false.)
   have_tg = .true.
   if ( STATUS .NE. NF_NOERR ) have_tg = .false.

   ! sensible heat flux
   call get_netcdf_var1d_real( ncid, 'shflx', shf_in, use_nf_real, &
        status,.false.)
   have_shflx = .true.
   if ( STATUS .NE. NF_NOERR ) then
      ! old name - backwards compatibility
      call get_netcdf_var1d_real( ncid, 'sh', shf_in, use_nf_real, &
           status,.false.)
      if ( STATUS .NE. NF_NOERR ) have_shflx = .false.
   end if

   ! latent heat flux
   call get_netcdf_var1d_real( ncid, 'lhflx', lhf_in, use_nf_real, &
        status,.false.)
   have_lhflx = .true.
   if ( STATUS .NE. NF_NOERR ) then
      ! old name - backwards compatibility
      call get_netcdf_var1d_real( ncid, 'lh', lhf_in, use_nf_real, &
           status,.false.)
      if ( STATUS .NE. NF_NOERR ) have_lhflx = .false.
   end if

   ! abort if surface fluxes are required, but are not present in netcdf file
   if(SFC_FLX_FXD.and.(.NOT.have_lhflx.OR..NOT.have_shflx)) then
      if(masterproc) then
         write(*,*) 'ERROR(readiopdata.f90): If SFC_FLX_FXD is true, '
         write(*,*) '          shflx and lhflx needed in SCAM iop netcdf file.'
      end if
      call task_abort()
   end if

   ! surface pressure
   call get_netcdf_var1d_real( ncid, 'Ps', Ps_in, use_nf_real, status, .true.)
!         
!====================================================================
!     check whether surface pressure exceeds largest pressure
!       in pressure sounding (dplevs)
!     
   if(MINVAL(Ps_in).le.MAXVAL(dplevs(1:nlev))) then
      ! Surface pressure is included in dplevs.
      ! Do not bother with adding surface data to soundings/forcings.
      get_add_surface_data = .false.
      ! do not leave room for surface data in input datasets
      nlev_in = nlev
   else
      ! Surface pressure exceeds max pressure in dplevs
      ! Get/add surface data to soundings/forcings.
      get_add_surface_data = .true.
      ! leave room for surface data in input datasets
      nlev_in = nlev+1
   end if

   !convert pressures to millibar
   dplevs(1:nlev) = dplevs(1:nlev)/100.
   Ps_in(:) = Ps_in(:)/100. ! convert to millibar
!         
!====================================================================
!     allocate variables with pressure and time dependence (q,T,etc.)
!     
   Allocate(T_in(ntime,nlev_in), q_in(ntime,nlev_in), &
        divT_in(ntime,nlev_in), divq_in(ntime,nlev_in), &
        divT3d_in(ntime,nlev_in), divq3d_in(ntime,nlev_in), &
        vertdivT_in(ntime,nlev_in), vertdivq_in(ntime,nlev_in), &
        u_in(ntime,nlev_in), v_in(ntime,nlev_in), &
        ug_in(ntime,nlev_in), vg_in(ntime,nlev_in), &
        omega_in(ntime,nlev_in), cldliq_in(ntime,nlev_in), &
        cldice_in(ntime,nlev_in), STAT=status)

   if(status.ne.0) then
      write(6,*) 'Could not allocate surface variables in readiopdata'
      call task_abort()
   end if
!
!====================================================================
!     read variables with pressure and time dependence (q,T,etc.)
!     
   ! Temperature
   T_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'T',ntime,nlev,T_in, &
        use_nf_real,status,.true.)

   ! Horizontal Advective Temperature Forcing
   divT_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'divT',ntime,nlev,divT_in, &
        use_nf_real,status,.false.)
   have_divT = .true.
   if( STATUS .NE. NF_NOERR ) have_divT = .false.

   ! Vertical Advective Temperature Forcing
   vertdivT_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'vertdivT',ntime,nlev,vertdivT_in, &
        use_nf_real,status,.false.)
   have_vertdivT = .true.
   if( STATUS .NE. NF_NOERR ) have_vertdivT = .false.

   ! Three-dimensional Advective Temperature Forcing
   divT3d_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'divT3d',ntime,nlev,divT3d_in, &
        use_nf_real,status,.false.)
   have_divT3d = .true.
   if( STATUS .NE. NF_NOERR ) have_divT3d = .false.

   !==================
   ! Moisture
   q_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'q',ntime,nlev,q_in,use_nf_real,status,.true.)

   cldliq_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'CLDLIQ',ntime,nlev,cldliq_in,use_nf_real,status,.false.)
   if( STATUS .EQ. NF_NOERR ) then
      ! if CLDLIQ is present, add cloud liquid water to water vapor and
      !   modify initial temperature to reflect release of latent heat.
      !   SAM does not support a specified initial cloud layer at this
      !   point.  The initial cloud will arise with the first saturation adjustment.
      q_in(:,1:nlev) = q_in(:,1:nlev) + cldliq_in(:,1:nlev)
      t_in(:,1:nlev) = t_in(:,1:nlev) - fac_cond*cldliq_in(:,1:nlev)
   end if

   cldice_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'CLDICE',ntime,nlev,cldice_in,use_nf_real,status,.false.)
   if( STATUS .EQ. NF_NOERR ) then
      ! if CLDICE is present, add cloud ice water to water vapor 
      !   modify initial temperature to reflect release of latent heat
      !   SAM does not support a specified initial cloud layer at this
      !   point.  The initial cloud will arise with the first saturation adjustment.
      q_in(:,1:nlev) = q_in(:,1:nlev) + cldice_in(:,1:nlev)
      t_in(:,1:nlev) = t_in(:,1:nlev) - fac_sub*cldice_in(:,1:nlev)
   end if

   ! Horizontal Advective Moisture Forcing
   divq_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'divq',ntime,nlev,divq_in, &
        use_nf_real,status,.false.)
   have_divq = .true.
   if( STATUS .NE. NF_NOERR ) have_divq = .false.

   ! Vertical Advective Moisture Forcing
   vertdivq_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'vertdivq',ntime,nlev,vertdivq_in, &
        use_nf_real,status,.false.)
   have_vertdivq = .true.
   if( STATUS .NE. NF_NOERR ) have_vertdivq = .false.

   ! Three-dimensional Advective Moisture Forcing
   divq3d_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'divq3d',ntime,nlev,divq3d_in, &
        use_nf_real,status,.false.)
   have_divq3d = .true.
   if( STATUS .NE. NF_NOERR ) have_divq3d = .false.

   !==================================
   ! omega: vertical pressure velocity
   omega_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'omega',ntime,nlev,omega_in, &
        use_nf_real,status,.false.)
   have_omega = .true.
   if( STATUS .NE. NF_NOERR ) have_omega = .false.

   !==================================
   ! horizontal wind
   u_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'u',ntime,nlev,u_in, &
        use_nf_real,status,.true.)

   v_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'v',ntime,nlev,v_in, &
        use_nf_real,status,.true.)

   !==================================
   ! geostrophic wind (not native to SCAM, but useful in SAM)
   ug_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'ug',ntime,nlev,ug_in, &
        use_nf_real,status,.false.)
   have_geostrophic_wind = .true.
   if( STATUS .NE. NF_NOERR ) have_geostrophic_wind = .false.

   vg_in(:,:) = missing_value
   call get_netcdf_var2d_real( ncid,'vg',ntime,nlev,vg_in, &
        use_nf_real,status,.false.)
   if( STATUS .NE. NF_NOERR ) have_geostrophic_wind = .false.

   !==================================
   !==================================
   ! READ IN SURFACE DATA AND PUT INTO FORCINGS/SOUNDINGS IF SURFACE
   !   PRESSURE IS BIGGER THAN MAX PRESSURE IN SOUNDING.
   if(get_add_surface_data) then
      ! temperature
      if(have_tsair) T_in(:,nlev+1) = Ts_in(:)

      ! temperature forcing
      call get_netcdf_var1d_real( ncid,'divTsrf',tmp_srf, &
           use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) divT_in(:,nlev+1) = tmp_srf(:)

      call get_netcdf_var1d_real(ncid,'vertdivTsrf', &
           tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) vertdivT_in(:,nlev+1) = tmp_srf(:)

      call get_netcdf_var1d_real(ncid,'divT3dsrf', &
           tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) divT3d_in(:,nlev+1) = tmp_srf(:)

      ! moisture
      call get_netcdf_var1d_real( ncid,'qsrf',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) q_in(:,nlev+1) = tmp_srf(:)

      ! moisture forcing
      call get_netcdf_var1d_real( ncid,'divqsrf',tmp_srf, &
           use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) divq_in(:,nlev+1) = tmp_srf(:)

      call get_netcdf_var1d_real(ncid,'vertdivqsrf', &
           tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) vertdivq_in(:,nlev+1) = tmp_srf(:)

      ! get surface data if available
      call get_netcdf_var1d_real(ncid,'divq3dsrf', &
           tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) divq3d_in(:,nlev+1) = tmp_srf(:)

      ! surface pressure tendency --> surface omega
      call get_netcdf_var1d_real(ncid,'Ptend',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) omega_in(:,nlev+1) = tmp_srf(:)

      ! winds
      call get_netcdf_var1d_real(ncid,'usrf',tmp_srf,use_nf_real,status,.false.)
      have_tausrf = .false.
      if( STATUS .EQ. NF_NOERR ) then
         u_in(:,nlev+1) = tmp_srf(:)
         tausrf_in(:) = tmp_srf(:)**2
         have_tausrf = .true.
      end if

      call get_netcdf_var1d_real(ncid,'vsrf',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) then
         v_in(:,nlev+1) = tmp_srf(:)
         if(have_tausrf) then
            tausrf_in(:) = tausrf_in(:) + tmp_srf(:)**2
         end if
      else
         have_tausrf = .false.
      end if

      call get_netcdf_var1d_real(ncid,'ugsrf',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) ug_in(:,nlev+1) = tmp_srf(:)

      call get_netcdf_var1d_real(ncid,'vgsrf',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) vg_in(:,nlev+1) = tmp_srf(:)

   else

      ! get surface data for winds regardless
      call get_netcdf_var1d_real(ncid,'usrf',tmp_srf,use_nf_real,status,.false.)
      have_tausrf = .false.
      if( STATUS .EQ. NF_NOERR ) then
         tausrf_in(:) = tmp_srf(:)**2
         have_tausrf = .true.
      end if

      call get_netcdf_var1d_real(ncid,'vsrf',tmp_srf,use_nf_real,status,.false.)
      if( STATUS .EQ. NF_NOERR ) then
         if(have_tausrf) then
            tausrf_in(:) = tausrf_in(:) + tmp_srf(:)**2
         end if
      else
         have_tausrf = .false.
      end if

   end if ! if(get_add_surface_data)

   !=========================================
   ! fix ground temperature if not in nc file
   if(.not.have_tg) then
      if ( have_tsair ) then
         write(6,*) 'In readiopdata(): Using Tsair for Tground'
         Tg_in = Ts_in(:)
      else
         write(6,*) 'In readiopdata(): Using lowest level T for Tground'
         Tg_in = T_in(:,nlev)
      end if
   end if

   !=========================================
   !bloss: We are ignoring the following variables that could appear
   ! in SCAM IOP netcdf input files:
   !
   ! 'cld' -- ????soundings of cloud fraction????
   ! 'clwp' -- ????soundings of cloud liquid water????
   ! 'CLDLIQ', 'dcldliq' -- large-scale tendency of cloud water
   ! 'CLDICE', 'dcldice' -- large-scale tendency of cloud ice
   ! 'divu', 'divusrf' -- large-scale tendency of u
   ! 'divv', 'divvsrf' -- large-scale forcing of v
   

   STATUS = NF_CLOSE( NCID )
   error_code = 0

   !bloss (10Mar2008): Modify to handle SAM's new forcing setup (as of ~6.5).
   !      This means that sounding and forcing data can sit on its own
   !      grid, rather than needing to be interpolated to the model grid.
   nsnd = ntime
   nzsnd = nlev_in
   nlsf = ntime
   nzlsf = nlev_in
   nsfc = ntime
   if(masterproc) print*,'sounding data: nsnd=',nsnd,'  nzsnd=',nzsnd
   if(masterproc)print*,'forcing data: nlsf=',nlsf,'  nzlsf=',nzlsf
   if(masterproc)print*,'surface forcing data: nsfc=',nsfc
   allocate(usnd(nzsnd,nsnd),vsnd(nzsnd,nsnd), &
        tsnd(nzsnd,nsnd),qsnd(nzsnd,nsnd), &
        zsnd(nzsnd,nsnd),psnd(nzsnd,nsnd),daysnd(nsnd), &
        ugls(nzlsf,nlsf),vgls(nzlsf,nlsf), wgls(nzlsf,nlsf), &
        dtls(nzlsf,nlsf),dqls(nzlsf,nlsf), &
        zls(nzlsf,nlsf),pls(nzlsf,nlsf),pres0ls(nlsf),dayls(nlsf),&
        daysfc(nsfc),sstsfc(nsfc),shsfc(nsfc),lhsfc(nsfc),&
        tausfc(nsfc), STAT=ierr)
   if(ierr.NE.0) then
      if(masterproc) then
         write(*,*) 'Error in allocating snd/lsf/rad/sfc vars in readiopdata'
      end if
      call task_abort()
   end if

   !bloss: save timeseries into SAM's snd/lsf/sfc/rad arrays
   do i = 1,ntime
      ! set times for snd, lsf, sfc
      call calcdate(bdate,tsec(i),idummy,idummy2,daysnd(i))
      daysfc(i) = daysnd(i)
      dayls(i) = daysnd(i)

      ! set surface forcing timeseries
      pres0ls(i) = Ps_in(i)
      sstsfc(i) = Tg_in(i)
      shsfc(i)   = shf_in(i)
      lhsfc(i)  = lhf_in(i)
      if(have_tausrf) then
         tausfc(i) = sqrt(tausrf_in(i)) !!!!! ????????FIX THIS?????? !!!!!
      else
         tausfc(i) = 0.
      end if
   end do

   if(dozero_out_day0) then
      daysnd(:) = daysnd(:) - daysnd(1)
      daysfc(:) = daysfc(:) - daysfc(1)
      dayls(:)  = dayls(:)  - dayls(1)
   end if

   ! change absolute temperature sounding to potential temperature
   do i = 1,ntime
      do n = 1,nlev
         T_in(i,n) = T_in(i,n)*(1000./dplevs(n))**(rgas/cp) ! T --> theta
      end do
   end do

   if(get_add_surface_data) then
      ! either use surface data from netcdf file or interpolate it to surface.

      ! fix surface theta
      do i = 1,ntime
         if(T_in(i,nlev+1).ne.missing_value) then
            ! set iop dataset surface pressure for this time
            n = nlev+1
            dplevs(n) = Ps_in(i) 
            T_in(i,n) = T_in(i,n)*(1000./dplevs(n))**(rgas/cp) ! T --> theta
         end if
      end do

      ! interpolate to fill surface data/forcings if necessary
      do i = 1,ntime

         ! extrapolate to surface if iop data at surface is not available.
         dplevs(nlev+1) = Ps_in(i) 
         coef = (dplevs(nlev+1)-dplevs(nlev-1))/(dplevs(nlev)-dplevs(nlev-1))

         ! fill in surface sounding data if missing
         if(T_in(i,nlev+1).eq.missing_value) &
              T_in(i,nlev+1) = (1-coef)*T_in(i,nlev-1) + coef*T_in(i,nlev)
         if(q_in(i,nlev+1).eq.missing_value) &
              q_in(i,nlev+1) = (1-coef)*q_in(i,nlev-1) + coef*q_in(i,nlev)
         if(u_in(i,nlev+1).eq.missing_value) &
              u_in(i,nlev+1) = (1-coef)*u_in(i,nlev-1) + coef*u_in(i,nlev)
         if(v_in(i,nlev+1).eq.missing_value) &
              v_in(i,nlev+1) = (1-coef)*v_in(i,nlev-1) + coef*v_in(i,nlev)

         ! fill in surface large-scale forcing data if missing
         if(divT_in(i,nlev+1).eq.missing_value) divT_in(i,nlev+1) = &
              (1-coef)*divT_in(i,nlev-1) + coef*divT_in(i,nlev)
         if(vertdivT_in(i,nlev+1).eq.missing_value) vertdivT_in(i,nlev+1) = &
              (1-coef)*vertdivT_in(i,nlev-1) + coef*vertdivT_in(i,nlev)
         if(divT3d_in(i,nlev+1).eq.missing_value) divT3d_in(i,nlev+1) = &
              (1-coef)*divT3d_in(i,nlev-1) + coef*divT3d_in(i,nlev)

         if(divq_in(i,nlev+1).eq.missing_value) divq_in(i,nlev+1) = &
              (1-coef)*divq_in(i,nlev-1) + coef*divq_in(i,nlev)
         if(vertdivq_in(i,nlev+1).eq.missing_value) vertdivq_in(i,nlev+1) = &
              (1-coef)*vertdivq_in(i,nlev-1) + coef*vertdivq_in(i,nlev)
         if(divq3d_in(i,nlev+1).eq.missing_value) divq3d_in(i,nlev+1) = &
              (1-coef)*divq3d_in(i,nlev-1) + coef*divq3d_in(i,nlev)

         if(omega_in(i,nlev+1).eq.missing_value) omega_in(i,nlev+1) = &
              (1-coef)*omega_in(i,nlev-1) + coef*omega_in(i,nlev)
         if(ug_in(i,nlev+1).eq.missing_value) &
              ug_in(i,nlev+1) = (1-coef)*ug_in(i,nlev-1) + coef*ug_in(i,nlev)
         if(vg_in(i,nlev+1).eq.missing_value) &
              vg_in(i,nlev+1) = (1-coef)*vg_in(i,nlev-1) + coef*vg_in(i,nlev)
      end do

   end if !if(get_add_surface_data)

   ! now fill in profiles for snd
   zsnd(:,:) = -999.
   do i = 1,nsnd
      psnd(1,i) = Ps_in(i)
      psnd(2:nzsnd,i) = dplevs(nzsnd-1:1:-1)
      tsnd(1:nzsnd,i) = T_in(i,nzsnd:1:-1)
      qsnd(1:nzsnd,i) = q_in(i,nzsnd:1:-1)
      usnd(1:nzsnd,i) = u_in(i,nzsnd:1:-1) 
      vsnd(1:nzsnd,i) = v_in(i,nzsnd:1:-1) 
   end do

   ! convert qsnd to g/kg to be consistent with Marat's implementation
   qsnd(:,:) = 1.e3*qsnd(:,:)

   ! now fill in profiles for lsf
   zls(:,:) = -999.
   do i = 1,nlsf
      pres0ls(i) = Ps_in(i)
      pls(1,i) = Ps_in(i)
      pls(2:nzlsf,i) = dplevs(nzlsf-1:1:-1)
   end do

   dtls(:,:) = 0.
   dqls(:,:) = 0.
   wgls(:,:) = 0.

   if(have_omega) then
      dosubsidence = .true.

      ! use omega for large-scale vertical advection if it exists
      do i = 1,nlsf
         ! NOTE: Here, we are putting omega into wgls (large-scale w)
         ! THIS WILL BE CONVERTED INTO w IN forcing()
         wgls(1:nzlsf,i) = omega_in(i,nzlsf:1:-1)
      end do
      wgls_holds_omega = .true.

      ! use large-scale horizontal advection w/omega if it exists.
      if(have_divT) then
         do i = 1,nlsf
            dtls(1:nzlsf,i) = divT_in(i,nzlsf:1:-1)
         end do
      end if
      if(have_divq) then
         do i = 1,nlsf
            dqls(1:nzlsf,i) = divq_in(i,nzlsf:1:-1)
         end do
      end if

   else

      ! if no omega in dataset, use 3d or vert+horiz forcing.
      dosubsidence = .false.

      if(have_divT3d) then
         do i = 1,nlsf
            dtls(1:nzlsf,i) = divT3d_in(i,nzlsf:1:-1)
         end do
      elseif(have_vertdivT.and.have_divT) then
         do i = 1,nlsf
            dtls(1:nzlsf,i) = divT_in(i,nzlsf:1:-1) + vertdivT_in(i,nzlsf:1:-1)
         end do
      elseif(have_divT) then
         do i = 1,nlsf
            dtls(1:nzlsf,i) = divT_in(i,nzlsf:1:-1)
         end do
      end if

      if(have_divq3d) then
         do i = 1,nlsf
            dqls(1:nzlsf,i) = divq3d_in(i,nzlsf:1:-1)
         end do
      elseif(have_vertdivq.and.have_divq) then
         do i = 1,nlsf
            dqls(1:nzlsf,i) = divq_in(i,nzlsf:1:-1) + vertdivq_in(i,nzlsf:1:-1)
         end do
      elseif(have_divq) then
         do i = 1,nlsf
            dqls(1:nzlsf,i) = divq_in(i,nzlsf:1:-1)
         end do
      end if

   end if

   if(have_geostrophic_wind) then
      do i = 1,nlsf
         ugls(1:nzlsf,i) = ug_in(i,nzlsf:1:-1)
         vgls(1:nzlsf,i) = vg_in(i,nzlsf:1:-1)
      end do
   else
      do i = 1,nlsf ! use wind sounding as geostrophic wind
         ugls(1:nzlsf,i) = u_in(i,nzlsf:1:-1)
         vgls(1:nzlsf,i) = v_in(i,nzlsf:1:-1)
      end do
   end if

   !set up sfc stuff (surface forcings)
   do i = 1,ntime
      sstsfc(i) = Tg_in(i)
      shsfc(i)   = shf_in(i)
      lhsfc(i)  = lhf_in(i)
      if(have_tausrf) then
         tausfc(i) = sqrt(tausrf_in(i)) !!!!! FIX THIS !!!!!
      else
         tausfc(i) = 0.
      end if
   end do

   if(ntime.eq.1) then
      if(masterproc) print*,'Error: minimum two sounding profiles are needed.'
      call task_abort()
   endif

   if(masterproc) print*,'Observed sounding interval (days):', &
        daysnd(1),daysnd(ntime)


   ! deallocate all of the variables still on the iop grid
   deallocate(tsec,dplevs,lat_in,lon_in,STAT=status)
   if(status.ne.0) then
      write(6,*) 'Processor ', rank, &
           'Could not de-allocate dimensions in readiopdata'
      call task_abort()
   end if

   deallocate(shf_in,lhf_in,Tg_in,Ts_in,Ps_in,tmp_srf,STAT=status)
   if(status.ne.0) then
      write(6,*) 'Processor ', rank, &
           'Could not de-allocate surface data arrays in readiopdata'
      call task_abort()
   end if

   deallocate(T_in,q_in,divT_in,divT3d_in,vertdivT_in, &
        u_in,ug_in,omega_in,vg_in,v_in, &
        vertdivq_in,divq3d_in,divq_in,STAT=status)
   if(status.ne.0) then
      write(6,*) 'Processor ', rank, &
           'Could not de-allocate sounding/forcing arrays in readiopdata'
      call task_abort()
   end if

   return
contains
  !=====================================================================
  subroutine get_netcdf_dimlength( NCID, dimName, dimlength, status, required)
    implicit none
    include 'netcdf.inc'

    ! input/output variables
    integer, intent(in)   :: NCID
    character, intent(in) :: dimName*(*)
    logical, intent(in) :: required

    integer, intent(out) :: status, dimlength

    ! local variables
    integer :: dimID

    ! get variable ID
    STATUS = NF_INQ_DIMID( NCID, dimName, dimID )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find dimension ID for ', &
               dimName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note(readiopdata.f90): No dimension ID for ', dimName
          return
       endif
    endif

    STATUS = NF_INQ_DIMLEN( NCID, dimID, dimlength )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find length of ',dimName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note - readiopdata.f90 : Could not find length of ',&
               dimName
       endif
    endif

  end subroutine get_netcdf_dimlength
  !=====================================================================
  subroutine get_netcdf_var1d_real( NCID, varName, var, use_nf_real, &
       status, required)
    implicit none
    include 'netcdf.inc'

    ! input/output variables
    integer, intent(in)   :: NCID
    character, intent(in) :: varName*(*)
    logical, intent(in) :: required, use_nf_real

    integer, intent(out) :: status
    real(4), intent(inout) :: var(:)

    ! local variables
    integer :: varID

    ! get variable ID
    STATUS = NF_INQ_VARID( NCID, varName, varID )
    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ID for ', &
               varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note(readiopdata.f90): Optional variable ', varName,&
               ' not found in ', TRIM(iopfile)
          return
       endif
    endif

    if (use_nf_real) then
       STATUS = NF_GET_VAR_REAL( NCID, varID, var )
    else
       STATUS = NF_GET_VAR_DOUBLE( NCID, varID, var )
    endif

    if (STATUS .NE. NF_NOERR ) then
       if(required) then
          if(masterproc) write(6,*) &
               'ERROR(readiopdata.f90):Could not find variable ', varName
          STATUS = NF_CLOSE( NCID )
          call task_abort()
       else
          if(masterproc) write(6,*) &
               'Note (readiopdata.f90) : Could not find ', varName
       endif
    endif

  end subroutine get_netcdf_var1d_real
  !=====================================================================
  subroutine get_netcdf_var2d_real( NCID, varName, ntime, nlev, &
       var, use_nf_real, status, required)
    !based on John Truesdale's getncdata_real_1d
    implicit none
    include 'netcdf.inc'

    ! input/output variables
    integer, intent(in)   :: NCID, ntime, nlev
    character, intent(in) :: varName*(*)
    logical, intent(in) :: required, USE_NF_REAL

    integer, intent(out) :: status
    real(4), intent(inout) :: var(:,:)

    ! local variables
    integer :: varID
    character     dim_name*( NF_MAX_NAME )
    integer     var_dimIDs( NF_MAX_VAR_DIMS )
    integer     start(5 ), count(5 )
    integer     var_ndims, dim_size, dims_set, i, n, var_type
    logical usable_var

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
               ' not found in ', TRIM(iopfile)
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
   do n = 1,nlev

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
         if ( dim_name .EQ. 'lat' ) then
            start( i ) = 1
            count( i ) = 1
            dims_set = dims_set + 1
            usable_var = .true.
         endif

         ! extract the single longitude in the iop file
         if ( dim_name .EQ. 'lon' ) then
            start( i ) = 1
            count( i ) = 1          
            dims_set = dims_set + 1
            usable_var = .true.
         endif

         ! extract all times at this level
         if ( dim_name .EQ. 'time' .OR. dim_name .EQ. 'tsec'  ) then
            start( i ) = 1
            count( i ) = ntime      ! Extract a single value 
            dims_set = dims_set + 1   
            usable_var = .true.
         endif

         ! extract one level with each call
         if ( dim_name .EQ. 'lev' ) then
            start( i ) = n
            count( i ) = 1
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
         STATUS = NF_GET_VARA_REAL( NCID, varID, start, count, var(1,n) )
      else   
         STATUS = NF_GET_VARA_DOUBLE( NCID, varID, start, count, var(1,n) )
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

   end do

 end subroutine get_netcdf_var2d_real

end subroutine readiopdata

!------------------------------------------------------------------------
! File: calcdate.F 
! Author: John Truesdale (jet@ucar.edu) 
! $Id$
!
! Modified by Peter Blossey (pblossey@u.washington.edu)
!    to handle leap years and output calendar day.
!------------------------------------------------------------------------
!bloss #include <params.h>
subroutine calcdate(inDate, inSecs,  outDate, outSecs, outCalday)
!-----------------------------------------------------------------------
!  calcdate           Calculate Date from base date plus seconds
!
! INPUTS:
!
!	inDate	       Base date as YYMMDD.
!       inSecs         number of seconds the model has run
!
! OUTPUTS:
!       outDate        Current date as YYMMDD
!       outSecs        number of seconds into current date
!       outCalday      calendar day of current year (Jan-01 00:00:00 == 1.)
!
!
!-----------------------------------------------------------------------
! Computational notes: 
!
! 86400 is the number of seconds in 1 day.
!
! Dividing an integer by 10**n has the effect of right-shifting the           
! decimal digits n positions (ex: 861231/100 = 008612).
!
! mod(integer,10**n) has the effect of extracting the rightmost     
! n decimal digits of the integer (ex: mod(861231,10000) = 1231).
!
   implicit none
!------------------------------Arguments--------------------------------
!
! Input arguments
!
   integer, intent(in) :: inDate
   integer, intent(in) :: inSecs       

!         
! Output arguments       
!                        
   integer, intent(out) :: outSecs
   integer, intent(out) :: outDate
   real, intent(out) :: outCalday
!
!---------------------------Local workspace-----------------------------
!
   integer     YY
   integer     MM
   integer     DD
   integer     i
   integer     iyear !bloss: added to deal with leap years
   integer     byear !bloss: added to deal with leap years
   integer     bmnth
   integer     bday
   integer     jday
   integer     jsec
   integer     jdcon(12)
   integer     ndm(12)
   integer     secs_per_year
   integer     secs_this_year  !bloss: added to deal with leap years
   integer     days_this_month  !bloss: added to deal with leap years

   data ndm/31,28,31,30,31,30,31,31,30,31,30,31/
   data jdcon/0,31,59,90,120,151,181,212,243,273,304,334/
!
!-----------------------------------------------------------------------
!
! Check validity of input data
!
   byear = inDate/10000 
   if(byear.lt.100) byear = 1900+byear !bloss: if 2 digit year, add 1900
   bmnth = mod(inDate,10000)/100
   bday =  mod(inDate,100)
   if (bmnth.lt.1 .or. bmnth.gt.12) then
      write(6,*)' CALCDATE: Invalid base month input:',bmnth
      call task_abort()
   end if
   if (bday.lt.1 .or. bday.gt.ndm(bmnth)) then
      write(6,*)' CALCDATE: Invalid base day of base date input:',bday
      call task_abort()
   end if
!
!
!
   jday = jdcon(bmnth) + bday

   !bloss: add a day if past February and this is a leap year.
   if( (bmnth.gt.2).and. isLeapYear(byear) ) jday = jday + 1

   jsec  = (jday-1) * 86400 + insecs

   secs_per_year = 86400 * 365

   !bloss: count through years until jsec is less than a year.
   do iyear = byear,byear+insecs/secs_per_year
      secs_this_year = secs_per_year
      if(isLeapYear(iyear)) secs_this_year = secs_this_year + 86400 ! leap day
      if(jsec.lt.secs_this_year) EXIT ! break from loop is jsec < one year
      jsec = jsec - secs_this_year
   end do

   YY  = mod(iyear,100) !bloss reduce year to two digit YY -- breaks 19YY

   outCalday = 1. + float(jsec)/86400. !bloss: compute calendar day

   !bloss: count through months until jsec is less than the next month
   do i=1, 12
      MM = i
      days_this_month = ndm(i)
      if(i.eq.2.and.isLeapYear(iyear)) days_this_month = days_this_month + 1
      if(jsec.lt.86400*days_this_month) EXIT
      jsec = jsec - 86400*days_this_month
   end do

   DD = jsec/86400 +1

   outSecs = mod(jsec,86400)

   outDate = YY*10000 + MM*100 + DD

!      write( *,* )'date =' , outDate
!
   return

contains

  logical function isLeapYear(iyear)
    implicit none
    integer iyear

    isLeapYear = mod(iyear,4).eq.0.and.mod(iyear,100).ne.0 &
                          .or.mod(iyear,400).eq.0
  end function isLeapYear

end subroutine calcdate
