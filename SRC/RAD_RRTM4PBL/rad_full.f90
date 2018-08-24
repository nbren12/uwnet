SUBROUTINE rad_full()
	
!++++++++++++++TAK: PBL interface for RRTM based on P. Blossey's GCSS-CFMIP interface+++++++++++++
! (1) The USEd parameters were organized for SAM6.10.4.
! (2) Adjustment for SAM6.10.4: t00(=300K) is added to sstxy (SST perturbation) for tg_slice.
! (3) Following RAD_RRTM/rad.f90, an option to pass the effective radius computed with Get_reffc
!     and Get_reffi defined in microphysics.f90 was added.
! (4) The CALL statment for rad_driver_rrtm is modified to pass computed effective radius.
! (5) Modifications were added to the subroutine, read_patch_background_sounding, so that it
!     patches a default sounding ported from WRF3.3.1 (module_ra_rrtmg_lw.f90) when no input file
!     is specified.
! (6) read_rad & write_rad were created for restart runs.
! (7) Several non-standard options were disabled: use_SCAM_ozon, do_scale_co2, co2factor
!     When ozon exist in SCAM IOP file, it will be used,
!     i.e., use_SCAM_ozon = .TRUE. when ozon is in IOP file, vice versa.
! (8) includePrecip: if true, qpl & qpi are included in radiation calculation
! (9) Further modifications are necessary when trace-gases and above-domain profiles are not
!     constant in time.
!
! 2017/01
! - Deleted the disabled non-standard options for cleaner code
! - Code modifications for time-varying patched soundings
! - THIS IS IMPORTANT: npatch_start, npatch_end, nzpatch are assumed to be constant throughout
!   simulation. This means that the domain top pressure always has to be between
!   psnd(npatch_start-1) and psnd(npatch_start). Initially presi(nz)-10 >= psnd(npatch_start), so
!   that resolution of pressure level for psnd should be larger than 20 mb or so if anticipated
!   presi(nz) change is less than 20 mb, which likely guarantees psnd(npatch_start-1) < presi(nz)
!   < psnd(npatch_start). Ways to mitigate this requirement are
!   - Have deep enough domain so that the domain top pressure does not change much,
!   - Make resolution of the background sounding (IOP file) coarse enough.
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	USE rad ! nradsteps, qrad, pres_input, tabs_slice, insolation_TOA, lwUp, etc from this module
	USE rad_driver, ONLY: rad_driver_rrtm, initialize_radiation, isInitialized_RadDriver, &
		tracesini, tracesupdate, read_rad, write_rad, land_frac
	USE parkind, ONLY: kind_rb ! RRTM expects reals with this kind parameter (8 byte reals)
	
	!==========================================================================================
	!============================== BEGIN CHANGES REQUIRED HERE ===============================
	!==========================================================================================
	! PORTABILITY NOTE:  Here, all of the stuff needed to call the radiation is drawn from
	!   various modules in SAM as an example.  You will need to bring all of the associated
	!   variables into this routine for your model (either by using the appropriate modules,
	!   passing them in as arguments or defining them here).
	!==========================================================================================
	
	! Logicals
	USE grid, ONLY: &
		masterproc, &  ! true if MPI rank==0.
		dostatisrad, & ! accumulate radiation statistics at this time step
		compute_reffc, compute_reffi, & ! get effective radius with Get_reffc, Get_reffi in microphysics
                compute_reffl ! option for radiatively-active drizzle
	
	USE params, ONLY: &
		dolongwave, &  ! do longwave radiation
		doshortwave, & ! do shortwave radiation
		doperpetual, & ! use diurnally-averaged insolation
		dosolarconstant, & ! specify mean insolation, zenith angle for perpetual insolatoin
		doseasons, &   ! allow diurnally-varying insolation to vary with time of year
		ocean ! if true, run is over ocean
	
        ! Tak Yamaguchi 2015/10: RADAEROS, RESLAST
	USE params, ONLY: &
		doradaerosimple, &
		dorestart_last_only
	
	! Characters
	USE grid, ONLY: &
		iopfile, & ! name of SCAM IOP forcing file, e.g. 'ctl_s11.nc'
		case ! used to construct path of SCAM IOP file in SAM
	
	! Integers
	USE grid, ONLY: &
		nx, ny, nzm, & ! number of grid points (x,y,z)
		nstep, icycle, & ! model step and substep number
		nrad, & ! call radiation every nrad timesteps
		nrestart, & ! switch to control starting/restarting of the model
		nrestart_skip, & ! number of skips of writing restart (default 0)
		nstat, & ! the interval in time steps to compute statistics
		nstop, & ! time step number to stop the integration
		nelapse ! time step number to elapse before stoping
	
	! Reals or real arrays
	USE params, ONLY: &
		cp, & ! specific heat of dry air at constant pressure, J/kg/K
		ggr, & ! gravitational acceleration, m/s2
		coszrs, & ! cosine of solar zenith angle
		latitude0, longitude0, & ! latitude, longitude in degrees
		solar_constant, & ! modified solar constant doperpetual==dosolarconstant==.true.
		zenith_angle ! zenith angle if doperpetual==dosolarconstant==.true.
	
	USE grid, ONLY: &
		dtn, & ! time step in seconds
		day, day0, & ! model day (current and at start of run) day=0. for 00Z, Jan 1.
		dz, adz ! vertical grid spacing is dz*adz(k) in this model
	
	! NOTE: when dosolarconstat==.true, insolation = solar_constant*cos(zenith_angle)
	
	USE vars, ONLY: &
		daysnd, & ! day in background sounding for SCAM IOP file
		t, tabs, qv, qcl, qci, qpl, qpi, & ! model fields
		sstxy, t00, & ! SST perturbation and 300 K
		pres, presi, & ! pressure at model levels/interfaces
		rho, & ! density profile.  In this anelastic model, rho=rho(z).
		radswup, radswdn, radqrsw, & ! radiation statistics, summed in x and y.
		radlwup, radlwdn, radqrlw, & ! radiation statistics, summed in x and y.
		s_flntoa, s_fsntoa, s_flntoac, s_fsntoac, s_solin, & ! TOA radiative flux statistics
                s_flnt, s_fsnt, & ! top-of-model fluxes
		s_flns, s_fsns, s_flnsc, s_fsnsc, s_fsds, s_flds, & ! surface radiative flux statistics
		lwnt_xy, swnt_xy, lwntc_xy, swntc_xy, solin_xy, & ! time-accumulated x-y fields of TOA radiative fluxes
		lwns_xy, swns_xy, lwnsc_xy, swnsc_xy ! time-accumulated x-y fields of surface radiative fluxes
	
	USE microphysics, ONLY: reffc, reffi, reffr, & ! These arrays replace get_reffc, get_reffi
                 dorrtm_cloud_optics_from_effrad_LegacyOption, &
                 Get_nca ! aerosol number concentration function (Tak Yamaguchi 2015/10: RADAEROS)
	
        !bloss(2017/07): compute effective radii in this routine if not provided by microphysics.
        use cam_rad_parameterizations, only : computeRe_Liquid, computeRe_Ice

	!==========================================================================================
	!=============================== END CHANGES REQUIRED HERE ================================
	!==========================================================================================
	
	IMPLICIT NONE
	
	! Parameters
	INTEGER, PARAMETER :: iyear = 1999 ! (TAK) RAD_RRTM/rad.f90 uses iyear=1999
	
	! Local variables
	LOGICAL, SAVE :: isInitialized_BackgroundSounding = .FALSE.
	LOGICAL, SAVE :: isRestart = .TRUE. ! true only at the first step of new and restart run
	CHARACTER(LEN=250) :: SoundingFileName ! input file for background soundings
	REAL(KIND=kind_rb), DIMENSION(nx,nzm) :: &
		swHeatingRate, & ! units: K/s
		lwHeatingRate, & !  units: K/s
		swHeatingRateClearSky, & !  units: K/s
		lwHeatingRateClearSky !  units: K/s
	REAL :: coef
	INTEGER :: ierr, i, j, k, i1, i2
	INTEGER, SAVE :: index_daysnd = 1 ! index for daysnd used for the last update of background snds
	
	!++++++++++++TAK: Including precipitating condensate in radiation calculation?+++++++++++++
	! Set TRUE if precipitation condensate (qpl, qpi) is included in radiation calculation,
	! e.g., effective radius from microphysics.
        !bloss(2018-02): Change default to false since effective radius may not be consisitent 
        !  with including rain in the liquid water content.  Similar for snow/graupel.
	LOGICAL :: includePrecip = .false.
	!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
	
	!==========================================================================================
	! PORTABILITY NOTE: all processors should have same pressure soundings (pres and presi) in
	!   hPa. This is important for the consistency of trace soundings across the different
	!   processors.
	!==========================================================================================
	
	IF ( icycle == 1 ) THEN ! Skip subcycles (i.e., when icycle /= 1)
	
		IF ( .NOT.isInitialized_RadDriver ) THEN
			! This IF block is TRUE only once.
			
			! Call initialization routine for radiation
			CALL initialize_radiation( cp, iyear, day0, latitude0, longitude0, doperpetual, ocean )
			
			! Allocate arrays that are fixed in size for the whole simulation.
			ALLOCATE( &
				NetlwUpSurface(nx,ny), NetlwUpSurfaceClearSky(nx,ny), &
				NetlwUpToa(nx,ny),     NetlwUpToaClearSky(nx,ny),     &
				NetlwUpToM(nx,ny),     NetswDownToM(nx,ny),     &
				NetswDownSurface(nx,ny), NetswDownSurfaceClearSky(nx,ny), &
				NetswDownToa(nx,ny),     NetswDownToaClearSky(nx,ny),     &
				insolation_TOA(nx,ny), swDownSurface(nx,ny), lwDownSurface(nx,ny), &
				swnsxy(nx,ny), lwnsxy(nx,ny), &
				STAT=ierr )
			IF ( ierr /= 0 ) THEN
				WRITE(*,*) 'Could not allocate fixed size arrays in rad_full'
				CALL rad_error()
			ENDIF
			
			isInitialized_RadDriver = .TRUE.
		ENDIF ! IF (.NOT.isInitialized_RadDriver)
		
		
		IF ( .NOT.isInitialized_BackgroundSounding ) THEN
			! This IF block is TRUE only once.
			
			! Read in sounding and define set up patching.
			! This will be stored in radiation restart file for restarts.
			SoundingFileName = './'//TRIM(case)//'/'//TRIM(iopfile)
			
			! Specify SoundingFileName as NoInputFile if there is no input file (TAK)
			IF( LEN('./'//TRIM(case)//'/') == LEN(TRIM(SoundingFileName)) ) &
				SoundingFileName='NoInputFile'
			
			! Setup background soundings
			! - Allocate psnd, tsnd, qsnd, o3snd, tsndng, qsndng, o3sndng
			! - Set have_o3mmr
			! - Read either default sounding or SCAM IOP file
			CALL read_patch_background_sounding( SoundingFileName, presi(nzm+1), masterproc )
			
			! nzpatch = number of patched levels (above domain top)
			! nzrad = total number of level for radiation
			IF ( npatch_end /= npatch_start ) THEN
				nzpatch = npatch_end - npatch_start + 1
			ELSE
				nzpatch = 0
			ENDIF
			nzrad = nzm + nzpatch
			
			IF ( masterproc ) WRITE(*,*) 'In RRTM radiation interface, using ozone profile'
			IF ( have_o3mmr .AND. masterproc ) THEN
				WRITE(*,*) '  from SCAM IOP netcdf file', SoundingFileName
			ELSE IF ( masterproc ) THEN
				WRITE(*,*) '  from default absorber profiles in rrtmg_lw.nc'
			ENDIF
			
			isInitialized_BackgroundSounding = .TRUE.
		ENDIF ! IF ( .NOT.isInitialized_BackgroundSounding )
		
		
		IF ( isAllocated_RadInputsOutputs .AND. nzrad /= nzrad_old ) THEN
			! This IF block is TRUE when nzpatch and nzrad were changed because the domain top
			! pressure becomes either smaller than psnd at previous npatch_start or larger than 
			! psnd at previous npatch_start-1.
			!
			! This IF block will be FALSE since nzrad will not be updated. This means that nzrad
			! is assumed to be constant. Have deep enough domain depth.
			
			! Deallocate old arrays
			! These variables have not been allocated yet at initialization/restart, thus this IF
			! block will be skipped.
			DEALLOCATE( &
				tabs_slice, qv_slice, qcl_slice, qci_slice, rel_slice, rei_slice, tg_slice, &
				nca_slice, &! Tak Yamaguchi 2015/10: RADAEROS
				pres_input, presi_input, lwUp, lwDown, lwUpClearSky, lwDownClearSky, &
				swUp, swDown, swUpClearSky, swDownClearSky, STAT=ierr )
			IF ( ierr /= 0 ) THEN
				WRITE(*,*) 'Could not deallocate input/output arrays in rad_full'
				CALL rad_error()
			ENDIF
			IF ( compute_reffc ) THEN
!bloss				DEALLOCATE( rel_rad, STAT=ierr )
!bloss				IF ( ierr /= 0 ) THEN
!bloss					WRITE(*,*) 'Could not deallocate rel_rad(nx,ny,nzm) in rad_full'
!bloss					CALL rad_error()
!bloss				ENDIF
			ENDIF
			IF ( compute_reffi ) THEN
!bloss				DEALLOCATE( rei_rad, STAT=ierr )
!bloss				IF ( ierr /= 0 ) THEN
!bloss					WRITE(*,*) 'Could not deallocate rei_rad(nx,ny,nzm) in rad_full'
!bloss					CALL rad_error()
!bloss				ENDIF
			ENDIF

                       ! Tak Yamaguchi 2015/10: RADAEROS
			IF ( doradaerosimple ) THEN
				DEALLOCATE( nca_rad, STAT=ierr )
				IF ( ierr /= 0 ) THEN
					WRITE(*,*) 'Could not deallocate nca_rad(nx,ny,nzm) in rad_full'
					CALL rad_error()
				ENDIF
			ENDIF

			isAllocated_RadInputsOutputs = .FALSE.
		ENDIF ! IF ( isAllocated_RadInputsOutputs .AND. nzrad /= nzrad_old )
		
		
		IF ( .NOT.isAllocated_RadInputsOutputs ) THEN
			! This IF block is TRUE only once for this version.
			! When nzrad is allowed to be changed, then this block should be TRUE whenever the above
			! IF block is TRUE.
			
			! Allocate arrays
			ALLOCATE( &
				tabs_slice(nx,nzrad), qv_slice(nx,nzrad), qcl_slice(nx,nzrad), &
				qci_slice(nx,nzrad), rel_slice(nx,nzrad), rei_slice(nx,nzrad), tg_slice(nx), &
				nca_slice(nx,nzrad), & ! Tak Yamaguchi 2015/10: RADAEROS
				pres_input(nzrad), presi_input(nzrad+1), &
				lwUp(nx,nzrad+2), lwDown(nx,nzrad+2), &
				lwUpClearSky(nx,nzrad+2), lwDownClearSky(nx,nzrad+2), &
				swUp(nx,nzrad+2), swDown(nx,nzrad+2), &
				swUpClearSky(nx,nzrad+2), swDownClearSky(nx,nzrad+2), &
				STAT=ierr )
			IF ( ierr /= 0 ) THEN
				WRITE(*,*) 'Could not allocate input/output arrays in rad_full'
				CALL rad_error()
			ENDIF
			IF ( compute_reffc ) THEN
!bloss				ALLOCATE( rel_rad(nx,ny,nzm), STAT=ierr )
!bloss				IF ( ierr /= 0 ) THEN
!bloss					WRITE(*,*) 'Could not allocate rel_rad(nx,ny,nzm) in rad_full'
!bloss					CALL rad_error()
!bloss				ENDIF
				rad_reffc(:,:,:) = 0.0
			ENDIF
			IF ( compute_reffi ) THEN
!bloss				ALLOCATE( rei_rad(nx,ny,nzm), STAT=ierr )
!bloss				IF ( ierr /= 0 ) THEN
!bloss					WRITE(*,*) 'Could not allocate rei_rad(nx,ny,nzm) in rad_full'
!bloss					CALL rad_error()
!bloss				ENDIF
				rad_reffi(:,:,:) = 0.0
			ENDIF

                        ! Tak Yamaguchi 2015/10: RADAEROS
			IF ( doradaerosimple ) THEN
				ALLOCATE( nca_rad(nx,ny,nzm), STAT=ierr )
				IF ( ierr /= 0 ) THEN
					WRITE(*,*) 'Could not allocate nca_rad(nx,ny,nzm) in rad_full'
					CALL rad_error()
				ENDIF
				nca_rad(:,:,:) = 0.0
			ENDIF

			isAllocated_RadInputsOutputs = .TRUE.
			nzrad_old = nzrad
		ENDIF ! IF ( .NOT.isAllocated_RadInputsOutputs )
		
		
		IF ( nstep == 1 .OR. isRestart ) THEN
			! Set up pressure inputs to radiation -- needed for initialize_radiation
			pres_input(1:nzm) = pres(1:nzm)
			presi_input(1:nzm+1) = presi(1:nzm+1)
			IF ( nzpatch > 0 ) THEN
				pres_input(nzm+1:nzrad) = psnd(npatch_start:npatch_end) ! layer pressures
				presi_input(nzm+2:nzrad) & ! interface pressures.
					= 0.5*(psnd(npatch_start:npatch_end-1) + psnd(npatch_start+1:npatch_end))
				presi_input(nzrad+1) &
					= MAX(0.5*psnd(npatch_end), 1.5*psnd(npatch_end) - 0.5*psnd(npatch_end-1))
			ENDIF
			
			! Allocate trace gases arrays used in RRTM.
			! Interpolates standard sounding of trace gas concentrations to grid for radiation.
			! For restart runs, trace gas profiles will be overwritten when they are read from the
			! restart data. Thus no time interpolation for o3snd is necessary.
			CALL tracesini( nzrad, pres_input, presi_input, ggr, nzsnd, psnd, o3snd, have_o3mmr, &
				1.0, masterproc )
		ENDIF ! IF ( nstep == 1 .OR. isRestart )
		
		
		! Read restart data at restart
		! This will read nradsteps, npatch_start, npatch_end, nzpatch, psnd, tsnd, qsnd, other
		! allocatables (qrad, etc. declared in vars.d90, rad.f90), and trace gases (o3, co2, etc.
		! declared in rad_driver.f90).
		IF ( isRestart ) THEN
			IF ( nrestart /= 0 ) CALL read_rad() ! Read radiation restart data
			isRestart = .FALSE.
		ENDIF
		
		
		! Update radiation
		nradsteps = nradsteps + 1
		IF ( nstep == 1 .OR. nradsteps >= nrad ) THEN
			! Zero out radiation statistics that are summed in x- and y-directions.
			radlwup(:) = 0.0
			radlwdn(:) = 0.0
			radqrlw(:) = 0.0
			radswup(:) = 0.0
			radswdn(:) = 0.0
			radqrsw(:) = 0.0
			radqrclw(:) = 0.0
			radqrcsw(:) = 0.0
			
			! Check if presi(nz) is within psnd(npatch_start-1) and psnd(npatch_start).
			! This is necessary since npatch_start (so as nzpatch, nzrad) is constant with time.
			! psnd(npatch_start-1) < presi(nz) is perhaps OK as long as psnd(npatch_start-1) ~
			! presi(nz), but I include this is also an error.
                        !bloss(2017-07): Make this only a warning/notice if presi(nz) < psnd(npatch_start-2)
			IF ( nzpatch > 0 ) THEN
				IF ( psnd(npatch_start-1) < presi(nzm+1) .OR. presi(nzm+1) < psnd(npatch_start) ) THEN
					IF ( masterproc ) THEN
                                           PRINT*,'Patching background sounding on top of dynamical model domain'
                                           PRINT*,'  for radiation computation.  '
                                           write(*,992) presi(1), presi(nzm+1), psnd(npatch_start), psnd(npatch_end)
                                           992 format('Dynamical model domain: ',F6.1,' < p (hPa) < ',F6.1, &
                                                    ', Patched sounding above: ',F6.1,' < p (hPa) < ',F6.1)
                                         END IF
                                         IF(abs(presi(nzm+1)-psnd(npatch_start)).gt.100.) THEN
                                           if(masterproc) PRINT*,'Gap seems too big, so stopping...'
                                           CALL task_abort()
                                         END IF
                                       ENDIF
			ENDIF
			
			! Set up pressure inputs to radiation
			! THIS MAY NOT BE UNNECESSARY WHEN PRES AND PRESI DO NOT CHANGE MUCH.
			pres_input(1:nzm) = pres(1:nzm)
			presi_input(1:nzm+1) = presi(1:nzm+1)
			IF ( nzpatch > 0 ) THEN
				pres_input(nzm+1:nzrad) = psnd(npatch_start:npatch_end) ! layer pressures
				presi_input(nzm+2:nzrad) & ! interface pressures
					= 0.5 * ( psnd(npatch_start:npatch_end-1) + psnd(npatch_start+1:npatch_end) )
				presi_input(nzrad+1) &
					= MAX( 0.5*psnd(npatch_end), 1.5*psnd(npatch_end) - 0.5*psnd(npatch_end-1) )
			ENDIF
			
			! Time interpolate background soundings
			IF ( ntsnd /= 1 ) THEN
				! Background profiles are from a SCAM IOP file
				! Use daysnd to identify the two closest times to the current time
				DO i = index_daysnd, ntsnd
					IF ( day < daysnd(i) ) THEN
						index_daysnd = i - 1
						EXIT
					ENDIF
				ENDDO
				i1 = index_daysnd
				i2 = i1 + 1
				coef = ( day - daysnd(i1) ) / ( daysnd(i2) - daysnd(i1) )
				tsnd(:) = tsndng(:,i1) + ( tsndng(:,i2) - tsndng(:,i1) ) * coef
				qsnd(:) = qsndng(:,i1) + ( qsndng(:,i2) - qsndng(:,i1) ) * coef
				
				! Update o3snd, then o3 following rad_driver.f90/tracesini
				IF ( have_o3mmr ) THEN
					o3snd(:) = o3sndng(:,i1) + ( o3sndng(:,i2) - o3sndng(:,i1) ) * coef
					CALL tracesupdate( nzrad, pres_input, presi_input, ggr, nzsnd, psnd, o3snd, &
						have_o3mmr, masterproc )
				ENDIF
			ENDIF
			
			! Obtain effective radius from microphysics.f90 (TAK)
                        !   or compute it using the CAM parameterizations.
			IF ( compute_reffc ) then
                           rad_reffc(1:nx,1:ny,1:nzm) = reffc(:,:,:)
                        ELSE
                           do k = 1,nzm
                             do j = 1,ny
                               do i = 1,nx
                                 rad_reffc(i,j,k) = computeRe_Liquid( tabs(i,j,k), land_frac, 0.0_kind_rb, 0.0_kind_rb )
                               end do
                             end do
                           end do
                         END IF

			IF ( compute_reffi ) THEN
                           rad_reffi(1:nx,1:ny,1:nzm) = reffi(:,:,:)
                        ELSE
                          rad_reffi(1:nx,1:ny,1:nzm) = computeRe_Ice( tabs(1:nx,1:ny,1:nzm) )
                        END IF
			
                        !bloss(2018-02): option for drizzle/rain being radiatively active
                        if(compute_reffl.AND.dorrtm_cloud_optics_from_effrad_LegacyOption) then
                          !bloss(2018-02): Combine cloud and rain/drizzle to get a composite species
                          !   with an average optical depth
                          do k = 1,nzm
                            do j = 1,ny
                              do i = 1,nx
                                if(qpl(i,j,k)+qcl(i,j,k).gt.0._kind_rb) then
                                  rad_reffc(i,j,k) = ( qcl(i,j,k) + qpl(i,j,k) ) &
                                       / (qcl(i,j,k)/reffc(i,j,k) + qpl(i,j,k)/reffr(i,j,k) )
                                end if
                              end do
                            end do
                          end do
!bloss(2018-02)                          rad_reffc(1:nx,1:ny,1:nzm) = reffl(1:nx,1:ny,1:nzm)
                          includePrecip = .true. ! include rain as radiatively active
                        end if

                        ! Tak Yamaguchi 2015/10: RADAEROS
			! Obtain aerosol number concentration from microphysics.f90
			IF ( doradaerosimple ) nca_rad(1:nx,1:ny,1:nzm) = Get_nca() ! #/cm3
			
			!
			DO j = 1, ny
				! Extract a slice from the three-dimensional domain on this processor.
				!   We need absolute temperature (K), mass mixing ratios (kg/kg) of
				!   water vapor, cloud liquid water and cloud ice, along with SST (K).
				tabs_slice(1:nx,1:nzm) = tabs(1:nx,j,1:nzm)
				qv_slice(1:nx,1:nzm) = qv(1:nx,j,1:nzm)
				IF ( .NOT.includePrecip ) THEN
					qcl_slice(1:nx,1:nzm) = qcl(1:nx,j,1:nzm)
					qci_slice(1:nx,1:nzm) = qci(1:nx,j,1:nzm)
				ELSE
					qcl_slice(1:nx,1:nzm) = qcl(1:nx,j,1:nzm) + qpl(1:nx,j,1:nzm)
					qci_slice(1:nx,1:nzm) = qci(1:nx,j,1:nzm) + qpi(1:nx,j,1:nzm)
				ENDIF
				tg_slice(1:nx) = sstxy(1:nx,j) + t00

                                !bloss(2017/07): 
                                rel_slice(1:nx,1:nzm) = rad_reffc(1:nx,j,1:nzm) ! (TAK)
				rei_slice(1:nx,1:nzm) = rad_reffi(1:nx,j,1:nzm) ! (TAK)

                                ! Tak Yamaguchi 2015/10: RADAEROS
				! Extract aerosol number concentration
				IF ( doradaerosimple ) nca_slice(1:nx,1:nzm) = nca_rad(1:nx,j,1:nzm)
				
				IF ( nzpatch > 0)  THEN
					! Patch sounding on top of model sounding for more complete radiation calculation.
					tabs_slice(1:nx,nzm+1:nzrad) &
						= SPREAD( tsnd(npatch_start:npatch_end), DIM=1, NCOPIES=nx )
					qv_slice(1:nx,nzm+1:nzrad) &
						= SPREAD( qsnd(npatch_start:npatch_end), DIM=1, NCOPIES=nx )
					qcl_slice(1:nx,nzm+1:nzrad) = 0.0
					qci_slice(1:nx,nzm+1:nzrad) = 0.0
					IF ( compute_reffc ) rel_slice(1:nx,nzm+1:nzrad) = 0.0 ! (TAK)
					IF ( compute_reffi ) rei_slice(1:nx,nzm+1:nzrad) = 0.0 ! (TAK)

                                        ! Tak Yamaguchi 2015/10: RADAEROS
					! Patch aerosol number concentration
                                        ! FOR THE CURRENT VERSION THIS IS MEANINGLESS SINCE AEROSOL OPTICAL DEPTH IS COMPUTED BELOW NZM
                                        ! SINCE THERE IS NO INFORMATION ABOUT DZ IN THE PATCHED SOUNDING. PATCHED SOUNDING IS BASED ON
                                        ! PRESSURE.
					IF ( doradaerosimple ) THEN
						DO k = nzm+1, nzrad
							nca_slice(1:nx,k) = nca_slice(1:nx,nzm)
						ENDDO
					ENDIF
				ENDIF
				
				! Make call to wrapper routine for RRTMG (v.4.8 for LW, v.3.8 for SW)
				CALL rad_driver_rrtm( nx, nzrad, nzm, j, pres_input, presi_input, &
					tabs_slice, qv_slice, qcl_slice, qci_slice, rel_slice, rei_slice, tg_slice, &
					nca_slice, & ! Tak Yamaguchi 2015/10: RADAEROS
					dolongwave, doshortwave, doperpetual, doseasons, &
					dosolarconstant, solar_constant, zenith_angle, &
					day, day0, latitude0, longitude0, &
					ocean, ggr, cp, masterproc, &
					lwUp, lwDown, lwUpClearSky, lwDownClearSky, &
					swUp, swDown, swUpClearSky, swDownClearSky, coszrs )
				
				! Compute heating rates from fluxes using local density in model.
				! Results in heating rates in K/s.
				! PORTABILIY NOTE: CHANGE THERMAL MASS TO THAT USED IN YOUR MODEL.
				!   Units below are cp*rho*deltaz ~ J/kg/K * kg/m3 * m
				!   where delta z = dz*adz(k) in SAM.
				DO k = 1, nzm ! loop over model levels
					swHeatingRate(1:nx,k) &
						= (swDown(:,k+1) - swDown(:,k) + swUp(:,k) - swUp(:,k+1)) / (cp*rho(k)*dz*adz(k))
					lwHeatingRate(1:nx,k) &
						= (lwDown(:,k+1) - lwDown(:,k) + lwUp(:,k) - lwUp(:,k+1)) / (cp*rho(k)*dz*adz(k))
					swHeatingRateClearSky(1:nx,k) &
						= ( swDownClearSky(:,k+1) - swDownClearSky(:,k) &
						+ swUpClearSky(:,k) - swUpClearSky(:,k+1) ) &
						/ (cp*rho(k)*dz*adz(k))
					lwHeatingRateClearSky(1:nx,k) &
						= ( lwDownClearSky(:,k+1) - lwDownClearSky(:,k) &
						+ lwUpClearSky(:,k) - lwUpClearSky(:,k+1) ) &
						/ (cp*rho(k)*dz*adz(k))
				ENDDO
				
				! Update total radiative heating rate of model
				qrad(1:nx,j,1:nzm) = swHeatingRate(1:nx,1:nzm) + lwHeatingRate(1:nx,1:nzm)
				
				! Accumulate heating rates and fluxes for horizontally-averaged statistics
				radlwup(:) = radlwup(:) + SUM(lwUp(:, 1:nzm+1),   DIM = 1)
				radlwdn(:) = radlwdn(:) + SUM(lwDown(:, 1:nzm+1), DIM = 1)
				radqrlw(1:nzm) = radqrlw(1:nzm) + SUM(lwHeatingRate(:, 1:nzm), DIM = 1)
				radqrclw(1:nzm) = radqrclw(1:nzm) + sum(lwHeatingRateClearSky(:, 1:nzm), dim = 1)
				radswup(:) = radswup(:) + SUM(swUp(:, 1:nzm+1),   DIM = 1)
				radswdn(:) = radswdn(:) + SUM(swDown(:, 1:nzm+1), DIM = 1)
				radqrsw(1:nzm) = radqrsw(1:nzm) + SUM(swHeatingRate(:, 1:nzm), DIM = 1)
				radqrcsw(:nzm) = radqrcsw(:nzm) + sum(swHeatingRateClearSky(:, 1:nzm), dim = 1)
				
				! Shortwave fluxes at top-of-atmosphere (TOA) and surface -- NOTE POSITIVE DOWNWARDS
				insolation_TOA(1:nx,j) = swDown(:,nzrad+2) ! shortwave down at TOA
				swDownSurface(1:nx,j) = swDown(:,1) ! shortwave down at surface
				
				NetswDownToM(1:nx,j) = swDown(:,nzm+1) - swUp(:,nzm+1) ! net shortwave down at Top of Model
				NetswDownToa(1:nx,j) = swDown(:,nzrad+2) - swUp(:,nzrad+2) ! net shortwave down at TOA
				NetswDownToaClearSky(1:nx,j) = swDownClearSky(:,nzrad+2) - swUpClearSky(:,nzrad+2) ! net clearsky shortwave down at TOA
				
				NetswDownSurface(1:nx,j) = swDown(:,1) - swUp(:,1) ! net shortwave down at surface
				NetswDownSurfaceClearSky(1:nx,j) = swDownClearSky(:,1) - swUpClearSky(:,1) ! net clearsky shortwave down at surface
				
				! longwave fluxes at top-of-atmosphere (TOA) and surface -- NOTE POSITIVE UPWARDS
				lwDownSurface(1:nx,j) = lwDown(:,1) ! longwave down at surface
				
				NetlwUpToM(1:nx,j) = lwUp(:,nzm+1) - lwDown(:,nzm+1) ! net longwave up at Top of Model
				NetlwUpToa(1:nx,j) = lwUp(:,nzrad+2) - lwDown(:,nzrad+2) ! net longwave up at TOA
				NetlwUpToaClearSky(1:nx,j) = lwUpClearSky(:,nzrad+2) - lwDownClearSky(:,nzrad+2) ! net clearsky longwave up at TOA
				
				NetlwUpSurface(1:nx,j) = lwUp(:,1) - lwDown(:,1) ! net longwave up at surface
				NetlwUpSurfaceClearSky(1:nx,j) = lwUpClearSky(:,1) - lwDownClearSky(:,1) ! net clearsky longwave up at surface
				
			ENDDO ! j = 1, ny
			! Re-initialize nradsteps for the next update
			nradsteps = 0
		ENDIF ! IF (updateRadiation)
		
		
		! Update 2d diagnostic fields
		! Net surface and toa fluxes
		! First two for ocean evolution
		lwnsxy(:, :) = NetlwUpSurface(:, :) ! instantaneous
		swnsxy(:, :) = NetswDownSurface(:, :)
		
		! net full sky radiative fluxes (varying in x and y, time-accumulated)
		lwns_xy(:, :) = lwns_xy(:, :) + NetlwUpSurface(:, :)
		swns_xy(:, :) = swns_xy(:, :) + NetswDownSurface(:, :)
		lwnt_xy(:, :) = lwnt_xy(:, :) + NetlwUpToa(:, :)
		swnt_xy(:, :) = swnt_xy(:, :) + NetswDownToa(:, :)
		
		! net clear sky radiative fluxes (varying in x and y, time-accumulated)
		lwnsc_xy(:, :) = lwnsc_xy(:, :) + NetlwUpSurfaceClearSky(:, :)
		swnsc_xy(:, :) = swnsc_xy(:, :) + NetswDownSurfaceClearSky(:, :)
		lwntc_xy(:, :) = lwntc_xy(:, :) + NetlwUpToaClearSky(:, :)
		swntc_xy(:, :) = swntc_xy(:, :) + NetswDownToaClearSky(:, :)
		
		! TOA Insolation
		solin_xy(:, :) = solin_xy(:, :) + insolation_Toa(:, :)
		
		
		! Update 1D diagnostics
		IF ( dostatisrad ) THEN
			s_flns = s_flns + sum(NetlwUpSurface(:, :))   ! lwnsxy
			s_fsns = s_fsns + sum(NetswDownSurface(:, :)) ! swnsxy
			s_flnt = s_flnt + sum(NetlwUpToM(:, :))       ! LW Net Up Top of Model
			s_fsnt = s_fsnt + sum(NetswDownToM(:, :))     ! SW Net Dn Top of Model
			s_flntoa = s_flntoa + sum(NetlwUpToa(:, :))       ! lwntxy TOA
			s_fsntoa = s_fsntoa + sum(NetswDownToa(:, :))     ! swntxy TOA
			s_flnsc = s_flnsc + sum(NetlwUpSurfaceClearSky(:, :))   ! lwnscxy
			s_fsnsc = s_fsnsc + sum(NetswDownSurfaceClearSky(:, :)) ! swnscxy 
			s_flntoac = s_flntoac + sum(NetlwUpToaClearSky(:, :))       ! lwntcxy
			s_fsntoac = s_fsntoac + sum(NetswDownToaClearSky(:, :))     ! swntcxy
			s_solin = s_solin + sum(insolation_TOA(:, :))           ! solinxy 
			! 
			s_fsds = s_fsds + sum(swDownSurface(:, :)) ! downwelling SW at surface
			s_flds = s_flds + sum(lwDownSurface(:, :)) ! downwelling LW at surface
		ENDIF ! if(dostatis)
		
		! Write radiation restart file
		IF ( MOD(nstep,nstat*(1+nrestart_skip)) == 0.OR.nstep == nstop.OR.nelapse == 0 ) THEN

                        ! Tak Yamaguchi 2015/10 RESLAST
			IF ( .NOT.dorestart_last_only &
			     .OR. ( dorestart_last_only .AND. (nstep == nstop .OR. nelapse == 0 ) ) ) &
			CALL write_rad()
		ENDIF
			
	ENDIF ! IF (icycle == 1)
	
	
	! Add radiative heating to model thermodynamic variable.
	! (here t is liquid-ice static energy divided by Cp.)
	! PORTABILITY NOTE: don't forget exner function if your model uses theta.
	t(1:nx, 1:ny, 1:nzm) = t(1:nx, 1:ny, 1:nzm) + qrad(:, :, :) * dtn
	
END SUBROUTINE rad_full
