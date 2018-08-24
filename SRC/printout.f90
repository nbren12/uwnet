	subroutine printout()

	use vars
        use tracers
	use params
	implicit none
	
	print*
	print*,'Case: ',trim(case)
	print*,'Caseid: ',trim(caseid)
        select case (nrestart) 
	case(0) 
	  print*,'New Run.'
	case(1)
	  print*,'Restart. nrestart=',nrestart
	case(2) 
          print*,'Branch run Restart Case:',trim(case_restart)
          print*,'Case-ID:',trim(caseid_restart)
        end select

	print*,'Day:',day

	if(.not.(OCEAN.or.LAND)) then
	 print*, 'Neither OCEAN nor LAND are set. Exitting...'
	 call task_abort()
	endif
	if(OCEAN.and.LAND) then
	 print*, 'Both OCEAN and LAND are set. Confused...'
	 call task_abort()
	endif
	if(.not.(CEM.or.LES)) then
	 print*, 'Neither CEM nor LES are set. Exitting...'
	 call task_abort()
	endif
	if(CEM.and.LES) then
	 print*, 'Both CEM and LES are set. Confused...'
	 call task_abort()
	endif

	if(LES) print*,'Model type: LES'
	if(CEM) print*,'Model type: CEM'
	
	print*, 'Finish at timestep:',nstop
	print*, 'Finish on day:',day+(nstop-nstep)*dt/3600./24.
	print*
	print*, 'Statistics file ouput frequency: ',nstat,' steps'
	print*, 'Statistics file sampling: every ',nstatfrq,' steps'
	print*, 'printouts frequency:',nprint,' steps'
	if(nstop-nstep.lt.nstat) then
	  print*, 'Error: job will finish before statistics is done'
	  call task_abort()
	endif
	
	if(nadams.eq.2.or.nadams.eq.3) then
	   print*, 'Adams-Bashforth scheme order:',nadams
	else
	   print*, 'Error: nadams =',nadams
	   call task_abort()
	endif 
	  	
	print*
	if(dx.gt.0.and.dy.gt.0.and.dz.gt.0. ) then
	   print*, 'Global Grid:',nx_gl,ny_gl,nz_gl
	   print*, 'Local Grid:',nx,ny,nzm
	   print*, 'Grid spacing (m) dx, dy, dz:',dx, dy, dz
	   print*, 'Domain dimensions (m):',	dx*nx_gl,dy*ny_gl,z(nzm)
	else
	   print*, 'Error: grid spacings dx, dy, dz:',dx, dy, dz
	   call task_abort()
	endif
        print*,'dowallx=',dowallx
        print*,'dowally=',dowally
	print*
	if(dt.gt.0) then
	   print*, 'Timestep (sec):',dt
	else
	   print*, 'Error: dt =',dt
	   call task_abort()
	endif   	
        print*,'domain translation velocities ug, vg:', ug, vg
        print*	
	print*, 'do column model', docolumn
	print*, 'do convective parameterization', docup

	print*, 'do spange damping at the domain top: ', dodamping
	print*, 'maintain grad. of scalars at the top:',doupperbound
	print*, 'clouds are allowed:',docloud
	print*, 'precipitation is allowed:',doprecip.and.docloud
        if(docloud.and.dosmoke) then
          if(masterproc) print*,'docloud and dosmoke can not be true simultaneously'
	  call task_abort()
        end if
	 print*,'smoke cloud',dosmoke
	print*, 'SGS scheme is on:',dosgs
	print*, 'larger-scale subsidence is on:',dosubsidence
	print*, 'larger-scale tendency is on:',dolargescale
	print*, 'coriolis force is allowed:',docoriolis	
	print*, 'vertical coriolis force is allowed:',docoriolisz
	if(docoriolis) then
            print*, 'do f-plane approximation:',dofplane
            if(dofplane) then
		print*, '   Coriolis parameter (1/s):',fcor
		print*, '   Vertical Coriolis parameter (1/s):',fcorz
            else
                print*, '   Coriolis parameter is the function of latitude'
            end if
	endif	
        if(doradforcing.and.(dolongwave.or.doshortwave)) then
          print*, 'prescribed rad. forcing and radiation '// &
          'calculations cannot be done at the same time.'
          call task_abort()
        endif
        if(dolongwave) print*, 'longwave radiation:',dolongwave  
        if(doshortwave) then
            print*, 'shortwave radiation:',doshortwave
            print*, 'do seasonal solar cycle:',doseasons
            print*, 'do perpetual sun:',doperpetual
            print*, 'compute effective radius for liquid:',compute_reffc
            print*, 'compute effective radius for ice:',compute_reffi
	endif
        print*,'doradsimple = ',doradsimple
        print*,'doradhomo =',doradhomo
        print*,'dosolarconstant = ',dosolarconstant
        print*,'solar_constant = ',solar_constant
        print*,'zenith_angle = ',zenith_angle
        print*,'doradlon = ',doradlon
        print*,'doradlat = ',doradlat
        print*, 'Latitude0:',latitude0
        print*, 'Longitude0:',longitude0
        print*,'rundatadir = ',trim(rundatadir)
        if(doradforcing) print*, 'radiation forcing is prescribed'
        print*,'factor to change present CO2 concentration: nxco2=',nxco2
	print*,'surface flux parameterization is on:',dosurface
	if(dosurface) then
	    if(LAND) then
               print*,'Surface type: LAND'
               print*,'soil_wetness=',soil_wetness
               print*,'z0=',z0
            end if
	    if(OCEAN) print*,'Surface type: OCEAN'
	    print*, ' sensible heat flux prescribed:',SFC_FLX_FXD
            if(SFC_FLX_FXD.and..not.dosfcforcing) print*, 'fluxt0 (W/m2)=',fluxt0*rhow(1)*cp
	    print*, ' latent heat flux prescribed:',SFC_FLX_FXD
            if(SFC_FLX_FXD.and..not.dosfcforcing) print*, 'fluxq0 (W/m2)=',fluxq0*rhow(1)*lcond
	    print*, ' surface stress prescribed:',SFC_TAU_FXD
            if(SFC_TAU_FXD.and..not.dosfcforcing) print*, 'tau0 (m2/s2)=',tau0
            print*,'dosfchomo = ',dosfchomo
	endif
        print*,'doisccp = ',doisccp
        print*,'domodis = ',domodis
        print*,'domisr = ',domisr
        print*,'dosimfilesout=',dosimfilesout
	
        if(dolargescale.or.dosubsidence) then
          if(    day.lt.dayls(1) &
            .or.day+(nstop-nstep)*dt/86400..gt.dayls(nlsf)) then
             print*,'Error: simulation time (from start to stop)'// &
              'can be beyond the l.s. forcing intervals'
             print*,'current day=',day
             print*,'stop day=',day+(nstop-nstep)*dt/86400.
             print*,'ls forcing: start =',dayls(1)
             print*,'ls forcing:   end =',dayls(nlsf)
             call task_abort()
          endif
        endif
        print*,'dodynamicocean =',dodynamicocean
        if(dodynamicocean) then
             print*,'ocean_type =',ocean_type
             print*,'ocean depth = ',depth_slab_ocean
             print*,'Initial SST = ',tabs_s
             print*,'SST sin-amplitude= ',delta_sst
             print*,'mean ocean transport =',Szero
             print*,'ocean transport linear max variation=',deltaS
        end if
        if(dosurface.and.dosfcforcing) then
          print*,'surface temperature prescribed: T'
          print*,'dossthomo =',dossthomo
          if(dodynamicocean) then
             print*,'ocean_type =',ocean_type
	     print*, 'dodynamicocean cannot be set to T'// &
                     'when dosfcforcing is also T'
	     call task_abort()
	  end if
          if(    day.lt.daysfc(1) &
             .or.day+(nstop-nstep)*dt/86400..gt.daysfc(nsfc))then
             print*,'Error: simulation time (from start to stop)'// &
              'can be beyond the sfc forcing intervals'
             print*,'current day=',day
             print*,'stop day=',day+(nstop-nstep)*dt/86400.
             print*,'sfc forcing:start =',daysfc(1)
             print*,'sfc forcing:  end =',daysfc(nsfc)
             call task_abort()
          endif
        endif
        if(doradforcing) then
          if ( day.lt.dayrfc(1) &
            .or.day+(nstop-nstep)/86400.*dt.gt.dayrfc(nrfc))then
             print*,'Error: simulation time (from start to stop)'// &
              'can be beyond the rad. forcing intervals'
             print*,'current day=',day
             print*,'stop day=',day+(nstop-nstep-1)*dt/86400.
             print*,'rad forcing:start =',dayrfc(1)
             print*,'rad forcing:  end =',dayrfc(nrfc)
             call task_abort()
          endif
        endif
        print*,'doseawater=',doseawater
        print*,'salt_factor (satur. correction for salty water)=',salt_factor

	if(donudging_uv) print*, 'Nudging of U and V:', donudging_uv
	if(donudging_uv) print*,'tauls = ',tauls
	if(donudging_tq) print*, 'Nudging of T and Q:', donudging_tq


        print*,'dotracers =',dotracers
        if(dotracers) then
           if(ntracers.eq.0) then
             print*,'dotracers is set to .true., yet ntracers = 0. Aborting ...'
             call task_abort()
           end if
           print*,ntracers, ' tracers are included'
           print*,'Tracer names:',tracername(1:ntracers)
        end if
        print*,'output_sep =',output_sep
        print*,'perturb_type = ',perturb_type
        print*,'doSAMconditionals =',doSAMconditionals
        print*,'dosatupdnconditionals =',dosatupdnconditionals
        print*,'doscamiopdata =',doscamiopdata
        print*,'iopfile:',trim(iopfile)
        print*,'dozero_out_day0 =',dozero_out_day0
        print*,'nsave2D = ',nsave2D
        print*,'nsave2Dstart = ',nsave2Dstart
        print*,'nsave2Dend = ',nsave2Dend
        print*,'save2Dbin = ',save2Dbin
        print*,'nsave3D = ',nsave3D
        print*,'nsave3Dstart = ',nsave3Dstart
        print*,'nsave3Dend = ',nsave3Dend
        print*,'save3Dbin = ',save3Dbin
        print*,'qnsave3D = ',qnsave3D
        print*,'docolumn = ',docolumn
        print*,'doensemble = ',doensemble
        print*,'nensemble = ',nensemble
        print*,'nstatmom = ',nstatmom
        print*,'nstatmomstart = ',nstatmomstart
        print*,'nstatmomend = ',nstatmomend
        print*,'savemombin = ',savemombin
        print*,'nmovie = ',nmovie
        print*,'nmoviestart = ',nmoviestart
        print*,'nmovieend = ',nmovieend
        
	return
	end

