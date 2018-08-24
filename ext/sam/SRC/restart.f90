	subroutine write_all()
	
	use vars
	implicit none
	character *4 rankchar
	character *256 filename
	integer irank
        integer, external :: lenstr

        call t_startf ('restart_out')

        if(masterproc) then
         print*,'Writing restart file ...'
         filename = './RESTART/'//trim(case)//'_'//trim(caseid)//'_misc_restart.bin'
         open(66,file=trim(filename), status='unknown',form='unformatted')
        end if


	if(restart_sep) then

          write(rankchar,'(i4)') rank

          filename = './RESTART/'//trim(case)//'_'//trim(caseid)//'_'//&
                rankchar(5-lenstr(rankchar):4)//'_restart.bin'


          open(65,file=trim(filename), status='unknown',form='unformatted')
          write(65) nsubdomains, nsubdomains_x, nsubdomains_y

	  call write_statement


	else
	  write(rankchar,'(i4)') nsubdomains
	  filename = './RESTART/'//trim(case)//'_'//trim(caseid)//'_'//&
                rankchar(5-lenstr(rankchar):4)//'_restart.bin'

	  do irank=0,nsubdomains-1
	
	     call task_barrier()

	     if(irank.eq.rank) then

	       if(masterproc) then
	      
	        open(65,file=trim(filename), status='unknown',form='unformatted')
	        write(65) nsubdomains, nsubdomains_x, nsubdomains_y

	       else

                open(65,file=trim(filename), status='unknown',form='unformatted',&
                   position='append')

	       end if

               call write_statement

             end if
	  end do

	end if ! restart_sep

	call task_barrier()

        call t_stopf ('restart_out')

        return
        end
 
 
 
 
     
	subroutine read_all()
	
	use vars
	implicit none
	character *4 rankchar
	character *256 filename
	integer irank, ii
        integer, external :: lenstr

        if(masterproc) print*,'Reading restart file ...'

        if(nrestart.ne.2) then
          filename = './RESTART/'//trim(case)//'_'//trim(caseid)//'_misc_restart.bin'
        else
          filename = './RESTART/'//trim(case_restart)//'_'//trim(caseid_restart)//'_misc_restart.bin'
        end if
        open(66,file=trim(filename), status='unknown',form='unformatted')


	if(restart_sep) then

           write(rankchar,'(i4)') rank

           if(nrestart.ne.2) then
             filename = './RESTART/'//trim(case)//'_'//trim(caseid)//'_'//&
                  rankchar(5-lenstr(rankchar):4)//'_restart.bin'
           else
             filename = './RESTART/'//trim(case_restart)//'_'//trim(caseid_restart)//'_'//&
                  rankchar(5-lenstr(rankchar):4)//'_restart.bin'
           end if


           open(65,file=trim(filename), status='unknown',form='unformatted')
           read(65)

	   call read_statement


	else

	  write(rankchar,'(i4)') nsubdomains

          if(nrestart.ne.2) then
	    filename='./RESTART/'//trim(case)//'_'//trim(caseid)//'_'//&
                  rankchar(5-lenstr(rankchar):4)//'_restart.bin'
          else
	    filename='./RESTART/'//trim(case_restart)//'_'//trim(caseid_restart)//'_'//&
                  rankchar(5-lenstr(rankchar):4)//'_restart.bin'
          end if
          open(65,file=trim(filename), status='unknown',form='unformatted')

	  do irank=0,nsubdomains-1
	
	     call task_barrier()

	     if(irank.eq.rank) then

	       read (65)
 
               do ii=0,irank-1 ! skip records
                 read(65)
	       end do

	       call read_statement

             end if

	  end do

	end if ! restart_sep

	call task_barrier()
	
        dtfactor = -1.

! update the boundaries 
! (just in case when some parameterization initializes and needs boundary points)

        call boundaries(0)
        call boundaries(4)

        return
        end
 
 
 
        subroutine write_statement()

        use vars
        use microphysics, only: micro_field, nmicro_fields
        use sgs, only: sgs_field, nsgs_fields, sgs_field_diag, nsgs_fields_diag
        use tracers
        use params
        use movies, only: irecc
        implicit none

        write(65)  &
         u, v, w, t, p, qv, qcl, qci, qpl, qpi, dudt, dvdt, dwdt, &
         tracer, micro_field, sgs_field, sgs_field_diag, z, pres, prespot, presi, &
         rho, rhow, bet, sstxy, rank, nx, ny, nz, irecc
        close(65)
        if(masterproc) then
           write(66) version, &
            at, bt, ct, dt, dtn, dt3, time, dx, dy, dz, doconstdz,&
            day, day0, nstep, na, nb, nc, caseid, case, &
            dodamping, doupperbound, docloud, doprecip, doradhomo, dosfchomo,&
            dolongwave, doshortwave, dosgs, dosubsidence, dotracers,  dosmoke, &
            docoriolis, dosurface, dolargescale,doradforcing, dossthomo, &
            dosfcforcing, doradsimple, donudging_uv, donudging_tq, &
            dowallx, dowally, doperpetual, doseasons, &
            docup, docolumn, soil_wetness, dodynamicocean, ocean_type, &
            delta_sst, depth_slab_ocean, Szero, deltaS, timesimpleocean, &
            pres0, ug, vg, fcor, fcorz, tabs_s, z0, fluxt0, fluxq0, tau0, &
            tauls, tautqls, timelargescale, epsv, nudging_uv_z1, nudging_uv_z2, &
            donudging_t, donudging_q, doisccp, domodis, domisr, dosimfilesout, & 
            dosolarconstant, solar_constant, zenith_angle, notracegases, &
            doSAMconditionals, dosatupdnconditionals, &
            nudging_t_z1, nudging_t_z2, nudging_q_z1, nudging_q_z2, &
            cem, les, ocean, land, sfc_flx_fxd, sfc_tau_fxd, &
            nrad, nxco2, latitude0, longitude0, dofplane, &
            docoriolisz, doradlon, doradlat, doseawater, salt_factor, &
            ntracers, nmicro_fields, nsgs_fields, nsgs_fields_diag
            close(66)
        end if
        if(rank.eq.nsubdomains-1) then
            print *,'Restart file was saved. nstep=',nstep
        endif

        return
        end




        subroutine read_statement()

        use vars
        use microphysics, only: micro_field, nmicro_fields
        use sgs, only: sgs_field, nsgs_fields, sgs_field_diag, nsgs_fields_diag
        use tracers
        use params
        use movies, only: irecc
        implicit none
        integer  nx1, ny1, nz1, rank1, ntr, nmic, nsgs, nsgsd
        character(100) case1,caseid1
        character(6) version1

        read(65)  &
         u, v, w, t, p, qv, qcl, qci, qpl, qpi, dudt, dvdt, dwdt, &
         tracer, micro_field, sgs_field, sgs_field_diag, z, pres, prespot, presi, &
         rho, rhow, bet, sstxy, rank1, nx1, ny1, nz1, irecc
        close(65)
        read(66) version1, &
            at, bt, ct, dt, dtn, dt3, time, dx, dy, dz, doconstdz, &
            day, day0, nstep, na, nb, nc, caseid1(1:sizeof(caseid)), case1(1:sizeof(case)), &
            dodamping, doupperbound, docloud, doprecip, doradhomo, dosfchomo,&
            dolongwave, doshortwave, dosgs, dosubsidence, dotracers,  dosmoke, &
            docoriolis, dosurface, dolargescale,doradforcing, dossthomo, &
            dosfcforcing, doradsimple, donudging_uv, donudging_tq, &
            dowallx, dowally, doperpetual, doseasons, &
            docup, docolumn, soil_wetness, dodynamicocean, ocean_type,&
            delta_sst, depth_slab_ocean, Szero, deltaS, timesimpleocean, &
            pres0, ug, vg, fcor, fcorz, tabs_s, z0, fluxt0, fluxq0, tau0, &
            tauls, tautqls, timelargescale, epsv, nudging_uv_z1, nudging_uv_z2, &
            donudging_t, donudging_q, doisccp, domodis, domisr, dosimfilesout,  &
            dosolarconstant, solar_constant, zenith_angle, notracegases, &
            doSAMconditionals, dosatupdnconditionals, &
            nudging_t_z1, nudging_t_z2, nudging_q_z1, nudging_q_z2, &
            cem, les, ocean, land, sfc_flx_fxd, sfc_tau_fxd, &
            nrad, nxco2, latitude0, longitude0, dofplane, &
            docoriolisz, doradlon, doradlat, doseawater, salt_factor, &
            ntr, nmic, nsgs, nsgsd
        close(66)

        if(version1.ne.version) then
          print *,'Wrong restart file!'
          print *,'Version of SAM that wrote the restart files:',version1
          print *,'Current version of SAM',version
          call task_abort()
        end if
        if(nrestart.ne.3) then
          if(rank.ne.rank1) then
             print *,'Error: rank of restart data is not the same as rank of the process'
             print *,'rank1=',rank1,'   rank=',rank
          endif
          if(nx.ne.nx1.or.ny.ne.ny1.or.nz.ne.nz1) then
             print *,'Error: domain dims (nx,ny,nz) set by grid.f'
             print *,' not correspond to ones in the restart file.'
             print *,'in executable:   nx, ny, nz:',nx,ny,nz
             print *,'in restart file: nx, ny, nz:',nx1,ny1,nz1
             print *,'Exiting...'
             call task_abort()
          endif
        end if
        if(nmic.ne.nmicro_fields) then
           print*,'Error: number of micro_field in restart file is not the same as nmicro_fields'
           print*,'nmicro_fields=',nmicro_fields,'   in file=',nmic
           print*,'Exiting...'
        end if
        if(nsgs.ne.nsgs_fields.or.nsgsd.ne.nsgs_fields_diag) then
           print*,'Error: number of sgs_field in restart file is not the same as nsgs_fields'
           print*,'nsgs_fields=',nsgs_fields,'   in file=',nsgs
           print*,'nsgs_fields_diag=',nsgs_fields_diag,'   in file=',nsgsd
           print*,'Exiting...'
        end if
        if(ntr.ne.ntracers) then
           print*,'Error: number of tracers in restart file is not the same as ntracers.'
           print*,'ntracers=',ntracers,'   ntracers(in file)=',ntr
           print*,'Exiting...'
        end if
        close(65)
        if(rank.eq.nsubdomains-1) then
           print *,'Case:',caseid
           print *,'Restarting at step:',nstep
           print *,'Time(s):',nstep*dt
        endif

        return
        end
 
