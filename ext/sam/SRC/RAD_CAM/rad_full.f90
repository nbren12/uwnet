subroutine rad_full()

  ! Interface to the longwave and shortwave radiation code from the
  ! NCAR Community Atmosphere Model (CAM3.0).
  !
  ! Originally written as interface to CCM3 radiation code by Marat
  !     Khairoutdinov
  ! Adapted to CAM3.0 radiation code by Peter Blossey, August 2004.
  !
  use rad
  use ppgrid
  use vars
  use params
  use shr_orb_mod, only: shr_orb_params
  use radae,        only: radaeini, initialize_radbuffer
  use pkg_cldoptics, only: cldefr, cldems
  use aer_optics, only: aer_optics_initialize
  use microphysics, only: reffc, reffi, &
         SnowMassMixingRatio, reffs, dosnow_radiatively_active

  implicit none

  ! Local space:

  real(r4) pmid(pcols,pver)	! Level pressure (Pa)
  real(r4) pint(pcols,pverp)	! Model interface pressure (Pa)
  real(r4) massl(pcols,pver)	! Level mass (g/m2)
  real(r4) pmidrd(pcols,pver)	! Level pressure (dynes/cm2)
  real(r4) pintrd(pcols,pverp)	! Model interface pressure (dynes/cm2)
  real(r4) pmln(pcols,pver)	! Natural Log of pmid
  real(r4) piln(pcols,pverp)	! Natural Log of pint
  real(r4) tlayer(pcols,pver)	! Temperature
  real(r4) qlayer(pcols,pver)	! Specific humidity
  real(r4) cld(pcols,pverp)	! Fractional cloud cover
  real(r4) cliqwp(pcols,pver)	! Cloud liquid water path
  real(r4) cicewp(pcols,pver)	! Cloud ice water path

  real(r4) fice(pcols,pver)	! Fractional ice content within cloud
  real(r4) rel(pcols,pver)	! Liquid effective drop radius (micron)
  real(r4) rei(pcols,pver)	! Ice effective drop size
  real(r4) o3vmr(pcols,pver)	! Ozone volume mixing ratio
  real(r4) o3mmr(pcols,pver)		! Ozone mass mixing ratio

  !bloss(2016-02-09): add variables to handle radiatively active snow
  real(r4) SnowWaterPath(pcols,pver)	! Snow water path
  real(r4) re_snow(pcols,pver)	! Snow effective particle size
  real(r4) CloudTauLW(pcols,pver)	! cloud optical depth in longwave

  integer lchnk              ! chunk identifier
  integer ncol               ! number of atmospheric columns
  integer nmxrgn(pcols)      ! Number of maximally overlapped regions

  real(r4) emis(pcols,pver)     ! cloud emissivity (fraction)
  real(r4) landfrac(pcols)      ! Land fraction (seems deprecated)
  real(r4) icefrac(pcols)       ! Ice fraction
  real(r4) psurface(pcols)      ! Surface pressure
  real(r4) player(pcols,pver)   ! Midpoint pressures
  real(r4) landm(pcols)         ! Land fraction
  real(r4) snowh(pcols)         ! snow depth, water equivalent (meters)

  real(r4) pmxrgn(pcols,pverp)  ! Maximum values of pmid for each

  real(r4) qrl(pcols,pver)	! Longwave heating rate (K/s)
  real(r4) qrs(pcols,pver)	! Shortwave heating rate (K/s)

  real(r4) fnl(pcols,pverp)	! Net Longwave Flux at interfaces
  real(r4) fns(pcols,pverp)	! Net Shortwave Flux at interfaces
  real(r4) fcnl(pcols,pverp)	! Net Clearsky Longwave Flux at interfaces
  real(r4) fcns(pcols,pverp)	! Net Clearsky Shortwave Flux at interfaces
  real(r4) flu(pcols,pverp)	! Longwave upward flux
  real(r4) fld(pcols,pverp)	! Longwave downward flux
  real(r4) fsu(pcols,pverp)	! Shortwave upward flux
  real(r4) fsd(pcols,pverp)	! Shortwave downward flux

  !	aerosols:

  real(r4) rh(pcols,pver)		! relative humidity for aerorsol 
  real(r4) aer_mass(pcols,pver,naer_all)  ! aerosol mass mixing ratio
  integer, parameter :: nspint = 19 ! # spctrl intrvls in solar spectrum
  integer, parameter :: naer_groups= 7 ! # aerosol grp for opt diagnostcs
  real(r4) aertau(nspint,naer_groups) ! Aerosol column optical depth
  real(r4) aerssa(nspint,naer_groups) ! Aero col-avg single scattering albedo
  real(r4) aerasm(nspint,naer_groups) ! Aerosol col-avg asymmetry parameter
  real(r4) aerfwd(nspint,naer_groups) ! Aerosol col-avg forward scattering

  !       Diagnostics:

  ! Longwave radiation
  real(r4) flns(pcols)          ! Surface cooling flux
  real(r4) flnt(pcols)          ! Net outgoing flux
  real(r4) flnsc(pcols)         ! Clear sky surface cooing
  real(r4) flntc(pcols)         ! Net clear sky outgoing flux
  real(r4) flwds(pcols)         ! Down longwave flux at surface

  !bloss: New in CAM3.0.
  real(r4) flut(pcols)          ! Upward flux at top of model
  real(r4) flutc(pcols)         ! Upward clear-sky flux at top of model

  ! Shortwave radiation
  real(r4) solin(pcols)        ! Incident solar flux
  real(r4) fsns(pcols)         ! Surface absorbed solar flux
  real(r4) fsnt(pcols)         ! Flux Shortwave Downwelling Top-of-Model
  real(r4) fsntoa(pcols)      ! Total column absorbed solar flux
  real(r4) fsds(pcols)         ! Flux Shortwave Downwelling Surface
  real(r4) fsdsc(pcols)        ! Clearsky Flux Shortwave Downwelling Surface

  real(r4) fsnsc(pcols)        ! Clear sky surface absorbed solar flux
  real(r4) fsntc(pcols)        ! Clear sky total column absorbed solar flx
  real(r4) fsntoac(pcols)      ! Clear sky total column absorbed solar flx
  real(r4) sols(pcols)         ! Direct solar rad incident on surface (< 0.7)
  real(r4) soll(pcols)         ! Direct solar rad incident on surface (>= 0.7)
  real(r4) solsd(pcols)        ! Diffuse solar rad incident on surface (< 0.7)
  real(r4) solld(pcols)        ! Diffuse solar rad incident on surface (>= 0.7)
  real(r4) fsnirtoa(pcols)     ! Near-IR flux absorbed at toa
  real(r4) fsnrtoac(pcols)     ! Clear sky near-IR flux absorbed at toa
  real(r4) fsnrtoaq(pcols)     ! Near-IR flux absorbed at toa >= 0.7 microns

  real(r4) frc_day(pcols)      ! = 1 for daylight, =0 for night columns
  real(r4) coszrs_in(pcols)    ! cosine of solar zenith angle

  real(r4) asdir(pcols)     ! Srf alb for direct rad   0.2-0.7 micro-ms
  real(r4) aldir(pcols)     ! Srf alb for direct rad   0.7-5.0 micro-ms
  real(r4) asdif(pcols)     ! Srf alb for diffuse rad  0.2-0.7 micro-ms
  real(r4) aldif(pcols)     ! Srf alb for diffuse rad  0.7-5.0 micro-ms

  real(r4) lwupsfc(pcols)   ! Longwave up flux in CGS units
  real(r4) temp_surf(pcols)   ! Longwave up flux in CGS units

  real(r4) qtot
  real(r4) dayy, lat_r4
  integer i,j,k,m,ii,jj,i1,j1,tmp_count,nrad_call
  integer iday, iday0
  real(r4) coef,factor,tmp(1)
  real(8) qradz(nzm),buffer(nzm)
  real perpetual_factor
  real(r4) clat(pcols),clon(pcols)
  real(r4) pii
  real(r4) tmp_ggr, tmp_cp, tmp_eps, tmp_ste, tmp_pst
  
  if(icycle.ne.1) goto 999  ! ugly way to handle the subcycles. add rad heating.

  nrad_call = 3600./dt
  pii = atan2(0.,-1.)

  ncol = 1 ! compute one column of radiation at a time.

  !-------------------------------------------------------
  ! Initialize some stuff
  !


  if(initrad) then

     ! check whether this processor's portion of the domain is
     ! evenly divisible into chunks of size ndiv (over which
     ! the absorbtivity/emissivity computations will be performed).
     if(mod(nx,ndiv).ne.0.or.(RUN3D.and.mod(ny,ndiv).ne.0)) then
        if(masterproc) print*,'nx or ny is not divisible by ndiv'
        if(masterproc) print*,'set in RAD_CAM/rad.f90'
        if(masterproc) print*,'Stop.'
        call abort()
     end if

     ! set up size and number of chunks of data for abs/ems computations
     nxdiv=max(1,nx/ndiv)
     nydiv=max(1,ny/ndiv)
     begchunk = 1
     endchunk = nxdiv*nydiv

     !bloss  subroutine initialize_radbuffer
     !bloss  inputs:  none
     !bloss  ouptuts: none (allocates and initializes abs/ems arrays)
     call initialize_radbuffer()

     !bloss  subroutine shr_orb_params
     !bloss  inputs:  iyear, log_print
     !bloss  ouptuts: eccen, obliq, mvelp, obliqr, lambm0, mvelpp
     call shr_orb_params(iyear    , eccen  , obliq , mvelp     ,     &
           &               obliqr   , lambm0 , mvelpp, .false.)

     !bloss  subroutine radaeini
     !bloss  inputs:  pstdx (=1013250 dynes/cm2), mwdry (mwair) and mwco2.
     !bloss  ouptuts: none (sets up lookup tables for abs/ems computat.)
     call radaeini( 1.013250e6_r4, mwdry, mwco2 )

     !bloss  subroutine aer_optics_initialize
     !bloss  inputs:  none
     !bloss  ouptuts: none (sets up lookup tables for aerosol properties)
     call aer_optics_initialize()

     ! sets up initial mixing ratios of trace gases.
     call tracesini()

     if(nrestart.eq.0) then

        do k=1,nzm
           do j=1,ny
              do i=1,nx
	         tabs_rad(i,j,k)=0.
	         qv_rad(i,j,k)=0.
	         qc_rad(i,j,k)=0.
	         qi_rad(i,j,k)=0.
	         cld_rad(i,j,k)=0.
	         rel_rad(i,j,k)=25.
	         rei_rad(i,j,k)=25.
	         qrad(i,j,k)=0.
              end do
           end do
        end do
        nradsteps=0	  
        do k=1,nz
           radlwup(k) = 0.
           radlwdn(k) = 0.
           radswup(k) = 0.
           radswdn(k) = 0.
           radqrlw(k) = 0.
           radqrsw(k) = 0.
           radqrclw(k) = 0.
           radqrcsw(k) = 0.
        end do

        if(compute_reffc) rel_rad(:,:,:) = 0.
        if(compute_reffi) rei_rad(:,:,:) = 0.

        if(dosnow_radiatively_active) then
          qs_rad(:,:,:) = 0.
          res_rad(:,:,:) = 0.
        end if

     else

        call read_rad()

     endif

     if(doperpetual) then
           ! perpetual sun (no diurnal cycle)
           do j=1,ny
              do i=1,nx
                 p_factor(i,j) = perpetual_factor(day0, latitude(i,j)&
                      &,longitude(i,j))
              end do
           end do
     end if

  endif

  !bloss  subroutine radini
  !bloss  inputs:  ggr, cp, epislo (=0.622), stebol (=5.67e-8), pstd
  !bloss  outputs: none (although it initializes constants, computes
  !                       ozone path lengths).
  tmp_ggr = ggr
  tmp_cp  = cp
  tmp_eps = mwh2o/mwdry
  tmp_ste = 5.67e-8_r4
  tmp_pst = 1.013250e6_r4
  call radini(tmp_ggr, tmp_cp, tmp_eps, tmp_ste, tmp_pst)
  
  !bloss  initialize co2 mass mixing ratio
  co2vmr = 3.550e-4_r4 * nxco2
  co2mmr = co2vmr*rmwco2 ! rmwco2: ratio of mw of co2 to that of dry air
  if ((nstep.eq.1).and.(icycle.eq.1)) then
     if (masterproc) write(*,*) 'CO2 VMR = ', co2vmr
     if (masterproc) write(*,*) 'CO2 MMR = ', co2mmr
  end if

  ! compute pressure levels using pressure at the top
  ! interface and hydrostatic balance below that level.  This
  ! may be important because the radiation code assumes that
  ! the mass of a layer is equal to the difference between the
  ! pressure levels at the upper and lower interfaces.
  pint(:,1)=presi(nz)*100.
  piln(:,1) = log(pint(:,1))
  do k = nzm,1,-1
     m=nz-k
     pint(:,m+1) = pint(:,m) + rho(k)*ggr*(zi(k+1)-zi(k))
     piln(:,m+1) = log(pint(:,m+1))
  end do
  do m=1,nzm
     pmid(:,m) = 0.5*(pint(:,m)+pint(:,m+1))
     pmln(:,m) = log(pmid(:,m))
  end do

  do k=1,nzm
     massl(:,k)=1000.*(pint(:,k+1)-pint(:,k))/ggr
     rh(:,k)=0.
     o3vmr(:,k)=0.6034*o3(k)
     qrl(:,k)=0.
     qrs(:,k)=0.
  end do
  do k=1,nz
     cld(:,k)=0.
     flu(:,k)=0.
     fld(:,k)=0.
     fsu(:,k)=0.
     fsd(:,k)=0.
  end do

  ! Initialize aerosol mass mixing ratio to zero.
  ! TODO: come up with scheme to input aerosol concentrations 
  ! similar to the current scheme for trace gases.
  aer_mass = 0.

  !-----------------------------------------------------------
  ! Check if it is time to compute gas absortion coefficients for
  ! longwave radiation. This coefficients are computed for
  ! horizontally average fields and storred internally, so
  ! they need not to be recalculated on every call of radclw() for
  ! efficiency reasons.

  if(initrad.or.mod(nstep,nrad_call).eq.0) then

     initrad=.false.

     lchnk = 0
     do jj=1,nydiv 
        j1=(jj-1)*(ny/nydiv) 
        do ii=1,nxdiv 
	   i1=(ii-1)*(nx/nxdiv) 
           lchnk = lchnk + 1
	   do k=1,nzm
              tlayer(:,k)=0.
              qlayer(:,k)=0.
              cld(:,k) = 0.
              cliqwp(:,k) = 0.
              fice(:,k) = 0.
              m=nz-k

              tmp_count = 0
              do j=j1+1,j1+(ny/nydiv)
                 do i=i1+1,i1+(nx/nxdiv)	 
                    tlayer(1,k)=tlayer(1,k)+tabs(i,j,m)
                    qlayer(1,k)=qlayer(1,k)+qv(i,j,m)
                    tmp_count = tmp_count+1
                 end do
              end do
              tlayer(1,k)=tlayer(1,k)/float(tmp_count)
              qlayer(1,k)=max(1.e-7,qlayer(1,k)/float(tmp_count))

	   end do

           !bloss  subroutine radinp
           !bloss  inputs:  lchnk, ncol, pmid, pint, o3vmr
           !bloss  outputs: pmidrd,  ! Pressure at mid-levels (dynes/cm*2)
           !                pintrd,  ! Pressure at interfaces (dynes/cm*2)
           !                eccf,    ! Earth-sun distance factor
           !                o3mmr    ! Ozone mass mixing ratio
           call radinp(lchnk,ncol,pmid,pint,o3vmr,pmidrd, &
                pintrd,eccf,o3mmr)
           if ((nstep.eq.1).and.(icycle.eq.1).and.(lchnk.eq.1)) then
              if (masterproc) write(*,*) 'Eccentricity ', eccen
           end if

           if(sstxy(1,1).le.-100.) then
             print*,'rad: sst is undefined. Quitting...'
             call task_abort()
           end if

           lwupsfc(1) = stebol*(sstxy(1,1)+t00)**4 ! CGS units

           !bloss: Set flag for absorbtivity/emissivity computation
           doabsems = .true. 

           ! bloss: Set number of maximally overlapped regions and maximum 
           !        pressure so that only a single region will be computed.
           nmxrgn = 1
           pmxrgn = 1.2e6

           !bloss  subroutine radclwmx
           !bloss  inputs:  lchnk, ncol, nmxrgn, pmxrgn, lwupsfc,
           !                tlayer, qlayer, o3vmr, pmid, pint, pmln, piln,
           !                n2o, ch4, cfc11, cfc12, cld, emis, aer_mass
           !bloss  outputs: qrl,   ! Longwave heating rate
           !                flns,  ! Surface cooling flux
           !                flnt,  ! Net outgoing flux
           !                flut,  ! Upward flux at top of model
           !                flnsc, ! Clear sky surface cooing
           !                flntc, ! Net clear sky outgoing flux
           !                flutc, ! Upward clear-sky flux at top of model
           !                flwds, ! Down longwave flux at surface
           !                fcnl,  ! clear sky net flux at interfaces
           !                fnl    ! net flux at interfaces 
           call radclwmx(lchnk   ,ncol    ,                   &
                lwupsfc ,tlayer  ,qlayer  ,o3vmr   , &
                pmidrd  ,pintrd  ,pmln    ,piln    ,          &
                n2o     ,ch4     ,cfc11   ,cfc12   , &
                cld     ,emis    ,pmxrgn  ,nmxrgn  ,qrl     , &
                flns    ,flnt    ,flnsc   ,flntc   ,flwds   , &
                flut    ,flutc   , &
                aer_mass,fnl     ,fcnl    ,flu     ,fld)
        end do
     end do

  endif


  !------------------------------------------------------
  !  Accumulate thermodynamical fields over nrad steps 
  !

  do k=1,nzm
     do j=1,ny
        do i=1,nx
           tabs_rad(i,j,k)=tabs_rad(i,j,k)+tabs(i,j,k)
           qv_rad(i,j,k)=qv_rad(i,j,k)+qv(i,j,k)
        end do
     end do
  end do

  if(dosnow_radiatively_active) then
    ! accumulate cloud liquid, cloud ice and snow mass mixing ratios
    do k=1,nzm
      do j=1,ny
        do i=1,nx
          qc_rad(i,j,k)=qc_rad(i,j,k)+qcl(i,j,k)
          qi_rad(i,j,k)=qi_rad(i,j,k)+qci(i,j,k)
          qs_rad(i,j,k)=qs_rad(i,j,k)+SnowMassMixingRatio(i,j,k)
          if(qcl(i,j,k)+qci(i,j,k)+SnowMassMixingRatio(i,j,k).gt.0.) cld_rad(i,j,k) = cld_rad(i,j,k)+1.
        end do
      end do
    end do
  else
    ! accumulate cloud liquid and cloud ice mass mixing ratios
    do k=1,nzm
      do j=1,ny
        do i=1,nx
          qc_rad(i,j,k)=qc_rad(i,j,k)+qcl(i,j,k)
          qi_rad(i,j,k)=qi_rad(i,j,k)+qci(i,j,k)
          if(qcl(i,j,k)+qci(i,j,k).gt.0.) cld_rad(i,j,k) = cld_rad(i,j,k)+1.
        end do
      end do
    end do
  end if
  ! Accumulate effective radius by weighting it with mass
  if(compute_reffc) then
    rel_rad(1:nx,1:ny,1:nzm) = rel_rad(1:nx,1:ny,1:nzm) + reffc(1:nx,1:ny,1:nzm) * qcl(1:nx,1:ny,1:nzm) 
  end if
  if(compute_reffi) then
    rei_rad(1:nx,1:ny,1:nzm) = rei_rad(1:nx,1:ny,1:nzm) + reffi(1:nx,1:ny,1:nzm) * qci(1:nx,1:ny,1:nzm) 
  end if
  if(dosnow_radiatively_active) then
    res_rad(1:nx,1:ny,1:nzm) = res_rad(1:nx,1:ny,1:nzm) + reffs(1:nx,1:ny,1:nzm) * SnowMassMixingRatio(1:nx,1:ny,1:nzm) 
  end if
  nradsteps=nradsteps+1

  !----------------------------------------------------
  ! Update radiation variables if the time is due
  !

  !kzm Oct.14, 03 changed .eq.nrad to .ge.nrad to handle the
  ! case when a smaller nrad is used in restart  
  if(nstep.eq.1.or.nradsteps.ge.nrad) then 

     ! Compute radiation fields for averaged thermodynamic fields


     coef=1./float(nradsteps)

     do k=1,nz
        radlwup(k) = 0.
        radlwdn(k) = 0.
        radswup(k) = 0.
        radswdn(k) = 0.
        radqrlw(k) = 0.
        radqrsw(k) = 0.
        radqrclw(k) = 0.
        radqrcsw(k) = 0.
     end do

     if(compute_reffc) then
       do k=1,nzm
          do j=1,ny
             do i=1,nx
                 rel_rad(i,j,k) = max(2.5,min(60.,rel_rad(i,j,k)/(1.e-8+qc_rad(i,j,k))))
             end do
          end do
       end do
     end if
     if(compute_reffi) then
       do k=1,nzm
          do j=1,ny
             do i=1,nx
                 rei_rad(i,j,k) = max(5.,min(250.,rei_rad(i,j,k)/(1.e-8+qi_rad(i,j,k))))
             end do
          end do
       end do
     end if
     do k=1,nzm
        do j=1,ny
           do i=1,nx
	      tabs_rad(i,j,k)=tabs_rad(i,j,k)*coef
	      qv_rad(i,j,k)=qv_rad(i,j,k)*coef
	      qc_rad(i,j,k)=qc_rad(i,j,k)*coef
	      qi_rad(i,j,k)=qi_rad(i,j,k)*coef
	      cld_rad(i,j,k)=cld_rad(i,j,k)*coef
           end do
        end do
     end do

     if(dosnow_radiatively_active) then
       !bloss(2016-02-09): Compute average snow effective radius and mass mixing ratio.
       !  Note that the average effective radius is mass-weighted.
       do k=1,nzm
         do j=1,ny
           do i=1,nx
             res_rad(i,j,k) = max(5.,min(250.,res_rad(i,j,k)/(1.e-8+qs_rad(i,j,k))))
           end do
         end do
       end do
       qs_rad(:,:,:)=qs_rad(:,:,:)*coef
     end if


     lchnk = 0
     do j=1,ny
        jj = (j-1)/(ny/nydiv)+1
        do i=1,nx
	   ii = (i-1)/(nx/nxdiv)+1
           lchnk = ii + (jj-1)*nxdiv
           do k=1,nzm
              m=nz-k
              tlayer(1,k)=tabs_rad(i,j,m)
              qtot = qc_rad(i,j,m)+qi_rad(i,j,m)
              qlayer(1,k)=max(1.e-7_r4,qv_rad(i,j,m))
              if(qtot.gt.0.) then
	         cliqwp(1,k) = qtot*massl(1,k)
	         fice(1,k) = qi_rad(i,j,m)/qtot
	         cld(1,k) = min(0.99_r4,cld_rad(i,j,m))
              else
                 if(dosnow_radiatively_active) then
                   cld(1,k) = min(0.99_r4,cld_rad(i,j,m))
                 else
                   cld(1,k) = 0.
                 end if
	         cliqwp(1,k) = 0.
	         fice(1,k) = 0.
              endif
           end do

        ! Default reff computation:
           !bloss  subroutine cldefr
           !bloss  inputs:  lchnk, ncol, landfrac, icefrac, pres0, 
           !                pmid, landm, icrfrac, snowh
           !bloss  outputs: rel, rei (liq/ice effective radii)
           player = pmid
           psurface = 100.*pres0
           icefrac = 0.
           snowh = 0.
           landm = 0.
           landfrac = 0.
           if (.not.OCEAN) then
              landfrac = 1.
              landm = 1.
           end if
           call cldefr(lchnk,ncol,landfrac,tlayer,rel,rei, & ! CAM3 interface
              psurface,player,landm,icefrac, snowh)
           if(compute_reffc) then
             rel(1,nzm:1:-1) = rel_rad(i,j,:)
           else
             rel_rad(i,j,:) = rel(1,nzm:1:-1)
           end if
           if(compute_reffi) then
             rei(1,nzm:1:-1) = rei_rad(i,j,:)
           else
             rei_rad(i,j,:) = rei(1,nzm:1:-1)
           end if

           !bloss  subroutine cldems
           !bloss  inputs:  lchnk, ncol, cliqwp, rei, fice
           !bloss  outputs: emis (cloud emissivity)
           call cldems(lchnk,ncol,cliqwp,fice,rei,emis)

           if(dosnow_radiatively_active) then
             !bloss(2016-02-08): Make snow radiatively active in longwave

             ! get the combined longwave optical depth for cloud liquid and cloud ice
             cloudTauLW(:,:) = - log( 1. - emis(:,:) )

             ! get effective radius and layer water path for snow.
             !   Remember to reverse indices in vertical.
             re_snow(1,1:nzm) = res_rad(i,j,nzm:1:-1)
             do k = 1,nzm
               m=nz-k
               SnowWaterPath(1,k) = qs_rad(i,j,m)*massl(1,k)
             end do

             ! taken from cldems
             !   ice absorption coefficient is kabsi = 0.005 + 1. / rei
             !   LW optical depth is 1.66*kabsi*iwp
             cloudTauLW(:,:) = cloudTauLW(:,:) &
                  + 1.66 * ( 0.005 + 1. / re_snow(:,:) ) * SnowWaterPath(:,:) 

             ! re-compute emissivity
             emis(:,:) = 1. - exp( - cloudTauLW(:,:) )

             ! Note that cloud fraction already accounts for presence of snow
           else
             ! zero out snow water path and effective radius
             re_snow(:,:) = 0.
             SnowWaterPath(:,:) = 0.
           end if


           !bloss  subroutine radinp
           !bloss  inputs:  lchnk, ncol, pmid, pint, o3vmr
           !bloss  outputs: pmidrd, pintrd, eccf, o3mmr
           call radinp(lchnk,ncol,pmid,pint,o3vmr,pmidrd, &
                pintrd,eccf,o3mmr)

           lwupsfc(1) = stebol*(sstxy(i,j)+t00)**4 ! CGS units

           if(dolongwave) then

              !bloss: Set flag for absorbtivity/emissivity computation
              doabsems = .false. 

              ! bloss: Set number of maximally overlapped regions 
              !        and maximum pressure so that only a single 
              !        region will be computed.
              nmxrgn = 1
              pmxrgn = 1.2e6

              !bloss  subroutine radclwmx
              !bloss  inputs:  lchnk, ncol, nmxrgn, pmxrgn, lwupsfc,
              !                tlayer, qlayer, o3vmr, pmid, pint, pmln, piln,
              !                n2o, ch4, cfc11, cfc12, cld, emis, aer_mass
              !bloss  outputs: qrl,   ! Longwave heating rate
              !                flns,  ! Surface cooling flux
              !                flnt,  ! Net outgoing flux
              !                flut,  ! Upward flux at top of model
              !                flnsc, ! Clear sky surface cooing
              !                flntc, ! Net clear sky outgoing flux
              !                flutc, ! Upward clear-sky flux at top of model
              !                flwds, ! Down longwave flux at surface
              !                fcnl,  ! clear sky net flux at interfaces
              !                fnl    ! net flux at interfaces 
              call radclwmx(lchnk   ,ncol    ,                   &
                   lwupsfc ,tlayer  ,qlayer  ,o3vmr   , &
                   pmidrd  ,pintrd  ,pmln    ,piln    ,          &
                   n2o     ,ch4     ,cfc11   ,cfc12   , &
                   cld     ,emis    ,pmxrgn  ,nmxrgn  ,qrl     , &
                   flns    ,flnt    ,flnsc   ,flntc   ,flwds   , &
                   flut    ,flutc   , &
                   aer_mass,fnl     ,fcnl    ,flu    ,fld)
              ! convert radiative heating from units of J/kg/s to K/s
              qrl = qrl/cp
              !
              ! change toa/surface fluxes from cgs to mks units
              !
              flnt     = 1.e-3*flnt
              flntc    = 1.e-3*flntc
              flns     = 1.e-3*flns
              flnsc    = 1.e-3*flnsc
              flwds    = 1.e-3*flwds
              flut     = 1.e-3*flut
              flutc    = 1.e-3*flutc
           endif

           if(doshortwave) then

              if (doseasons) then
                 ! The diurnal cycle of insolation will vary
                 ! according to time of year of the current day.
                 dayy = day
              else
                 ! The diurnal cycle of insolation from the calendar
                 ! day on which the simulation starts (day0) will be
                 ! repeated throughout the simulation.
                 iday0 = day0
                 iday = day
                 dayy = day-iday
                 dayy = iday0 + dayy
              end if
              if(doperpetual) then
                 if (dosolarconstant) then
                    ! fix solar constant and zenith angle as specified
                    ! in prm file.
                    coszrs_in(1) = cos(zenith_angle*pii/180.)
                    eccf = solar_constant/(1367.)
                 else
                    ! perpetual sun (no diurnal cycle) - Modeled after Tompkins
                    coszrs_in(1) = 0.637 ! equivalent to zenith angle of 50.5 deg
                    eccf = p_factor(i,j)/coszrs_in(1) ! Adjst solar constant
                 end if
              elseif (doequinox) then
                 ! this section is added by noah
                 eccf = 1.

                 ! covert real types to 4 byte reals
                 ! needed if compiled in double precision
                 dayy = day
                 lat_r4 = latitude(i,j)
                 call equinox_cos_zenith_angle(dayy, lat_r4, coszrs_in(1))
              else
                 !bloss  subroutine zenith
                 !bloss  inputs:  dayy, latitude, longitude, ncol
                 !bloss  outputs: coszrs  ! Cosine solar zenith angle
                 clat(1) = pie*latitude(i,j)/180.
                 clon(1) = pie*longitude(i,j)/180.
                 coszrs_in(1) = coszrs
                 call zenith(dayy,clat,clon,coszrs_in,ncol)
              end if

	      coszrs = coszrs_in(1) ! needed for the isccp simulator

              !bloss  subroutine albedo
              !bloss  inputs: OCEAN (land/ocean flag), coszrs_in
              !bloss  outputs: 
              !     asdir  ! Srf alb for direct rad   0.2-0.7 micro-ms
              !     aldir  ! Srf alb for direct rad   0.7-5.0 micro-ms
              !     asdif  ! Srf alb for diffuse rad  0.2-0.7 micro-ms
              !     aldif  ! Srf alb for diffuse rad  0.7-5.0 micro-ms
              temp_surf(:) = sstxy(1,1)+t00
              call albedo(1,1,OCEAN,coszrs_in,temp_surf,asdir,aldir,asdif,aldif)

              ! bloss: Set number of maximally overlapped regions 
              !        and maximum pressure so that only a single 
              !        region will be computed.
              nmxrgn = 1
              pmxrgn = 1.2e6

              !bloss: compute separate cloud liquid & ice water paths
              cicewp = fice*cliqwp
              cliqwp = cliqwp - cicewp

              !bloss: set up day fraction.
              frc_day(1) = 0.
              if (coszrs_in(1).gt.0.) frc_day(1) = 1.

              !bloss  subroutine radcswmx
              !bloss  inputs:  
              !     lchnk             ! chunk identifier
              !     ncol              ! number of atmospheric columns
              !     pmid     ! Level pressure
              !     pint     ! Interface pressure
              !     qlayer   ! Specific humidity (h2o mass mix ratio)
              !     o3mmr    ! Ozone mass mixing ratio
              !     aer_mass   ! Aerosol mass mixing ratio
              !     rh       ! Relative humidity (fraction)
              !     cld      ! Fractional cloud cover
              !     cicewp   ! in-cloud cloud ice water path
              !     cliqwp   ! in-cloud cloud liquid water path
              !     csnowp   ! in-cloud snow water path -- bloss(2016-02-09)
              !     rel      ! Liquid effective drop size (microns)
              !     rei      ! Ice effective drop size (microns)
              !     res      ! snow effective particle size (microns) -- bloss (2016-02-09)
              !     eccf     ! Eccentricity factor (1./earth-sun dist^2)
              !     coszrs_in! Cosine solar zenith angle
              !     asdir    ! 0.2-0.7 micro-meter srfc alb: direct rad
              !     aldir    ! 0.7-5.0 micro-meter srfc alb: direct rad
              !     asdif    ! 0.2-0.7 micro-meter srfc alb: diffuse rad
              !     aldif    ! 0.7-5.0 micro-meter srfc alb: diffuse rad
              !     scon     ! solar constant
              !bloss  in/outputs: 
              !     pmxrgn   ! Maximum values of pressure for each
              !              !    maximally overlapped region. 
              !     nmxrgn   ! Number of maximally overlapped regions
              !bloss  outputs: 
              !     solin     ! Incident solar flux
              !     qrs       ! Solar heating rate
              !     fsns      ! Surface absorbed solar flux
              !     fsnt      ! Total column absorbed solar flux
              !     fsntoa    ! Net solar flux at TOA
              !     fsds      ! Flux shortwave downwelling surface
              !     fsnsc     ! Clear sky surface absorbed solar flux
              !     fsdsc     ! Clear sky surface downwelling solar flux
              !     fsntc     ! Clear sky total column absorbed solar flx
              !     fsntoac   ! Clear sky net solar flx at TOA
              !     sols      ! Direct solar rad on surface (< 0.7)
              !     soll      ! Direct solar rad on surface (>= 0.7)
              !     solsd     ! Diffuse solar rad on surface (< 0.7)
              !     solld     ! Diffuse solar rad on surface (>= 0.7)
              !     fsnirtoa  ! Near-IR flux absorbed at toa
              !     fsnrtoac  ! Clear sky near-IR flux absorbed at toa
              !     fsnrtoaq  ! Net near-IR flux at toa >= 0.7 microns
              !     frc_day   ! = 1 for daylight, =0 for night columns
              !     aertau    ! Aerosol column optical depth
              !     aerssa    ! Aerosol column avg. single scattering albedo
              !     aerasm    ! Aerosol column averaged asymmetry parameter
              !     aerfwd    ! Aerosol column averaged forward scattering
              !     fns       ! net flux at interfaces
              !     fcns      ! net clear-sky flux at interfaces
              !     fsu       ! upward shortwave flux at interfaces
              !     fsd       ! downward shortwave flux at interfaces
              call radcswmx(lchnk   ,ncol    ,                   &
                   pintrd  ,pmid    ,qlayer  ,rh      ,o3mmr   , &
                   aer_mass  ,cld     ,cicewp  ,cliqwp  ,SnowWaterPath  ,rel     , &
                   rei     ,re_snow     ,eccf    ,coszrs_in,scon    ,solin   , &
                   asdir   ,asdif   ,aldir   ,aldif   ,nmxrgn  , &
                   pmxrgn  ,qrs     ,fsnt    ,fsntc   ,fsntoa  , &
                   fsntoac ,fsnirtoa,fsnrtoac,fsnrtoaq,fsns    , &
                   fsnsc   ,fsdsc   ,fsds    ,sols    ,soll    , &
                   solsd   ,solld   ,frc_day ,                   &
                   aertau  ,aerssa  ,aerasm  ,aerfwd  ,fns     , &
                   fcns    ,fsu     ,fsd     )
              ! convert radiative heating from units of J/kg/s to K/s
              qrs = qrs/cp
              !
              ! change toa/surface fluxes from cgs to mks units
              !
              fsnt     = 1.e-3*fsnt
              fsntc    = 1.e-3*fsntc
              fsntoa   = 1.e-3*fsntoa
              fsntoac  = 1.e-3*fsntoac
              fsnirtoa = 1.e-3*fsnirtoa
              fsnrtoac = 1.e-3*fsnrtoac
              fsnrtoaq = 1.e-3*fsnrtoaq
              fsns     = 1.e-3*fsns
              fsnsc    = 1.e-3*fsnsc
              fsds     = 1.e-3*fsds
              fsdsc    = 1.e-3*fsdsc

              solin    = 1.e-3*solin
           endif

           !
           ! Satellite simulator diagnostics using time-averaged values
           !
           if(doisccp .or. domodis .or. domisr) then 
             tau_067_cldliq (i,j,nzm:1:-1) = compute_tau_l(cliqwp(1,1:nzm), rel(1,1:nzm))
             tau_067_cldice (i,j,nzm:1:-1) = compute_tau_i(cicewp(1,1:nzm), rei(1,1:nzm)) 
             if(dosnow_radiatively_active) then
               tau_067_snow (i,j,nzm:1:-1) = compute_tau_i(SnowWaterPath(1,1:nzm), re_snow(1,1:nzm)) 
             else
               tau_067_snow (i,j,:) = 0. 
             end if
             tau_067 (i,j,:) = tau_067_cldliq (i,j,:) + tau_067_cldice (i,j,:) &
                  + tau_067_snow (i,j,:)
             emis_105(i,j,1:nzm) = emis(1,nzm:1:-1)
           end if 
           
           do k=1,nzm
              m=nz-k
              qrad(i,j,m)=qrl(1,k)+qrs(1,k)
              radlwup(m)=radlwup(m)+flu(1,k)*1.e-3
              radlwdn(m)=radlwdn(m)+fld(1,k)*1.e-3
              radqrlw(m)=radqrlw(m)+qrl(1,k)
              radswup(m)=radswup(m)+fsu(1,k)*1.e-3
              radswdn(m)=radswdn(m)+fsd(1,k)*1.e-3
              radqrsw(m)=radqrsw(m)+qrs(1,k)
              !bloss: clearsky heating rates
              radqrclw(m)=radqrclw(m)+(fcnl(1,k+1)-fcnl(1,k))/massl(1,k)/cp
              radqrcsw(m)=radqrcsw(m)-(fcns(1,k+1)-fcns(1,k))/massl(1,k)/cp
           enddo

           lwnsxy(i,j) = flns(1)
           swnsxy(i,j) = fsns(1)
           lwntxy(i,j) = flut(1)
           lwntmxy(i,j) = flnt(1)
           swntxy(i,j) = fsntoa(1)
           swntmxy(i,j) = fsnt(1)
           lwnscxy(i,j) = flnsc(1)
           swnscxy(i,j) = fsnsc(1)
           lwntcxy(i,j) = flntc(1)
           swntcxy(i,j) = fsntoac(1)
           swdsxy(i,j) = fsds(1)
           lwdsxy(i,j) = flwds(1)
           solinxy(i,j) = solin(1)

        end do
     end do

     ! MODIS simulator diagnostics
     if(domodis) then 
       rad_reffc(1:nx,1:ny,1:nzm) = rel_rad(1:nx,1:ny,1:nzm)
       rad_reffi(1:nx,1:ny,1:nzm) = rei_rad(1:nx,1:ny,1:nzm) 
     end if

     tabs_rad(:,:,:)=0.
     qv_rad(:,:,:)=0.
     qc_rad(:,:,:)=0.
     qi_rad(:,:,:)=0.
     cld_rad(:,:,:)=0.
     if(compute_reffc) then
       rel_rad(:,:,:) = 0. 
     end if
     if(compute_reffi) then
       rei_rad(:,:,:) = 0. 
     end if
     nradsteps=0
     
     if(dosnow_radiatively_active) then
       !bloss(2016-02-12): Zero out accumulated snow variables.
       qs_rad(:,:,:) = 0.
       res_rad(:,:,:) = 0.
     end if
     
     if(masterproc.and.doshortwave.and..not.doperpetual) &
          print*,'radiation: coszrs=',coszrs_in(1),' solin=',solin(1)
     if(masterproc.and.doshortwave.and.doperpetual) &
          print*,'radiation: perpetual sun, solin=',solin(1)
     if(masterproc.and..not.doshortwave) &
          print*,'longwave radiation is called'

   ! Homogenize radiation:

     if(doradhomo) then    

        factor = 1./dble(nx*ny)
        do k=1,nzm
          qradz(k) = 0.
           do j=1,ny
             do i=1,nx
              qradz(k) = qradz(k) + qrad(i,j,k)
             end do
           end do
           qradz(k) = qradz(k) * factor
           buffer(k) = qradz(k)
        end do

        factor = 1./float(nsubdomains)
        if(dompi) call task_sum_real8(qradz,buffer,nzm)

        do k=1,nzm
           qradz(k)=buffer(k)*factor
           do j=1,ny
             do i=1,nx
               qrad(i,j,k) = qradz(k) 
             end do
           end do
        end do

     end if



  endif ! (nradsteps.eq.nrad) 

!--------------------------------------------------------
! Prepare statistics:
  
  do j=1,ny
     do i=1,nx
        ! Net surface and toa fluxes
        lwns_xy(i,j) = lwns_xy(i,j) + lwnsxy(i,j) 
        swns_xy(i,j) = swns_xy(i,j) + swnsxy(i,j)
        lwnt_xy(i,j) = lwnt_xy(i,j) + lwntxy(i,j) 
        swnt_xy(i,j) = swnt_xy(i,j) + swntxy(i,j)
        lwnsc_xy(i,j) = lwnsc_xy(i,j) + lwnscxy(i,j)
        swnsc_xy(i,j) = swnsc_xy(i,j) + swnscxy(i,j)
        lwntc_xy(i,j) = lwntc_xy(i,j) + lwntcxy(i,j)
        swntc_xy(i,j) = swntc_xy(i,j) + swntcxy(i,j)
        ! TOA Insolation
        solin_xy(i,j) = solin_xy(i,j) + solinxy(i,j)
     end do
  end do
!----------------------------------------------------------------
  if(dostatisrad) then

     do j=1,ny
        do i=1,nx
           s_flns = s_flns + lwnsxy(i,j) 
           s_flnsc = s_flnsc + lwnscxy(i,j) 
           s_flnt = s_flnt + lwntmxy(i,j) 
           s_flntoa = s_flntoa + lwntxy(i,j) 
           s_flntoac = s_flntoac + lwntcxy(i,j) 
           s_fsnt = s_fsnt + swntmxy(i,j) 
           s_fsntoa = s_fsntoa + swntxy(i,j) 
           s_fsntoac = s_fsntoac + swntcxy(i,j) 
           s_fsns = s_fsns + swnsxy(i,j) 
           s_fsnsc = s_fsnsc + swnscxy(i,j) 
           s_fsds = s_fsds + swdsxy(i,j) 
           s_flds = s_flds + lwdsxy(i,j) 
           s_solin = s_solin + solinxy(i,j) 
        end do
     end do
  end if
!----------------------------------------------------------------
!  Write the radiation-restart file:
  
  
  if(mod(nstep,nstat*(1+nrestart_skip)).eq.0.or.nstep.eq.nstop.or.nelapse.eq.0) then
  
     call write_rad() ! write radiation restart file
  
  endif

!-------------------------------------------------------
! Update the temperature field:	

999 continue

  do k=1,nzm
     do j=1,ny
        do i=1,nx
	   t(i,j,k)=t(i,j,k)+qrad(i,j,k)*dtn
        end do
     end do
  end do

contains
  ! -------------------------------------------------------------------------------
  elemental function compute_tau_l(lwp, re_l)
    real(r4), intent(in) :: lwp, re_l
    real(r4)             :: compute_tau_l
    !
    ! Diagnose optical thickness (nominally at 0.67 microns) from water clouds
    !
    ! This version comes from radcswmx.f90 in the CAM3 radiation package, which 
    !   expects L/IWP in g/m2 and particle sizes in microns
  
    real(r4), parameter :: abarl = 2.817e-02, bbarl = 1.305, & 
                       abari = 3.448e-03, bbari = 2.431

    compute_tau_l = 0.
    if(re_l > 0.) compute_tau_l= (abarl + bbarl/re_l) * lwp  
  
  end function compute_tau_l

  ! -------------------------------------------------------------------------------
  elemental function compute_tau_i(iwp, re_i)
    real(r4), intent(in) :: iwp, re_i
    real(r4)             :: compute_tau_i
    !
    ! Diagnose optical thickness (nominally at 0.67 microns) from ice clouds
    !
    ! This version comes from radcswmx.f90 in the CAM3 radiation package, which 
    !   expects L/IWP in g/m2 and particle sizes in microns
  
    real(r4), parameter :: abarl = 2.817e-02, bbarl = 1.305, & 
                       abari = 3.448e-03, bbari = 2.431

    compute_tau_i = 0.
    if(re_i > 0.) compute_tau_i = (abari + bbari/re_i) * iwp 
  
  end function compute_tau_i
  ! -------------------------------------------------------------------------------
end subroutine rad_full


real function perpetual_factor(day, lat, lon)
use shr_kind_mod, only: r4 => shr_kind_r4
use grid, ONLY: dt, nrad
use ppgrid, only: eccf, pie
implicit none

!  estimate the factor to multiply the solar constant
!  so that the sun hanging perpetually right above
!  the head (zenith angle=0) would produce the same
!  total input the TOA as the sun subgect to diurnal cycle.
!  coded by Marat Khairoutdinov, 2004
!
! Input arguments
!
real day             ! Calendar day, without fraction
real lat                ! Current centered latitude (degrees)
real lon          ! Centered longitude (degrees)

! Local:
real(r4) :: tmp
real(r4) :: ttime
real(r4) :: coszrs(1) 
real(r4) :: clat(1), clon(1)
real :: dttime 

ttime = day
dttime = dt*float(nrad)/86400.
tmp = 0.
pie = acos(-1.)

do while (ttime.lt.day+1.)
   clat = pie*lat/180.
   clon = pie*lon/180.
  call zenith(ttime, clat, clon, coszrs, 1)
  tmp = tmp + min(dttime,day+1-ttime)*max(0._r4,eccf*coszrs(1))
  ttime = ttime+dttime
end do

perpetual_factor = tmp

end function perpetual_factor


subroutine equinox_cos_zenith_angle(day, lat, mu)
  use shr_kind_mod, only: SHR_KIND_R4, SHR_KIND_IN
  implicit none
  real(SHR_KIND_R4), intent(in) :: day, lat
  real(SHR_KIND_R4), intent(out) :: mu
  real(SHR_KIND_R4) :: pi, time_of_day
  pi  = atan(1.0)*4
  time_of_day = mod(day, 1.0)
  mu = -cos(2*pi*time_of_day) * cos(pi * lat / 180.)
end subroutine equinox_cos_zenith_angle
