module microphysics

! main interface to Thompson microphysics.
! original implementation by Peter Blossey, UW,
! based on M2005 interface.

use params, only: cp, rgas, rv, lcond, lsub, fac_cond, fac_sub, ggr, &
     doprecip, docloud, LES, salt_factor, tabs_s

use grid, only: nx,ny,nzm,nz, &  !grid dimensions; nzm = nz-1 # of scalar lvls
     dimx1_s,dimx2_s,dimy1_s,dimy2_s, & ! actual scalar-array dimensions in x,y
     z, zi, dz, adz, pres, dtn, dt, &
     dostatis, masterproc, doSAMconditionals, dosatupdnconditionals, &
     case, caseid, rank, nrestart, dompi, nsubdomains, &
     compute_reffc, compute_reffi, doDge_SnowAndIce, &
     save2Dbin, save2Davg, nsave2D

use vars, only: rho, w, t, tabs, qv, qcl, qci, qpl, qpi, &
     gamaz, precsfc, precflux, qpfall, tlat, prec_xy, &
     tlatqi, tabs0, qv0, fluxbq, fluxtq, dtfactor, &
     sstxy, t00, rhow, &
     condavg_mask, ncondavg, condavgname, condavglongname, &
     nstep,nprint, nstatis, icycle, total_water_prec, &
     nrainy, ncmn, nrmn

use hbuffer, only: hbuf_put     

use micro_params

use module_mp_thompson, only: thompson_init, mp_thompson, calc_effectRadAndDge, calc_refl10cm
use module_mp_thompiso, only: thompiso_init, mp_thompiso, &
     calc_effectRadAndDge_iso => calc_effectRadAndDge,  calc_refl10cm_iso => calc_refl10cm

use module_mp_water_isotopologues, only: niso_max, water_isotopologue_setparm, initprofile_isotope, &
     isotopic_surface_flux

implicit none

logical :: isallocatedMICRO = .false.

integer :: nmicro_fields ! total number of prognostic water vars
integer :: niso = 0 ! number of water isotopologues in addition to standard H_2^16O

real, allocatable, dimension(:,:,:,:) :: micro_field  ! holds mphys quantities

! indices of water quantities in micro_field, e.g. qv = micro_field(:,:,:,iqv)
integer, dimension(niso_max) :: iso_index
character(LEN=3), dimension(niso_max) :: isoname

integer :: nmass_fields, nnumber_fields

integer :: iqv, iqcl, iqci, iqr, iqs, iqg
integer :: iqv_iso(niso_max), iqcl_iso(niso_max), iqci_iso(niso_max), &
     iqr_iso(niso_max), iqs_iso(niso_max), iqg_iso(niso_max)
integer :: inci, inr, incl, inwfa, inifa

integer :: index_water_vapor ! separate water vapor index used by SAM

real, allocatable, dimension(:) :: lfac
integer, allocatable, dimension(:) :: flag_wmass, flag_precip, flag_number
integer, allocatable, dimension(:) :: flag_micro3Dout
logical, allocatable, dimension(:) :: is_water_vapor

! number of fields output from micro_write_fields2D, micro_write_fields3D
integer :: nfields2D_micro=0 
integer :: nfields3D_micro=0 

integer, parameter :: index_cloud_ice = -1 ! historical variable (don't change)

real, allocatable, dimension(:,:,:) :: fluxbmk, fluxtmk !surface/top fluxes
real, dimension(nx,ny,niso_max) :: iso_fluxbq, iso_fluxtq ! isotope fluxes at surface/top

! fields for radiation.
logical :: dosnow_radiatively_active = .true. 
logical :: DORRTM_CLOUD_OPTICS_FROM_EFFRAD_LEGACYOPTION =  .false.
real, allocatable, dimension(:,:,:) :: reffc, reffi, reffs, reffr
real, allocatable, dimension(:,:,:) :: &
     CloudLiquidMassMixingRatio, CloudIceMassMixingRatio, SnowMassMixingRatio, &
     CloudLiquidGammaExponent, CloudLiquidLambda

real, allocatable, dimension(:,:) :: & ! statistical arrays
     mk0, & ! domain-averaged profiles
     mk_ref, & ! reference profiles, mainly for vapor and isotopologues
     mkwle, & ! resolved vertical flux
     mkwsb, & ! SGS vertical flux
     mksed, & ! sedimentation vertical flux
     mkadv, & ! tendency due to vertical advection
     mkdiff, &! tendency due to vertical diffusion
     mklsadv, & ! tendency due to large-scale vertical advection
     mfrac, & ! fraction of domain with microphysical quantity > 1.e-6
     stend, & ! tendency due to sedimentation
     mtend, & ! tendency due to microphysical processes (other than sedimentation)
     trtau, & ! optical depths of various species
     proc_ratesq, &
     proc_ratesn, &
     iso_precflux, &
     precicesfc, precice_xy, &
     iso_preciceflux

real, allocatable, dimension(:) :: tmtend, preciceflux

real, allocatable, dimension(:,:) :: evap_h2o_xy, prec_h2o_xy
real, allocatable, dimension(:,:,:) :: iso_proc_ratesq, &
     iso_precsfc, iso_prec_xy, iso_precicesfc, iso_precice_xy, &
     iso_evap_xy

real :: sfcpcp, sfcicepcp

! arrays with names/units for microphysical outputs in statistics.
character*4, allocatable, dimension(:) :: mkname
character*80, allocatable, dimension(:) :: mklongname
character*10, allocatable, dimension(:) :: mkunits
real, allocatable, dimension(:) :: mkoutputscale
logical :: douse_reffc = .true., douse_reffi = .true.

! You can also have some additional, diagnostic, arrays, for example, total
! nonprecipitating cloud water, etc:

!bloss: array which holds temperature tendency due to microphysics
real, allocatable, dimension(:,:,:), SAVE :: tmtend3d

! external functions (from sat.f90)
real, external :: qsatw, dtqsatw, qsati

!bloss(2018-03): Use this to turn on a column mass conservation check after
!  each call to mp_thompiso
logical, parameter :: debug = .false.

CONTAINS
!----------------------------------------------------------------------
function micro_scheme_name()
  character(len=32) :: micro_scheme_name
  ! Return the scheme name, normally the same as the directory name with leading "MICRO_" removed  
  micro_scheme_name = "thompson" 
end function   micro_scheme_name
!----------------------------------------------------------------------
!!! Read microphysical options from prm file and allocate variables
!
subroutine micro_setparm()
  implicit none

  integer ierr, ios, ios_missing_namelist, place_holder, counter
  
  integer :: n

   NAMELIST /MICRO_THOMPSON/ & ! DEFAULT VALUES in micro_params.f90
      doicemicro, &         ! use ice species (snow/cloud ice/graupel)
      doaerosols, &         ! use Thompson-Eidhammer scheme with water- and ice-friendly aerosols.
      doisotopes, &         ! option to enable computation of water isotopologues.
      Nc0,            &     ! initial/specified cloud droplet number conc (#/cm3)
      fixed_mu_r, fixed_mu_i, fixed_mu_g, & ! gamma exponents for rain, cloud ice, graupel.
      dofix_mu_c, fixed_mu_c, & ! option to specify pgam (exponent of cloud water's gamma distn)
      BrownEtAl2017_pnr_sml_fix, & !option to use backported fix for rain number from melting snow (default=true)
      doFieldEtAl2007Snow, & ! option to use Field et al (2007) snow moment relationships instead of the default Field et al (2005).  Default = false
      TropicalSnow, &        ! in the 2007 snow parameterization, choose between tropical and mid-latitude snow size distributions.  Default = true
      douse_reffc, &        ! use computed effective radius in radiation computation
      douse_reffi, &        ! use computed effective ice size in radiation computation
      do_output_process_rates, &
      dosnow_radiatively_active, &
      dorrtm_cloud_optics_from_effrad_legacyoption

   !bloss: Create dummy namelist, so that we can figure out error code
   !       for a mising namelist.  This lets us differentiate between
   !       missing namelists and those with an error within the namelist.
   NAMELIST /BNCUIODSBJCB/ place_holder

     !----------------------------------
   !  Read namelist for microphysics options from prm file:
   !------------
   open(55,file='./'//trim(case)//'/prm', status='old',form='formatted') 

   !bloss: get error code for missing namelist (by giving the name for
   !       a namelist that doesn't exist in the prm file).
   read (UNIT=55,NML=BNCUIODSBJCB,IOSTAT=ios_missing_namelist)
   rewind(55) !note that one must rewind before searching for new namelists

   !bloss: read in MICRO_THOMPSON namelist
   read (55,MICRO_THOMPSON,IOSTAT=ios)

   if (ios.ne.0) then
     !namelist error checking
     if(ios.ne.ios_missing_namelist) then
       write(*,*) '****** ERROR: bad specification in MICRO_THOMPSON namelist'
       rewind(55)
       read (55,MICRO_THOMPSON) ! try to get a useful error message
       call task_abort()
     elseif(masterproc) then
       write(*,*) '****************************************************'
       write(*,*) '****** No MICRO_THOMPSON namelist in prm file *********'
       write(*,*) '****************************************************'
     end if
   end if
   close(55)

   ! write namelist values out to file for documentation
   if(masterproc) then
      open(unit=55,file='./'//trim(case)//'/'//trim(case)//'_'//trim(caseid)//'.nml', form='formatted', position='append')    
      write (unit=55,nml=MICRO_THOMPSON,IOSTAT=ios)
      write(55,*) ' '
      close(unit=55)
   end if

   ! If isotopologues are enabled, figure out which ones.
   if(doisotopes) call water_isotopologue_setparm(niso,isoname,iso_index)

   if(doisotopes) then
     ! output evaporation rate for standard and heavy isotopologues
     nfields2D_micro = nfields2D_micro + 1 + niso

     ! output surface precip in kg/m2/s if enabled
     if(doprecip) nfields2D_micro = nfields2D_micro + 1 + niso
   end if

   ! set up process rate outputs
   if(do_output_process_rates) then
     if(doicemicro) then
       nproc_rates_mass = nproc_rates_mass_thompson_cold
       nproc_rates_number = nproc_rates_number_thompson_cold
     else
       nproc_rates_mass = nproc_rates_mass_thompson_warm
       nproc_rates_number = nproc_rates_number_thompson_warm
     end if
   end if

   !Always: water vapor, cloud liquid mass, rain mass, rain number
   nmass_fields = 3
   nnumber_fields = 1

   if(doicemicro) then
     ! add cloud ice, cloud ice number, snow mass, graupel mass
     nmass_fields = nmass_fields + 3
     nnumber_fields = nnumber_fields + 1
   end if

   if(doaerosols) then
     ! add variables for water-friendly aerosol, ice-friendly aerosol 
     !   and cloud droplet number
     nnumber_fields = nnumber_fields + 3
   end if

   if(doisotopes) then
     ! duplicate mass fields for each isotopologue
     nmass_fields = nmass_fields*(1 + niso)
   end if

   nmicro_fields = nmass_fields + nnumber_fields

  ! specify index of various quantities in micro_field array
  !  *** note that not all of these may be used if(.not.doicemicro) ***

   ! By convention, put all of the mass variables first, including isotopes,
   !   followed by the number concentrations
   iqv = 1   ! water vapor mass mixing ratio [kg H2O / kg dry air]
   iqcl = 2  ! cloud liquid mass mixing ratio [kg H2O / kg dry air]
   iqr = 3  ! rain mass mixing ratio [kg H2O / kg dry air]

   counter = 3

   if(doicemicro) then
     iqci = counter + 1
     iqs = counter + 2
     iqg = counter + 3
     counter = counter + 3
   end if
     
   if(doisotopes) then
     do n = 1,niso
       iqv_iso(n) = counter + 1
       iqcl_iso(n) = counter + 2
       iqr_iso(n) = counter + 3
       counter = counter + 3

       if(doicemicro) then
         iqci_iso(n) = counter + 1
         iqs_iso(n) = counter + 2
         iqg_iso(n) = counter + 3
         counter = counter + 3
       end if
     end do
   end if

   ! now create indices for the number concentrations
   inr = counter + 1 ! always have rain number
   counter = counter + 1

   if(doicemicro) then
     inci = counter + 1 ! cloud ice number concentration 
     counter = counter + 1
   end if

   if(doaerosols) then
     inwfa = counter + 1 ! water-friendly aerosol number concentration 
     incl = counter + 2 ! cloud liquid droplet number concentration 
     inifa = counter + 3 ! ice-friendly aerosol number concentration
     counter = counter + 3
   end if

   ! stop if counter does not equal nmicro_fields.
   if(counter.ne.nmicro_fields) then
     if(masterproc) write(*,*) 'Stopping: Mismatch between nmicro_fields and number of variables in use in Thompson Microphysics'
     call task_abort()
  end if

  ! stop if icemicro is specified without precip -- we don't support this right now.
  if((doicemicro).and.(.not.doprecip)) then
     if(masterproc) write(*,*) 'Stopping: Thompson Microphysics does not support both doice and .not.doprecip'
     call task_abort()
  end if

  ! stop if niso==0 and doisotopes==.true.
  if(doisotopes.AND.(niso.eq.0)) then
     if(masterproc) write(*,*) 'Stopping: In Thompson microphysics, doisotopes==.true. and niso==0'
     call task_abort()
  end if

  ! stop if doaerosols==true
  if(doaerosols) then
     if(masterproc) write(*,*) 'Stopping: In Thompson microphysics, aerosols not yet supported'
     call task_abort()
  end if

  index_water_vapor = iqv ! set SAM water vapor flag

  if(.not.isallocatedMICRO) then
     ! allocate microphysical variables
     allocate(micro_field(dimx1_s:dimx2_s,dimy1_s:dimy2_s,nzm,nmicro_fields), &
          fluxbmk(nx,ny,nmicro_fields), fluxtmk(nx,ny,nmicro_fields), &
          reffc(nx,ny,nzm), reffi(nx,ny,nzm), reffs(nx,ny,nzm), &
          CloudLiquidMassMixingRatio(nx,ny,nzm), &
          CloudIceMassMixingRatio(nx,ny,nzm), SnowMassMixingRatio(nx,ny,nzm), &
          mkwle(nz,nmicro_fields), mkwsb(nz,nmicro_fields), &
          mkadv(nz,nmicro_fields), mkdiff(nz,nmicro_fields), &
          mklsadv(nz,nmicro_fields), &
          mk0(nzm,nmicro_fields), mk_ref(nzm,nmicro_fields), & 
          stend(nzm,nmicro_fields), mtend(nzm,nmicro_fields), &
          mfrac(nzm,nmicro_fields), trtau(nzm,nmicro_fields), &
          mksed(nzm,nmicro_fields), tmtend(nzm), &
          proc_ratesq(nzm,nproc_rates_mass_thompson_cold), &
          proc_ratesn(nzm,nproc_rates_number_thompson_cold), &
          precicesfc(nx,ny), precice_xy(nx,ny), preciceflux(nz), &
          evap_h2o_xy(nx,ny), prec_h2o_xy(nx,ny), &
          tmtend3d(nx,ny,nzm), flag_micro3Dout(nmicro_fields), &
          flag_wmass(nmicro_fields), flag_precip(nmicro_fields), &
          flag_number(nmicro_fields), lfac(nmicro_fields), &
          mkname(nmicro_fields), mklongname(nmicro_fields), &
          mkunits(nmicro_fields), mkoutputscale(nmicro_fields), &
          is_water_vapor(nmicro_fields), &
          STAT=ierr)
     if(ierr.ne.0) then
        write(*,*) 'Failed to allocate microphysical arrays on proc ', rank
        call task_abort()
     else
        isallocatedMICRO = .true.
     end if

     if(doisotopes) then
       ! allocate microphysical variables
       allocate( &
            iso_proc_ratesq(nzm,nproc_rates_mass_thompson_cold,niso), &
            iso_precsfc(nx,ny,niso), iso_prec_xy(nx,ny,niso), &
            iso_precicesfc(nx,ny,niso), iso_precice_xy(nx,ny,niso), &
            iso_preciceflux(nz,niso), iso_precflux(nz,niso), &
            iso_evap_xy(nx,ny,niso), &
            STAT=ierr)
       if(ierr.ne.0) then
         write(*,*) 'Failed to allocate microphysical arrays on proc ', rank
         call task_abort()
       end if
     end if

     isallocatedMICRO = .true.

     ! zero out statistics variables associated with cloud ice sedimentation
     !   in Marat's default SAM microphysics
     tlatqi = 0.

     ! initialize these arrays
     micro_field = 0.
     fluxbmk = 0.
     fluxtmk = 0.
     mkwle = 0.
     mkwsb = 0.
     mkadv = 0.
     mkdiff = 0.
     mklsadv = 0.
     mk0 = 0.

     ! initialize flag arrays to zero
     flag_wmass = 0
     flag_number = 0
     flag_precip = 0
     flag_micro3Dout = 0

     ! initialize fields useful for radiation
     reffc = 25.
     reffi = 25.
     reffs = 100.

     CloudLiquidMassMixingRatio = 0.
     CloudIceMassMixingRatio = 0.
     SnowMassMixingRatio = 0.

  end if

  ! by default, effective radii in microphysics will be used in radiation,
  !   though this can be changed in the namelist using douse_reff*
  compute_reffc = douse_reffc
  compute_reffi = douse_reffi

  if(dorrtm_cloud_optics_from_effrad_LegacyOption) then
    !bloss(2016-02-09): If using legacy radiative treatment, make sure snow is
    !   not radiatively active.
    dosnow_radiatively_active = .false.
    if(masterproc) write(*,*) '*** Snow is not radiatively active when using legacy radiation ***'
  end if

end subroutine micro_setparm

!----------------------------------------------------------------------
!!! Initialize microphysics:
!
! this one is guaranteed to be called by SAM at the 
!   beginning of each run, initial or restart:
subroutine micro_init()

  implicit none
  
  real, dimension(nzm) :: qc0, qi0, nc1d, dummy, &
       re_qc0, re_qi0, re_qs0, dge_qi0, dge_qs0, &
       qv_iso, qc_iso
  real :: tsurf, qsat_surf

  real, external :: satadj_water
  integer :: k, n

  ! initialize flag arrays

  ! first mass.  NOTE: no isotopes here, because they are negligible in terms of mass
  flag_wmass(:) = 0
  flag_wmass(iqv) = 1 ! vapor
  flag_wmass(iqcl) = 1 ! cloud liquid
  flag_wmass(iqr) = 1 ! rain
  if(doicemicro) then
    flag_wmass(iqci) = 1 ! cloud ice
    flag_wmass(iqs) = 1 ! snow
    flag_wmass(iqg) = 1 ! graupel 
 end if

  ! next precip
  flag_precip(:) = 0
  flag_precip(iqr) = 1 ! rain mass
  flag_precip(inr) = 1 ! rain number
  if(doicemicro) then
    flag_precip(iqci) = 1 ! cloud ice mass
    flag_precip(inci) = 1 ! cloud ice number
    flag_precip(iqs) = 1 ! snow
    flag_precip(iqg) = 1 ! graupel
  end if
  
  if(doisotopes) then
    do n = 1,niso
      flag_precip(iqr_iso(n)) = 1 ! rain mass
      if(doicemicro) then
        flag_precip(iqci_iso(n)) = 1 ! cloud ice mass
        flag_precip(iqs_iso(n)) = 1 ! snow
        flag_precip(iqg_iso(n)) = 1 ! graupel
      end if
    end do
  end if

  ! next number
  flag_number(:) = 0
  flag_number(inr) = 1 ! rain number
  if(doicemicro) then
    flag_number(inci) = 1 ! cloud ice number
  end if
  if(doaerosols) then
    flag_number(incl) = 1 ! cloud liquid number
    flag_number(inwfa) = 1 ! water-friendly aerosol number
    flag_number(inifa) = 1 ! ice-friendly aerosol number
  end if

  ! output all microphysical fields to 3D output files if using more than
  !   just docloud.  Otherwise, rely on basic SAM outputs
  if(docloud.AND.doprecip) then
     flag_micro3Dout = 1
  end if

  !fix is_water_vapor indicator array.  This will be used in upperbound.f90 to
  !  make sure treatment of water vapor is consistent across isotopes at the 
  !  top boundary.
  is_water_vapor(:) = .false.
  is_water_vapor(iqv) = .true.
  if(doisotopes) then
    do n = 1,niso
      is_water_vapor(iqv_iso(n)) = .true.
    end do
  end if

  ! initialize factor for latent heat
  lfac(:) = 1. ! use one as default for number species
  lfac(iqv) = lcond
  lfac(iqcl) = lcond
  lfac(iqr) = lcond
  if(doicemicro) then
     lfac(iqci) = lsub
     lfac(iqs) = lsub
     lfac(iqg) = lsub
  end if

  if(doisotopes) then
    call thompiso_init() ! different routine for isotope-enabled routine
  else  
    call thompson_init() ! call initialization routine within mphys module
  end if

  if(nrestart.eq.0) then

    ! compute initial profiles of liquid water - M.K.
    call satadj_liquid(nzm,tabs0,qv0,qc0,pres*100.)

     ! initialize microphysical quantities
     do k = 1,nzm
        micro_field(:,:,k,iqv) = qv0(k)
        micro_field(:,:,k,iqcl) = qc0(k)
        tabs(:,:,k) = tabs0(k)
      end do

      if(MAXVAL(qc0).gt.0.) then
        dummy(:) = 0.
        nc1d(:) = 1.e6*Nc0/rho(:)
        if(doisotopes) then
          call calc_effectRadAndDge_iso (tabs0, 100.*pres, qv0, qc0, nc1d, dummy, dummy, dummy,   &
               re_qc0, re_qi0, re_qs0, dge_qi0, dge_qs0, 1,nzm)
        else
          call calc_effectRadAndDge (tabs0, 100.*pres, qv0, qc0, nc1d, dummy, dummy, dummy,   &
               re_qc0, re_qi0, re_qs0, dge_qi0, dge_qs0, 1,nzm)
        end if
        do k = 1,nzm
          ! convert to units of microns
          reffc(:,:,k) = MAX( 2.51, MIN( 50., 1.e6*re_qc0(k) ) )
        end do
      end if

      mk_ref(:,:) = 0.
      mk_ref(:,index_water_vapor) = qv0(1:nzm)

    if(doisotopes) then
      ! initial profiles of isotopes.
      do n = 1,niso
        tsurf = tabs_s 
        qsat_surf = qsatw(tsurf,pres(1))*salt_factor ! pres in hPa
        call initprofile_isotope(nzm,tsurf,qsat_surf, &
             tabs0,qv0,qc0,qv_iso,qc_iso,pres,iso_index(n))
        do k = 1,nzm
          micro_field(:,:,k,iqv_iso(n)) = qv_iso(k)
          micro_field(:,:,k,iqcl_iso(n)) = qc_iso(k)
        end do
        mk_ref(:,iqv_iso(n)) = qv_iso(:)
      end do
    end if

     if(doaerosols) then
       ! initialize concentration somehow...
!!$       do k = 1,nzm
!!$         if(qc0(k).gt.0.) then
!!$            micro_field(:,:,k,incl) = 0.5*ccnconst*1.e6
!!$            micro_field(:,:,k,inwfa) = 0.5*ccnconst*1.e6
!!$            micro_field(:,:,k,inifa) = 0.5*ccnconst*1.e6
!!$         end if
!!$       end do
     end if

     if(docloud) call micro_diagnose()   ! leave this here

   else

!!! Re-compute Isotope Initialization to provide a value for mk_ref
     !  NOTE: This will generate differences between restarted and non-restarted runs....

     ! compute initial profiles of liquid water - M.K.
     call satadj_liquid(nzm,tabs0,qv0,qc0,pres*100.)

     mk_ref(:,:) = 0.
     mk_ref(:,index_water_vapor) = qv0(1:nzm)

     if(doisotopes) then
       ! initial profiles of isotopes.
       do n = 1,niso
         tsurf = tabs_s 
         qsat_surf = qsatw(tsurf,pres(1))*salt_factor ! pres in hPa
         call initprofile_isotope(nzm,tsurf,qsat_surf, &
              tabs0,qv0,qc0,qv_iso,qc_iso,pres,iso_index(n))
         mk_ref(:,iqv_iso(n)) = qv_iso(:)
       end do
     end if

   end if

  ! set up names, units and scales for these microphysical quantities
  mkname(iqv) = 'QV'
  mklongname(iqv) = 'WATER VAPOR MASS MIXING RATIO'
  mkunits(iqv) = 'g/kg'
  mkoutputscale(iqv) = 1.e3

  mkname(iqcl) = 'QC'
  mklongname(iqcl) = 'CLOUD WATER MASS MIXING RATIO'
  mkunits(iqcl) = 'g/kg'
  mkoutputscale(iqcl) = 1.e3

  mkname(iqr) = 'QR'
  mklongname(iqr) = 'RAIN MASS MIXING RATIO'
  mkunits(iqr) = 'g/kg'
  mkoutputscale(iqr) = 1.e3

  mkname(inr) = 'NR'
  mklongname(inr) = 'RAIN NUMBER MIXING RATIO'
  mkunits(inr) = '#/kg' ! change from M2005: !!output as mass mixing ratios!!
  mkoutputscale(inr) = 1.

  if(doicemicro) then
    mkname(iqci) = 'QI'
    mklongname(iqci) = 'CLOUD ICE MASS MIXING RATIO'
    mkunits(iqci) = 'g/kg'
    mkoutputscale(iqci) = 1.e3

    mkname(inci) = 'NI'
    mklongname(inci) = 'CLOUD ICE NUMBER MIXING RATIO'
    mkunits(inci) = '#/kg'
    mkoutputscale(inci) = 1.

    mkname(iqs) = 'QS'
    mklongname(iqs) = 'SNOW MASS MIXING RATIO'
    mkunits(iqs) = 'g/kg'
    mkoutputscale(iqs) = 1.e3

    mkname(iqg) = 'QG'
    mklongname(iqg) = 'GRAUPEL MASS MIXING RATIO'
    mkunits(iqg) = 'g/kg'
    mkoutputscale(iqg) = 1.e3
  end if

  if(doisotopes) then

    do n = 1,niso
      mkname(iqv_iso(n)) = TRIM(isoname(n)) // 'V'
      mklongname(iqv_iso(n)) = TRIM(isoname(n)) // ' WATER VAPOR MASS MIXING RATIO, ASSUMING VSMOW==1'
      mkunits(iqv_iso(n)) = 'g/kg'
      mkoutputscale(iqv_iso(n)) = 1.e3

      mkname(iqcl_iso(n)) = TRIM(isoname(n)) // 'C'
      mklongname(iqcl_iso(n)) = TRIM(isoname(n)) // ' CLOUD LIQUID MASS MIXING RATIO, ASSUMING VSMOW==1'
      mkunits(iqcl_iso(n)) = 'g/kg'
      mkoutputscale(iqcl_iso(n)) = 1.e3

      mkname(iqr_iso(n)) = TRIM(isoname(n)) // 'R'
      mklongname(iqr_iso(n)) = TRIM(isoname(n)) // ' RAIN MASS MIXING RATIO, ASSUMING VSMOW==1'
      mkunits(iqr_iso(n)) = 'g/kg'
      mkoutputscale(iqr_iso(n)) = 1.e3

      if(doicemicro) then
        mkname(iqci_iso(n)) = TRIM(isoname(n)) // 'I'
        mklongname(iqci_iso(n)) = TRIM(isoname(n)) // ' CLOUD ICE MASS MIXING RATIO, ASSUMING VSMOW==1'
        mkunits(iqci_iso(n)) = 'g/kg'
        mkoutputscale(iqci_iso(n)) = 1.e3

        mkname(iqs_iso(n)) = TRIM(isoname(n)) // 'S'
        mklongname(iqs_iso(n)) = TRIM(isoname(n)) // ' SNOW MIXING RATIO, ASSUMING VSMOW==1'
        mkunits(iqs_iso(n)) = 'g/kg'
        mkoutputscale(iqs_iso(n)) = 1.e3

        mkname(iqg_iso(n)) = TRIM(isoname(n)) // 'G'
        mklongname(iqg_iso(n)) = TRIM(isoname(n)) // ' GRAUPEL MASS MIXING RATIO, ASSUMING VSMOW==1'
        mkunits(iqg_iso(n)) = 'g/kg'
        mkoutputscale(iqg_iso(n)) = 1.e3
      end if
    end do
  end if

  if(doaerosols) then
    mkname(incl) = 'NC'
    mklongname(incl) = 'CLOUD WATER NUMBER MIXING RATIO'
    mkunits(incl) = '#/kg'
    mkoutputscale(incl) = 1.

    mkname(inwfa) = 'NWFA'
    mklongname(inwfa) = 'WATER-FRIENDLY AEROSOL NUMBER MIXING RATIO'
    mkunits(inwfa) = '#/kg'
    mkoutputscale(inwfa) = 1.

    mkname(inifa) = 'NIFA'
    mklongname(inifa) = 'ICE-FRIENDLY AEROSOL NUMBER MIXING RATIO'
    mkunits(inifa) = '#/kg'
    mkoutputscale(inifa) = 1.
  end if

end subroutine micro_init

!----------------------------------------------------------------------
!!! fill-in surface and top boundary fluxes:
!
! Obviously, for liquid/ice water variables those fluxes are zero. They are not zero
! only for water vapor variable and, possibly, for CCN and IN if you have those.

subroutine micro_flux()

implicit none

integer :: n, i, j
real, dimension(nx,ny) :: iso_qv, iso_flux, ts

fluxbmk(:,:,:) = 0. ! initialize all fluxes at surface to zero
fluxtmk(:,:,:) = 0. ! initialize all fluxes at top of domain to zero

fluxbmk(:,:,index_water_vapor) = fluxbq(:,:) ! surface qv (latent heat) flux
fluxtmk(:,:,index_water_vapor) = fluxtq(:,:) ! top of domain qv flux

! accumulate surface fluxes for 2D output (this duplicates lhf_xy, but is useful 
!  because it is exactly analogous to the iso_evap_xy ouputs).
evap_h2o_xy(:,:) = evap_h2o_xy(:,:) + fluxbmk(:,:,index_water_vapor)*dtfactor

do n = 1,niso
  ! handle surface fluxes for water vapor isotopologues
  if(LES) then
    ! single surface flux for whole domain
    iso_qv(1,1) = mk0(1,iqv_iso(n))
    ts(1,1) = sstxy(1,1) + t00
    call isotopic_surface_flux(1,1,ts,iso_qv,iso_flux,iso_index(n))
    fluxbmk(:,:,iqv_iso(n)) = iso_flux(1,1) ! spread that flux across domain.

  else
    ! surface fluxes computed locally
    iso_qv(:,:) = micro_field(1:nx,1:ny,1,iqv_iso(n))
    ts(:,:) = sstxy(1:nx,1:ny) + t00
    call isotopic_surface_flux(nx,ny,ts,iso_qv,iso_flux,iso_index(n))
    fluxbmk(:,:,iqv_iso(n)) = iso_flux(:,:)
  end if

  ! accumulate surface fluxes for 2D output
  iso_evap_xy(:,:,n) = iso_evap_xy(:,:,n) + fluxbmk(:,:,iqv_iso(n))*dtfactor
end do

if(mk0(nzm,iqv).gt.1.e-8) then
   ! if there is water vapor at top of model, give incoming vapor
   !  flux same fractionation as mean at top model level.
   do n = 1,niso
      fluxtmk(:,:,iqv_iso(n)) = fluxtq(:,:)*mk0(nzm,iqv_iso(n))/mk0(nzm,iqv)
   end do
end if

end subroutine micro_flux

!----------------------------------------------------------------------
!!! compute local microphysics processes (beyond advection and SGS diffusion):
!
!  This is the place where the condensation/sublimation, accretion, coagulation, freezing,
!  melting, etc., that is  all the microphysics processes except for the spatial transport happen.

! IMPORTANT: You need to use the thermodynamic constants like specific heat, or
! specific heat of condensation, gas constant, etc, the same as in file params.f90
! Also, you should assume that the conservative thermodynamic variable during these
! proceses is the liquid/ice water static energy: t = tabs + gz - Lc (qc+qr) - Ls (qi+qs+qg) 
! It should not be changed during all of your point microphysical processes!

subroutine micro_proc()

implicit none

real, dimension(nzm) :: &
     qv1d,qcl1d,qr1d,qci1d,qs1d,qg1d, &
     nci1d,nr1d,ni1d,t1d,p1d,rho1d,dzq, &
     re_qcl1d, re_qci1d, re_qs1d, &
     dge_qci1d, dge_qs1d, &
     w1d, wsub1d, ncl1d

real :: ppt_rain, ppt_snow, ppt_grau, ppt_clice, &
     ppticencv, pptncv, tmpr, tmpc, tmpi, tmps
real :: proc_ratesq1d(nzm,nproc_rates_mass_thompson_cold)
real :: proc_ratesn1d(nzm,nproc_rates_number_thompson_cold)

real, dimension(nzm,6) :: stendq1d, mtendq1d
real, dimension(nzm,2) :: stendn1d, mtendn1d

real, dimension(nzm,niso) :: &
     iso_qv1d,iso_qcl1d,iso_qr1d, iso_qci1d,iso_qs1d,iso_qg1d
real, dimension(niso) :: &
     iso_ppt_rain, iso_ppt_snow, iso_ppt_grau, iso_ppt_clice, &
     iso_ppticencv, iso_pptncv, iso_total_water_prec

real :: iso_proc_ratesq1d(nzm,nproc_rates_mass_thompson_cold,niso)
real :: iso_stendq1d(nzm,6,niso)
real :: iso_mtendq1d(nzm,6,niso)

integer :: i1, i2, j1, j2, i, j, k, m, n, nn

real(8) :: tmp_total, tmptot, twp_before, twp_after

call t_startf ('micro_proc')

if(mod(nstep-1,nstatis).eq.0.and.icycle.eq.1) then
  precsfc(:,:)=0.
  precflux(:) = 0.
  preciceflux(:) = 0.

   if(doisotopes) then
     ! 2d arrays.
     iso_precsfc(:,:,:) = 0.
     iso_precicesfc(:,:,:) = 0.
     ! vertical profiles
     iso_precflux(:,:) = 0.
     iso_preciceflux(:,:) = 0.
   end if
end if

if(dostatis) then ! initialize arrays for statistics
   mfrac(:,:) = 0.
   mtend(:,:) = 0.
   stend(:,:) = 0.
   trtau(:,:) = 0.
   qpfall(:)=0.
   tlat(:) = 0.
   tmtend3d(:,:,:) = 0.
   proc_ratesq(:,:) = 0.
   proc_ratesn(:,:) = 0.
   if(doisotopes) then
     iso_proc_ratesq(:,:,:) = 0.
   end if
end if

stend(:,:) = 0.
mksed(:,:) = 0.

!!$if(doprecip) total_water_prec = total_water_prec + total_water()
 
do j = 1,ny
   do i = 1,nx

      ! zero out mixing ratios of microphysical species
      qv1d(:) = 0.
      qcl1d(:) = 0.
      qr1d(:) = 0.
      nr1d(:) = 0.
      qci1d(:) = 0.
      nci1d(:) = 0.
      qs1d(:) = 0.
      qg1d(:) = 0.

      ! get microphysical quantities in this grid column
      qv1d(:) = micro_field(i,j,:,iqv) 
      qcl1d(:) = micro_field(i,j,:,iqcl)
      qr1d(:) = micro_field(i,j,:,iqr)
      nr1d(:) = micro_field(i,j,:,inr)

      if(doicemicro) then
        qci1d(:) = micro_field(i,j,:,iqci)
        nci1d(:) = micro_field(i,j,:,inci)
        qs1d(:) = micro_field(i,j,:,iqs)
        qg1d(:) = micro_field(i,j,:,iqg)
      end if

      ! get absolute temperature in this column
      !          this is Tcl = T - (L/Cp)*qcl (the cloud liquid water temperature)
      t1d(:) = t(i,j,:)  &           ! liquid water-ice static energy over Cp
           - gamaz(:) &                                   ! potential energy
           + fac_cond * (qcl1d(:) + qr1d(:)) &    ! bloss/qt: liquid latent energy due to rain/cloud liquid
           + fac_sub  * (qci1d(:) + qs1d(:) + qg1d(:)) ! ice latent energy

      dzq(:) = adz(:)*dz
!      tmpw = 0.5*(w(i,j,1:nzm) + w(i,j,2:nz))  ! MK: changed for stretched grids 
      w1d(:) = ((zi(2:nz)-z(1:nzm))*w(i,j,1:nzm)+ &
           (z(1:nzm)-zi(1:nzm))*w(i,j,2:nz))/(zi(2:nz)-zi(1:nzm))
      wsub1d(:) = 0.

      p1d(:) = 100.*pres(1:nzm)
      rho1d(:) = rho(:)

      mtendq1d(:,:) = 0.
      mtendn1d(:,:) = 0.
      stendq1d(:,:) = 0.
      stendn1d(:,:) = 0.

      ppt_rain = 0.
      ppt_snow = 0.
      ppt_grau = 0.
      ppt_clice = 0.

      proc_ratesq1d(:,:) = 0.
      proc_ratesn1d(:,:) = 0.

      if(.NOT.doisotopes) then

        if(doaerosols) then

          !THOMPSON W/AEROSOLS FROM WRF 3.6 (THOMPSON-EIDHAMMER, 2014)
          if(masterproc) then
            write(*,*) 'New Thompson-Eidmann scheme not included yet'
          end if
          STOP ' in THOMPSON microphysics'

        else
          
          !THOMPSON, NO AEROSOLS FROM WRF 3.5.1 (THOMPSON ET AL, 2008)
          call mp_thompson (qv1d, qcl1d, qci1d, qr1d, qs1d, qg1d, nci1d, &
               nr1d, t1d, p1d, rho1d, dzq, &
               ppt_rain, ppt_snow, ppt_grau, ppt_clice, &
               mtendq1d, mtendn1d, stendq1d, stendn1d, &
               proc_ratesq1d, proc_ratesn1d, &
               1,nzm,dtn,1,1,doprecip, &
               do_output_process_rates)

        end if

      else

        !ISOTOPE-ENABLED VERSION OF NO-AEROSOL THOMPSON FROM WRF 3.5.1 (THOMPSON ET AL, 2008)

        do nn = 1,niso
          ! get isotope mass mixing ratios in this grid column
          iso_qv1d(:,nn) = micro_field(i,j,:,iqv_iso(nn)) 
          iso_qcl1d(:,nn) = micro_field(i,j,:,iqcl_iso(nn))
          iso_qr1d(:,nn) = micro_field(i,j,:,iqr_iso(nn))

          if(doicemicro) then
            iso_qci1d(:,nn) = micro_field(i,j,:,iqci_iso(nn))
            iso_qs1d(:,nn) = micro_field(i,j,:,iqs_iso(nn))
            iso_qg1d(:,nn) = micro_field(i,j,:,iqg_iso(nn))
          end if
        end do

        iso_mtendq1d(:,:,:) = 0.
        iso_stendq1d(:,:,:) = 0.
        iso_proc_ratesq1d(:,:,:) = 0.

        ! isotopologue surface precip
        iso_ppt_snow(:) = 0.
        iso_ppt_grau(:) = 0.
        iso_ppt_clice(:) = 0.
        iso_ppt_rain(:) = 0.

        call mp_thompiso (qv1d, qcl1d, qci1d, qr1d, qs1d, qg1d, &
             nci1d, nr1d, t1d, p1d, rho1d, dzq, &
             ppt_rain, ppt_snow, ppt_grau, ppt_clice, &
             iso_qv1d, iso_qcl1d, iso_qci1d, iso_qr1d, iso_qs1d, iso_qg1d, &
             iso_ppt_rain, iso_ppt_snow, iso_ppt_grau, iso_ppt_clice, &
             mtendq1d, mtendn1d, stendq1d, stendn1d, &
             proc_ratesq1d, proc_ratesn1d, &
             iso_mtendq1d, iso_stendq1d, iso_proc_ratesq1d, &
             1,nzm,dtn,1,1,niso,iso_index,doprecip, &
             do_output_process_rates)

        if(debug.AND.doicemicro) then
          ! check for column mass balance in microphysics: standard water
          twp_before = 0.
          twp_after = 0.
          do k = 1,nzm
            twp_before = twp_before + dz*adz(k)*rho(k)*SUM(micro_field(i,j,k,1:6))
            twp_after = twp_after + dz*adz(k)*rho(k)*(qv1d(k)+qcl1d(k)+qr1d(k)+qci1d(k)+qs1d(k)+qg1d(k))
          end do
          if(ABS(twp_before-twp_after-ppt_rain).gt.1.e-8*twp_before) then
            write(*,895) i,j,nstep
            write(*,896) twp_before, twp_after, ppt_rain, twp_after+ppt_rain-twp_before, &
                 (twp_after+ppt_rain-twp_before)/twp_before
895         format('H2O column mass error in mp_thompiso: i,j,nstep = ',i4,i4,i10)
896         format('TWP before/TWP after/pcp = ',3f10.6,' deltaTWP,deltaTWP/TWP before = ',2e12.4)
          end if

          do nn = 1,niso
            ! check for column mass balance in microphysics: isotopologues
            twp_before = 0.
            twp_after = 0.
            do k = 1,nzm
              twp_before = twp_before + dz*adz(k)*rho(k)*SUM(micro_field(i,j,k,6*nn+1:6*nn+6))
              twp_after = twp_after &
                   + dz*adz(k)*rho(k)*(iso_qv1d(k,nn)+iso_qcl1d(k,nn) &
                                      +iso_qr1d(k,nn)+iso_qci1d(k,nn)+iso_qs1d(k,nn)+iso_qg1d(k,nn))
            end do
            if(ABS(twp_before-twp_after-iso_ppt_rain(nn)).gt.1.e-8*twp_before) then
              write(*,897) i,j,nstep,nn
              write(*,896) twp_before, twp_after, iso_ppt_rain(nn), twp_after+iso_ppt_rain(nn)-twp_before, &
                   (twp_after+iso_ppt_rain(nn)-twp_before)/twp_before
897           format('Iso column mass error in mp_thompiso: i,j,nstep,niso = ',i4,i4,i10,i4)
            end if
          end do
        end if

        if(doprecip) then
          iso_ppticencv(:) = iso_ppt_snow(:) + iso_ppt_grau(:) + iso_ppt_clice(:)
          iso_pptncv(:) = iso_ppticencv(:) + iso_ppt_rain(:)

          iso_total_water_prec(:) = iso_total_water_prec(:) + iso_ppt_rain(:)

          ! take care of surface precipitation
          iso_precsfc(i,j,:) = iso_precsfc(i,j,:) + iso_pptncv(:)/dz
          iso_prec_xy(i,j,:) = iso_prec_xy(i,j,:) + iso_pptncv(:)/dz

          iso_precicesfc(i,j,:) = iso_precicesfc(i,j,:) + iso_ppticencv(:)/dz
          iso_precice_xy(i,j,:) = iso_precice_xy(i,j,:) + iso_ppticencv(:)/dz
        else
          ! add rain to cloud
          iso_qcl1d(:,:) = iso_qcl1d(:,:) + iso_qr1d(:,:) ! add rain mass back to cloud water
          iso_qr1d(:,:) = 0. ! zero out rain 

          ! add rain tendencies to cloud
          iso_mtendq1d(:,2,:) = iso_mtendq1d(:,2,:) + iso_mtendq1d(:,4,:) 
          iso_mtendq1d(:,4,:) = 0. ! zero out rain tendencies

          ! NOTE THAT WE TURNED OFF SEDIMENTATION, SO THAT THOSE TENDENCIES SHOULD BE ZERO
        end if

        do nn = 1,niso
          ! isotopologue mass mixing ratios
          micro_field(i,j,:,iqv_iso(nn)) = iso_qv1d(:,nn)
          micro_field(i,j,:,iqcl_iso(nn)) = iso_qcl1d(:,nn)
          micro_field(i,j,:,iqr_iso(nn)) = iso_qr1d(:,nn)
          if(doicemicro) then
            micro_field(i,j,:,iqci_iso(nn)) = iso_qci1d(:,nn)
            micro_field(i,j,:,iqs_iso(nn)) = iso_qs1d(:,nn)
            micro_field(i,j,:,iqg_iso(nn)) = iso_qg1d(:,nn)
          end if

          if(dostatis) then
            ! isotopologue tendencies due to microphysical transformations
            mtend(:,iqv_iso(nn)) = mtend(:,iqv_iso(nn)) + iso_mtendq1d(:,1,nn)
            mtend(:,iqcl_iso(nn)) = mtend(:,iqcl_iso(nn)) + iso_mtendq1d(:,2,nn)
            mtend(:,iqr_iso(nn)) = mtend(:,iqr_iso(nn)) + iso_mtendq1d(:,4,nn) ! rain index==4
            if(doicemicro) then
              mtend(:,iqci_iso(nn)) = mtend(:,iqci_iso(nn)) + iso_mtendq1d(:,3,nn) ! cl ice index==3
              mtend(:,iqs_iso(nn)) = mtend(:,iqs_iso(nn)) + iso_mtendq1d(:,5,nn)
              mtend(:,iqg_iso(nn)) = mtend(:,iqg_iso(nn)) + iso_mtendq1d(:,6,nn)
            end if

            iso_proc_ratesq(:,:,:) = iso_proc_ratesq(:,:,:) + iso_proc_ratesq1d(:,:,:) 
          end if

        end do

      end if ! if(.NOT.doisotopes)

      if(doprecip) then
        ppticencv = ppt_snow + ppt_grau + ppt_clice
        pptncv = ppticencv + ppt_rain

        total_water_prec = total_water_prec + ppt_rain

        ! take care of surface precipitation
        precsfc(i,j) = precsfc(i,j) + pptncv/dz
        prec_xy(i,j) = prec_xy(i,j) + pptncv/dz
        prec_h2o_xy(i,j) = prec_h2o_xy(i,j) + pptncv/dz ! duplicate so that isotopic ratio can be easily computed

        precicesfc(i,j) = precicesfc(i,j) + ppticencv/dz
        precice_xy(i,j) = precice_xy(i,j) + ppticencv/dz
      else
        ! add rain to cloud
        qcl1d(:) = qcl1d(:) + qr1d(:) ! add rain mass back to cloud water
        qr1d(:) = 0. ! zero out rain 
        nr1d(:) = 0. ! zero out rain number

        ! add rain tendencies to cloud
        mtendq1d(:,2) = mtendq1d(:,2) + mtendq1d(:,4) 
        mtendq1d(:,4) = 0. ! zero out rain tendencies
        mtendn1d(:,2) = 0. ! zero out rain number tendencies

        ! NOTE THAT WE TURNED OFF SEDIMENTATION, SO THAT THOSE TENDENCIES SHOULD BE ZERO
      end if

      ! save updated mass mixing ratios
      micro_field(i,j,:,iqv) = qv1d(:)
      micro_field(i,j,:,iqcl) = qcl1d(:)
      micro_field(i,j,:,iqr) = qr1d(:)
      micro_field(i,j,:,inr) = nr1d(:)
      CloudLiquidMassMixingRatio(i,j,:) = qcl1d(:)
      if(doicemicro) then
        micro_field(i,j,:,iqci) = qci1d(:)
        micro_field(i,j,:,inci) = nci1d(:)
        micro_field(i,j,:,iqs) = qs1d(:)
        micro_field(i,j,:,iqg) = qg1d(:)
        CloudIceMassMixingRatio(i,j,:) = qci1d(:)
        SnowMassMixingRatio(i,j,:) = qs1d(:)
      end if

      ! Update effective radii for radiation.
      if(MAXVAL(qcl1d(:)+qci1d(:)+qs1d(:)).gt.TINY(1.)) then
        ncl1d(:) = 1.e6*Nc0/rho(:)
        if(doisotopes) then
          call calc_effectRadAndDge_iso (t1d, p1d, qv1d, qcl1d, ncl1d, qci1d, nci1d, qs1d, &
               &                re_qcl1d, re_qci1d, re_qs1d, dge_qci1d, dge_qs1d, 1,nzm)
        else
          call calc_effectRadAndDge (t1d, p1d, qv1d, qcl1d, ncl1d, qci1d, nci1d, qs1d, &
               &                re_qcl1d, re_qci1d, re_qs1d, dge_qci1d, dge_qs1d, 1,nzm)
        end if

        !bloss: option for using newly computed generalized effective size for
        !   radiative properties of cloud ice and snow.
        if(doDge_SnowAndIce) then
          ! replace effective radii with generalize effective size (Fu, 1996, JAS)
          re_qci1d(:) = dge_qci1d(:)
          re_qs1d(:) = dge_qs1d(:)
        end if
        !bloss: Note that t1d may not be the exactly correct absolute temperature if thompson
        !   makes different assumptions about latent heats than SAM, but I've tried to
        !   homogenize these and the effective radius is probably quite weakly sensitive
        !   to temperature anyways, so that I'll leave it alone.
        
        !bloss: convert to microns, limit on the low and high end, as in the Thompson routine.
        re_qcl1d(:) = MAX(2.51, MIN( 50.,  1.e6*re_qcl1d(:) ) )
        re_qci1d(:) = 1.e6*re_qci1d(:) 
        re_qs1d(:)  = 1.e6*re_qs1d(:)  
!bloss        re_qci1d(:) = MAX(5.01, MIN(125.,  1.e6*re_qci1d(:) ) )
!bloss        re_qs1d(:)  = MAX(25.0, MIN(999.,  1.e6*re_qs1d(:)  ) )

      else
        ! set effective radii to default values.
        re_qcl1d(:) = 25.
        re_qci1d(:) = 25.
        re_qs1d(:) = 100.
      end if
      reffc(i,j,:) = re_qcl1d(:)
      reffi(i,j,:) = re_qci1d(:)
      reffs(i,j,:) = re_qs1d(:)

      if(dostatis) then
        ! tendencies due to microphysical transformations
        mtend(:,iqv) = mtend(:,iqv) + mtendq1d(:,1)
        mtend(:,iqcl) = mtend(:,iqcl) + mtendq1d(:,2)
        mtend(:,iqr) = mtend(:,iqr) + mtendq1d(:,4) ! rain index==4
        mtend(:,inr) = mtend(:,inr) + mtendn1d(:,2) ! rain number, index==2
        if(doicemicro) then
          mtend(:,iqci) = mtend(:,iqci) + mtendq1d(:,3) ! cl ice index==3
          mtend(:,inci) = mtend(:,inci) + mtendn1d(:,1) ! cl ice number, index==1
          mtend(:,iqs) = mtend(:,iqs) + mtendq1d(:,5)
          mtend(:,iqg) = mtend(:,iqg) + mtendq1d(:,6)
        end if

        proc_ratesq(:,:) = proc_ratesq(:,:) + proc_ratesq1d(:,:) 
        proc_ratesn(:,:) = proc_ratesn(:,:) + proc_ratesn1d(:,:) 

         ! approximate optical depth = 1.5*lwp (g/m2)/effrad
         !  integrated up to level at which output
         tmpc = 0.
         tmpi = 0.
         tmps = 0.

         do k = 1,nzm
           if( (qcl1d(k).gt.1.e-6) .AND. (re_qcl1d(k).gt.1.) ) then
             tmpc = tmpc + 1.5e-3*rho(k)*dz*adz(k)*qcl1d(k)/(TINY(1.)+1.e-6*re_qcl1d(k))
           end if
            trtau(k,iqcl) = trtau(k,iqcl) + tmpc

            if(doicemicro) then
              if( (qci1d(k).gt.1.e-6) .AND. (re_qci1d(k).gt.1.) ) then
                tmpi = tmpi + 1.5e-3*rho(k)*dz*adz(k)*qci1d(k)/(TINY(1.)+1.e-6*re_qci1d(k))
              end if
              if( (qs1d(k).gt.1.e-6) .AND. (re_qs1d(k).gt.1.) ) then
               tmps = tmps + 1.5e-3*rho(k)*dz*adz(k)*qs1d(k)/(TINY(1.)+1.e-6*re_qs1d(k))
             end if

               trtau(k,iqci) = trtau(k,iqci) + tmpi
               trtau(k,iqs) = trtau(k,iqs) + tmps
            end if
         end do

      end if !if(dostatis)

      !=====================================================
      ! update liquid-ice static energy due to precipitation
      t(i,j,:) = t(i,j,:) &
           - dtn*fac_cond*(stendq1d(:,2)+stendq1d(:,4)) ! cloud liquid + rain
      if(doicemicro) then
        t(i,j,:) = t(i,j,:) &
             - dtn*fac_sub*(stendq1d(:,3)+stendq1d(:,5)+stendq1d(:,6)) ! cl ice + snow + graupel
      end if
      !=====================================================

      do n = 2,nmicro_fields
        do k = 1,nzm
          mfrac(k,n) = mfrac(k,n) + MAX(0., SIGN(1., micro_field(i,j,k,n)-1.e-6 ) )
        end do
      end do

      if(doicemicro) then
        do k = 1,nzm
          ! save area fraction for radiatively-active hydometeors: cloud liquid, cloud ice, snow.
          !  Store this in the vapor location
          mfrac(k,iqv) = mfrac(k,iqv) + MAX(0., SIGN(1., &
               micro_field(i,j,k,iqcl)+micro_field(i,j,k,iqci)+micro_field(i,j,k,iqs)-1.e-6 ) )
        end do
      else
        do k = 1,nzm
          ! save area fraction for radiatively-active hydometeors: cloud liquid, cloud ice, snow.
          !  Store this in the vapor location
          mfrac(k,iqv) = mfrac(k,iqv) + MAX(0., SIGN(1., micro_field(i,j,k,iqcl)-1.e-6 ) )
        end do
      end if

      tlat(1:nzm) = tlat(1:nzm) &
           - dtn*fac_cond*(stendq1d(:,2)+stendq1d(:,4)) 
      qpfall(1:nzm) = qpfall(1:nzm) + dtn*stendq1d(:,4)
      tmtend3d(i,j,1:nzm) = &       !bloss: temperature tendency (sensible heating) due to phase changes
           fac_cond*(mtendq1d(:,2)+mtendq1d(:,4)) 
      if(doicemicro) then
        tlat(1:nzm) = tlat(1:nzm) &
             - dtn*fac_sub*(stendq1d(:,3)+stendq1d(:,5)+stendq1d(:,6))
        qpfall(1:nzm) = qpfall(1:nzm) + dtn*(stendq1d(:,5)+stendq1d(:,6))
        tmtend3d(i,j,1:nzm) = tmtend3d(i,j,1:nzm) &
           + fac_sub*(mtendq1d(:,3)+stendq1d(:,5)+stendq1d(:,6))
      end if

      ! accumulate sedimentation tendencies
      !  -- do this every timestep, so that we can accumulate precipitation fluxes
      stend(:,iqv) = stend(:,iqv) + stendq1d(:,1)
      stend(:,iqcl) = stend(:,iqcl) + stendq1d(:,2)
      stend(:,iqr) = stend(:,iqr) + stendq1d(:,4) ! rain index==4
      stend(:,inr) = stend(:,inr) + stendn1d(:,2) ! rain number
      if(doicemicro) then
        stend(:,iqci) = stend(:,iqci) + stendq1d(:,3) ! cl ice index==3
        stend(:,inci) = stend(:,inci) + stendn1d(:,1) ! cl ice number
        stend(:,iqs) = stend(:,iqs) + stendq1d(:,5)
        stend(:,iqg) = stend(:,iqg) + stendq1d(:,6)
      end if

      if(doisotopes) then
        do nn = 1,niso
          ! isotopologue tendencies due to sedimentation transformations
          stend(:,iqv_iso(nn)) = stend(:,iqv_iso(nn)) + iso_stendq1d(:,1,nn)
          stend(:,iqcl_iso(nn)) = stend(:,iqcl_iso(nn)) + iso_stendq1d(:,2,nn)
          stend(:,iqr_iso(nn)) = stend(:,iqr_iso(nn)) + iso_stendq1d(:,4,nn) ! rain index==4
          if(doicemicro) then
            stend(:,iqci_iso(nn)) = stend(:,iqci_iso(nn)) + iso_stendq1d(:,3,nn) ! cl ice index==3
            stend(:,iqs_iso(nn)) = stend(:,iqs_iso(nn)) + iso_stendq1d(:,5,nn)
            stend(:,iqg_iso(nn)) = stend(:,iqg_iso(nn)) + iso_stendq1d(:,6,nn)
          end if
        end do
      end if

    end do ! i = 1,nx
  end do ! j = 1,ny

!!$if(doprecip) total_water_prec = total_water_prec - total_water()

  ! integrate sedimentation tendency for all hydrometeor masses and numbers
  !   (including isotopes) so that we can reconstruct the instantaneous domain-average
  !   precipitation profile
  do n = 1,nmicro_fields
    tmpr = 0.
    do k = 1,nzm
      m = nz-k
      tmpr = tmpr + stend(m,n)*rho(m)*dz*adz(m)
      mksed(m,n) = tmpr
    end do
  end do

  precflux(1:nzm) = precflux(1:nzm) - ( mksed(:,iqcl) + mksed(:,iqr) ) * dtn/dz

  if(doicemicro) then
    precflux(1:nzm) = precflux(1:nzm) - ( mksed(:,iqci) + mksed(:,iqs) + mksed(:,iqg) ) * dtn/dz
    preciceflux(1:nzm) = preciceflux(1:nzm) &
         - ( mksed(:,iqci) + mksed(:,iqs) + mksed(:,iqg) ) * dtn/dz
  end if

  if(doisotopes) then
    do nn = 1,niso
      iso_precflux(1:nzm,nn) = iso_precflux(1:nzm,nn) - ( mksed(:,iqcl_iso(nn)) + mksed(:,iqr_iso(nn)) ) * dtn/dz

      if(doicemicro) then
        iso_precflux(1:nzm,nn) = iso_precflux(1:nzm,nn) &
             - ( mksed(:,iqci_iso(nn)) + mksed(:,iqs_iso(nn)) + mksed(:,iqg_iso(nn)) ) * dtn/dz
        iso_preciceflux(1:nzm,nn) = iso_preciceflux(1:nzm,nn) &
             - ( mksed(:,iqci_iso(nn)) + mksed(:,iqs_iso(nn)) + mksed(:,iqg_iso(nn)) ) * dtn/dz
      end if ! icemicro
    end do ! nn=1,niso
  end if !if(doisotopes)

if (docloud)  call micro_diagnose()   ! leave this line here

call t_stopf ('micro_proc')

end subroutine micro_proc

!----------------------------------------------------------------------
!!! Diagnose arrays nessesary for dynamical core and radiation:
!
!  This is the pace where the microphysics field that SAM actually cares about
!  are diagnosed.

subroutine micro_diagnose()

integer :: n, k
real :: tmp(nzm,nmicro_fields)
real :: qv_iso(nzm), qc_iso(nzm), dummy(nzm), tsurf, qsat_surf

qv(1:nx,1:ny,1:nzm) = micro_field(1:nx,1:ny,1:nzm,iqv) ! water vapor
qcl(1:nx,1:ny,1:nzm) = micro_field(1:nx,1:ny,1:nzm,iqcl) ! cloud liquid water
qpl(1:nx,1:ny,1:nzm) = micro_field(1:nx,1:ny,1:nzm,iqr) ! rain water


if(doicemicro) then
   qci(1:nx,1:ny,1:nzm) = micro_field(1:nx,1:ny,1:nzm,iqci) ! cloud ice
   qpi(1:nx,1:ny,1:nzm) = micro_field(1:nx,1:ny,1:nzm,iqs) & ! snow
           + micro_field(1:nx,1:ny,1:nzm,iqg) ! graupel
end if

do n = 1,nmicro_fields
  do k = 1,nzm
    mk0(k,n) = SUM(micro_field(1:nx,1:ny,k,n))/real(nx*ny)
  end do
end do
if(dompi) then
  call task_sum_real(mk0,tmp,nzm*nmicro_fields)
  mk0 = tmp/real(nsubdomains)
end if

if(doisotopes) then
  ! reset the reference profiles.
  mk_ref(:,:) = 0.

  mk_ref(:,iqv) = mk0(:,iqv)
  
  ! Use initprofile routine to map this qv profile onto 
  !   profiles of deltaD and deltaO18.
  do n = 1,niso
    tsurf = tabs_s 
    qsat_surf = qsatw(tsurf,pres(1))*salt_factor ! pres in hPa
    dummy(:) = 0.
    call initprofile_isotope(nzm,tsurf,qsat_surf, &
         tabs0,mk0(1,iqv),dummy,qv_iso,qc_iso,pres,iso_index(n))
    mk_ref(:,iqv_iso(n)) = qv_iso(:)
  end do
end if

end subroutine micro_diagnose

!----------------------------------------------------------------------
!!! functions to compute terminal velocity for precipitating variables:
!
! you need supply functions to compute terminal velocity for all of your 
! precipitating prognostic variables. Note that all functions should
! compute vertical velocity given two microphysics parameters var1, var2, 
! and temperature, and water vapor (single values, not arrays). Var1 and var2 
! are some microphysics variables like water content and concentration.
! Don't change the number of arguments or their meaning!

!!$real function term_vel_qr(qr,nr,tabs,rho)
!!$! .......  
!!$end function term_vel_qr
!!$
!!$real function term_vel_Nr(qr,nr,tabs,rho)
!!$! .......  
!!$end function term_vel_Nr
!!$
!!$real function term_vel_qs(qs,ns,tabs,rho)
!!$! .......  
!!$end function term_vel_qs

! etc.

!----------------------------------------------------------------------
!!! compute sedimentation 
!
!  The perpose of this subroutine is to prepare variables needed to call
! the precip_all() for each of the falling hydrometeor varibles
subroutine micro_precip_fall()

! before calling precip_fall() for each of falling prognostic variables,
! you need to set hydro_type and omega(:,:,:) variables.
! hydro_type can have four values:
! 0 - variable is liquid water mixing ratio
! 1 - hydrometeor is ice mixing ratio
! 2 - hydrometeor is mixture-of-liquid-and-ice mixing ratio. (As in original SAM microphysics).
! 3 - variable is not mixing ratio, but, for example, rain drop concentration
! OMEGA(:,:,:) is used only for hydro_type=2, and is the fraction of liquid phase (0-1).
! for hour hypothetical case, there is no mixed hydrometeor, so omega is not actually used.

integer hydro_type
real omega(nx,ny,nzm) 

integer i,j,k

return ! do not need this routine -- sedimentation done in m2005micro.

!!$! Initialize arrays that accumulate surface precipitation flux
!!$
!!$ if(mod(nstep-1,nstatis).eq.0.and.icycle.eq.1) then
!!$   do j=1,ny
!!$    do i=1,nx
!!$     precsfc(i,j)=0.
!!$    end do
!!$   end do
!!$   do k=1,nzm
!!$    precflux(k) = 0.
!!$   end do
!!$ end if
!!$
!!$ do k = 1,nzm ! Initialize arrays which hold precipitation fluxes for stats.
!!$    qpfall(k)=0.
!!$    tlat(k) = 0.
!!$ end do
!!$   
!!$! Compute sedimentation of falling variables:
!!$
!!$ hydro_type=0
!!$ call precip_fall(qr, term_vel_qr, hydro_type, omega)
!!$ hydro_type=3
!!$ call precip_fall(Nr, term_vel_Nr, hydro_type, omega)
!!$ hydro_type=1
!!$ call precip_fall(qs, term_vel_qs, hydro_type, omega)
!!$ hydro_type=3
!!$ call precip_fall(Ns, term_vel_Ns, hydro_type, omega)
!!$ hydro_type=1
!!$ call precip_fall(qg, term_vel_qg, hydro_type, omega)
!!$ hydro_type=3
!!$ call precip_fall(Ng, term_vel_Ng, hydro_type, omega)
!!$


end subroutine micro_precip_fall

!----------------------------------------------------------------------
! called when stepout() called

subroutine micro_print()
  implicit none
  integer :: k

  ! print out min/max values of all microphysical variables
  do k=1,nmicro_fields
     call fminmax_print(trim(mkname(k))//':', &
          micro_field(:,:,:,k),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm)
  end do

end subroutine micro_print

!----------------------------------------------------------------------
!!! Initialize the list of microphysics statistics that will be outputted
!!  to *.stat statistics file

subroutine micro_hbuf_init(namelist,deflist,unitlist,status,average_type,count,microcount)


character(*) namelist(*), deflist(*), unitlist(*)
integer status(*),average_type(*),count,microcount

character*8 name
character*80 longname
character*10 units

integer :: nn, n, ii, jj, ncond

microcount = 0

name = 'QTFLUX'
longname = 'Total (resolved + subgrid) total water (vapor+cloud) flux'
units = 'W/m2'
call add_to_namelist(count,microcount,name,longname,units,0)

name = 'PRECICE'
longname = 'Ice-phase precipitation rate'
units = 'mm/day'
call add_to_namelist(count,microcount,name,longname,units,0)

if(doisotopes) then
  do nn = 1,niso
    name = isoname(nn) // 'PREC'
    longname = 'Precipitation rate of ' // isoname(nn) // ', assuming VSMOW=1'
    units = 'mm/day'
    call add_to_namelist(count,microcount,name,longname,units,0)

    name = isoname(nn) // 'PRECI'
    longname = 'Ice-phase Precipitation rate of ' // isoname(nn) // ', assuming VSMOW=1'
    units = 'mm/day'
    call add_to_namelist(count,microcount,name,longname,units,0)
  end do
end if

! add area fraction of microphysical field to statistics
name = 'CLDSNWFR'
longname = 'Fracion of radiatively active hydrometerors (cl liq+cl ice+snow)'
units = '1'
call add_to_namelist(count,microcount,name,longname,units,0)

do n = 1,nmicro_fields
  if(n.ne.iqv) then
    ! add mean value of microphysical field to statistics
    !   EXCEPT for water vapor (added in statistics.f90)
    name = trim(mkname(n))
    longname = trim(mklongname(n))
    units = trim(mkunits(n))
    call add_to_namelist(count,microcount,name,longname,units,0)
  end if

  if(is_water_vapor(n)) then
    ! add Reference value for vapor and vapor isotopologues
    name = trim(mkname(n))//'_REF'
    longname = 'Reference value of '//trim(mklongname(n))// &
         ' from observations or initialization'
    units = trim(mkunits(n))
    call add_to_namelist(count,microcount,name,longname,units,0)
  end if

  ! define units for tendencies
  units = trim(mkunits(n))//'/day'

  ! add advective tendency
  name = trim(mkname(n))//'ADV'
  longname = 'Tendency of '//trim(mklongname(n))// &
       ' due to resolved vertical advection'
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add diffusive tendency
  name = trim(mkname(n))//'DIFF'
  longname = 'Tendency of '//trim(mklongname(n))// &
       ' due to vertical SGS transport'
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add tendency due to large-scale vertical advection
  name = trim(mkname(n))//'LSAD'
  longname = 'Tendency of '//trim(mklongname(n))// &
       ' due to large-scale vertical advection'
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add tendency due to microphysical processes
  name = trim(mkname(n))//'MPHY'
  longname = 'Tendency of '//trim(mklongname(n))// &
       ' due to microphysical processes'
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add vertical diffusive tendency
  name = trim(mkname(n))//'SED'
  longname = 'Tendency of '//trim(mklongname(n))//' due to sedimentation'
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! define units for fluxes
  if(flag_number(n).gt.0) then
    units = '#/m2/s'
  else
    units = 'kg/m2/s'
  end if

  ! add flux of microphysical fields to scalar
  name = trim(mkname(n))//'FLXR'
  longname = 'Resolved flux of '//trim(mklongname(n))
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add subgrid flux of microphysical fields to scalar
  name = trim(mkname(n))//'FLXS'
  longname = 'Subgrid flux of '//trim(mklongname(n))
  call add_to_namelist(count,microcount,name,longname,units,0)

  ! add sedimentation flux of microphysical fields to scalar
  name = trim(mkname(n))//'SDFL'
  longname = 'Sedimentation flux of '//trim(mklongname(n))
  call add_to_namelist(count,microcount,name,longname,units,0)

  if((n.eq.iqcl).OR.(n.eq.iqci).OR.(n.eq.iqs)) then
    ! add area fraction of microphysical field to statistics
    name = trim(mkname(n))//'FRAC'
    longname = trim(mklongname(n))//' FRACTION'
    units = '1'
    call add_to_namelist(count,microcount,name,longname,units,0)

    ! add approximate optical depth of hydrometeor fields
    name = 'TAU'//trim(mkname(n))
    longname = 'Approx optical depth of '//trim(mklongname(n))
    units = '1'
    call add_to_namelist(count,microcount,name,longname,units,0)

    ! add field which can be used to recover mean effective radius.
    name = trim(mkname(n))//'OEFFR'
    longname = 'Mixing ratio of '//trim(mklongname(n)) &
         //' over effective radius, EFFR = ' &
         //trim(mkname(n))//'/'//trim(mkname(n))//'OEFFR'
    units = 'g/kg/microns'
    call add_to_namelist(count,microcount,name,longname,units,0)
  end if

end do

call add_to_namelist(count,microcount,'QLAT', &
     'Sensible energy tendency due to phase changes', 'K/day',0)
 
do ncond = 1,ncondavg
   ! add conditional averages of hydrometeor fields
   call add_to_namelist(count,microcount,'QLAT' // TRIM(condavgname(ncond)), &
        'Sensible energy tendency due to phase changes in ' // TRIM(condavglongname(ncond)), &
        'K/day',ncond)

   do n = 1,nmicro_fields
      call add_to_namelist(count,microcount,trim(mkname(n)) // TRIM(condavgname(ncond)), &
           trim(mklongname(n)) // ' in ' // TRIM(condavglongname(ncond)), &
           trim(mkunits(n)),ncond)
   end do

end do

if(masterproc) then
   write(*,*) 'Added ', microcount, ' arrays to statistics for THOMPSON microphysics'
end if

end subroutine micro_hbuf_init

!----------------------------------------------------------------------
!!!! Collect microphysics history statistics (vertical profiles)
!! Note that only the fields declared in micro_hbuf_init() are allowed to
! be collected

subroutine micro_statistics()
  implicit none

real, dimension(nzm) :: tr0, tr2

real tmp(2), factor_xy
integer i,j,k,m, n, ii, jj, nn, ncond

call t_startf ('micro_statistics')

factor_xy = 1./float(nx*ny)

do n = 1,nmicro_fields
   do k = 1,nzm
      tmp(1) = dz
      tmp(2) = dz/dtn
      tr0(k) = SUM(micro_field(1:nx,1:ny,k,n))
      tr2(k) = SUM(micro_field(1:nx,1:ny,k,n)*micro_field(1:nx,1:ny,k,n))
      mkwle(k,n) = mkwle(k,n)*tmp(2) ! resolved flux - output as kg/m2/s
      mkwsb(k,n) = mkwsb(k,n)*tmp(1) ! subgrid flux - output as kg/m2/s
      mksed(k,n) = mksed(k,n) ! sedimentation flux - output as kg/m2/s
   end do

   if(n.ne.iqv) then
     ! mean microphysical field
     call hbuf_put(trim(mkname(n)),tr0,mkoutputscale(n)*factor_xy)
   end if

   if(is_water_vapor(n)) then
     ! reference/initialization value for vapor fields
     call hbuf_put(trim(mkname(n))//'_REF',mk_ref(1,n),mkoutputscale(n))
   end if

   ! do not rescale fluxes
   call hbuf_put(trim(mkname(n))//'FLXR',mkwle(1,n),factor_xy)
   call hbuf_put(trim(mkname(n))//'FLXS',mkwsb(1,n),factor_xy)
   call hbuf_put(trim(mkname(n))//'SDFL',mksed(1,n),factor_xy)

   ! tendencies
   call hbuf_put(trim(mkname(n))//'ADV', &
        mkadv(:,n),mkoutputscale(n)*factor_xy*86400./dtn)
   call hbuf_put(trim(mkname(n))//'DIFF', &
        mkdiff(:,n),mkoutputscale(n)*factor_xy*86400./dtn)
   call hbuf_put(trim(mkname(n))//'LSAD', &
        mklsadv(:,n),mkoutputscale(n)*factor_xy*86400.)
   call hbuf_put(trim(mkname(n))//'MPHY', &
        mtend(:,n),mkoutputscale(n)*factor_xy*86400.)
   call hbuf_put(trim(mkname(n))//'SED', &
        stend(:,n),mkoutputscale(n)*factor_xy*86400.)

   do ncond = 1,ncondavg
      do k = 1,nzm
         tr0(k) = SUM(micro_field(1:nx,1:ny,k,n)*condavg_mask(1:nx,1:ny,k,ncond))
      end do
      if(flag_number(n).eq.1) tr0(:) = tr0(:)*rho(:) ! remove factor of rho from number concentrations
      call hbuf_put(TRIM(mkname(n)) // TRIM(condavgname(ncond)), &
           tr0,mkoutputscale(n))
   end do

end do

!bloss/qt: in total water formulation, fluxes of qv and qcl computed together.
tr0(:) = lfac(iqv)*( mkwle(1:nzm,iqv) + mkwsb(1:nzm,iqv) ) &! qv + qcl tendencies
     + lfac(iqcl)*( mkwle(1:nzm,iqcl) + mkwsb(1:nzm,iqcl) ) 
if(doicemicro) then
   tr0(:) = tr0(:) + lfac(iqci)*( mkwle(1:nzm,iqci) + mkwsb(1:nzm,iqci) )
end if
call hbuf_put('QTFLUX',tr0,factor_xy)

! ice and isotope precipitation
call hbuf_put('PRECICE',preciceflux,factor_xy/dt*dz*86400./(nstatis+1.e-5) )
if(doisotopes) then
  do nn = 1,niso
    call hbuf_put( TRIM(isoname(nn)) // 'PREC', iso_precflux(1,nn), &
         factor_xy/dt*dz*86400./(nstatis+1.e-5) )
    call hbuf_put( TRIM(isoname(nn)) // 'PRECI', iso_preciceflux(1,nn), &
         factor_xy/dt*dz*86400./(nstatis+1.e-5) )
  end do
end if

! (sensible) latent heating due to microphysical processes
do k = 1,nzm
   tr0(k) = SUM(tmtend3d(1:nx,1:ny,k))
end do
call hbuf_put('QLAT',tr0,factor_xy*86400.)

do ncond = 1,ncondavg
   do k = 1,nzm
      tr0(k) = SUM(tmtend3d(1:nx,1:ny,k)*condavg_mask(1:nx,1:ny,k,ncond))
   end do
   call hbuf_put('QLAT' // TRIM(condavgname(ncond)),tr0,86400.)
end do

!bloss: output area fraction of radiatively-active hydrometeors
call hbuf_put('CLDSNWFR',mfrac(1,1),factor_xy) ! cloud-snow fraction

!bloss: output fraction, approx optical depth and Q/EFFR for 
!  cloud liquid, cloud ice and snow
do n = 1,nmicro_fields
  if ((n.eq.iqcl).OR.(n.eq.iqci).OR.(n.eq.iqs)) then
    ! fractional area of microphysical field > 1.e-6
    call hbuf_put(trim(mkname(n))//'FRAC',mfrac(1,n),factor_xy)

    ! approx optical depth
    call hbuf_put('TAU'//trim(mkname(n)),trtau(:,n),factor_xy)

    !bloss (Apr 09): Make an alternate statistic that can be used
    ! to easily compute the mean effective radius in a consistent
    ! way from optical depth.  This quantity Q*OEFFR is essentially
    ! the layer optical depth scaled in such a way that
    !
    !    EFFR = <Q*> / <Q*OEFFR>
    !
    ! where <.> is a time- and horizontal average.
    tr2(:) = 0.
    tr2(1) = trtau(1,n) / (1.e6*1.5e-3*rho(1)*dz*adz(1)*1.e-3)
    do k = 2,nzm
      tr2(k) = (trtau(k,n)-trtau(k-1,n)) / (1.e6*1.5e-3*rho(k)*dz*adz(k)*1.e-3) 
    end do
    call hbuf_put(trim(mkname(n))//'OEFFR',tr2,factor_xy)

  end if
end do

!bloss: compute number concentration metrics
if(doaerosols) then
  nn = 0.
  tmp(1)=0.
  do k=1,nzm
   do j=1,ny
    do i=1,nx
      if(qcl(i,j,k).gt.0.) then
         tmp(1) = tmp(1) + micro_field(i,j,k,incl)*1.e-6
         nn = nn + 1
       end if
    end do
   end do      
  end do
  if (nn.gt.0) ncmn = ncmn + tmp(1)/dble(nn)
else
  ncmn = Nc0
end if
if(doprecip) then
  nn = 0.
  tmp(1)=0.
  do k=1,nzm
   do j=1,ny
    do i=1,nx 
      if(micro_field(i,j,k,iqr).gt.0.) then 
         tmp(1) = tmp(1) + micro_field(i,j,k,inr)*1.e-6
         nn = nn + 1
       end if
    end do
   end do
  end do
  if (nn.gt.0) then
      nrainy = nrainy + 1
      nrmn = nrmn + tmp(1)/dble(nn)
  end if
else
  nrmn = 0.
end if

call t_stopf ('micro_statistics')

end subroutine micro_statistics

!-----------------------------------------
subroutine micro_stat_2Dinit(ResetStorage)
  implicit none
  integer, intent(in) :: ResetStorage

  ! initialize microphysical 2D outputs
  evap_h2o_xy(:,:) = 0.
  prec_h2o_xy(:,:) = 0.
  if(doisotopes) then
    iso_evap_xy(:,:,:) = 0.
    iso_prec_xy(:,:,:) = 0.
  end if

  if(ResetStorage.eq.1) then
    !bloss: If computing storage terms for individual hydrometeors,
    !         store initial profiles for computation of storage terms in budgets
    !bloss mkstor(:,:) = mk0(:,:)
  end if

end subroutine micro_stat_2Dinit
!-----------------------------------------
subroutine micro_write_fields2D(nfields1)
  implicit none
  integer, intent(inout) :: nfields1
  integer :: i, j, nn
  real(4) tmp(nx,ny,nzm)
  real :: coef
  character *80 long_name
  character *8 name
  character *10 units

  if(save2Davg) then
    coef = 1./float(nsave2D)
  else
    coef = 1.
  end if

  if(doisotopes) then
    ! if isotopes are enabled, output surface fluxes (in water mass, not energy, units)
    !   of both standard isotope and heavy isotopologues.
    nfields1=nfields1+1
    do j=1,ny
      do i=1,nx
        tmp(i,j,1)=evap_h2o_xy(i,j)*rhow(1)*coef
        evap_h2o_xy(i,j) = 0.
      end do
    end do
    name='EVAPH2O'
    long_name='Instantaneous surface evaporation rate (for standard isotope H216O)'
    units='kg/m2/s'
    call compress3D(tmp,nx,ny,1,name,long_name,units, &
         save2Dbin,dompi,rank,nsubdomains)

    do nn = 1,niso
      nfields1=nfields1+1
      do j=1,ny
        do i=1,nx
          tmp(i,j,1)=iso_evap_xy(i,j,nn)*rhow(1)*coef
          iso_evap_xy(i,j,nn) = 0.
        end do
      end do
      name='EVAP' // TRIM(mkname(iqv_iso(nn)))
      long_name='Instantaneous surface evaporation rate for ' // TRIM(mkname(iqv_iso(nn))) //' (relative to SMOW)'
      units='kg/m2/s'
      call compress3D(tmp,nx,ny,1,name,long_name,units, &
           save2Dbin,dompi,rank,nsubdomains)
    end do

    if(doprecip) then
      nfields1=nfields1+1
      do j=1,ny
        do i=1,nx
          tmp(i,j,1)=prec_h2o_xy(i,j)*dz/dt*coef
          prec_h2o_xy(i,j) = 0.
        end do
      end do
      name='PRECH2O'
      long_name='Surface precipitation rate (for standard isotope H216O)'
      units='kg/m2/s'
      call compress3D(tmp,nx,ny,1,name,long_name,units, &
           save2Dbin,dompi,rank,nsubdomains)

      ! output isotopic precip rate accumulated over the last 2D output period.
      do nn = 1,niso
        nfields1=nfields1+1
        do j=1,ny
          do i=1,nx
            tmp(i,j,1)=iso_prec_xy(i,j,nn)*dz/dt*coef
            iso_prec_xy(i,j,nn) = 0.
          end do
        end do
        name='PREC' // TRIM(mkname(iqv_iso(nn)))
        long_name='Surface precipitation rate for ' // TRIM(mkname(iqv_iso(nn))) //' (relative to SMOW)'
        units='kg/m2/s'
        call compress3D(tmp,nx,ny,1,name,long_name,units, &
             save2Dbin,dompi,rank,nsubdomains)
      end do


    end if
  end if

end subroutine micro_write_fields2D

!-----------------------------------------
subroutine micro_write_fields3D(nfields1)
  implicit none
  integer, intent(inout) :: nfields1

  ! nothing here yet

end subroutine micro_write_fields3D

!-----------------------------------------
subroutine satadj_liquid(nzm,tabs,qt,qc,pres)
  !bloss/qt: Utility routine based on cloud.f90 in 
  !  MICRO_SAM1MOM that was written by Marat Khairoutdinov.
  !  This routine performs a saturation adjustment for
  !  cloud liquid water only using a Newton method.
  !  While 20 iterations are allowed, most often this
  !  routine should exit in five iterations or less.
  !  Only a single calculation of the saturation vapor
  !  pressure is required in subsaturated air.

  implicit none

  integer, intent(in) :: nzm
  real, intent(inout), dimension(nzm) :: tabs ! absolute temperature, K
  real, intent(inout), dimension(nzm) :: qt  ! on input: qt; on output: qv
  real, intent(out), dimension(nzm) :: qc ! cloud liquid water, kg/kg
  real, intent(in), dimension(nzm) :: pres ! pressure, Pa

  real tabs1, dtabs, thresh, esat1, qsat1, fff, dfff
  integer k, niter

  integer, parameter :: maxiter = 20

  !bloss/qt: quick saturation adjustment to compute cloud liquid water content.
  do k = 1,nzm
    tabs1 = tabs(k) 
    qsat1 = qsatw( tabs(k), 0.01*pres(k) ) ! convert to hPa
    qc(k) = 0. ! no cloud unless qt > qsat
    
    if (qt(k).gt.qsat1) then

      ! if unsaturated, nothing to do (i.e., qv=qt, T=Tl) --> just exit.
      ! if saturated, do saturation adjustment 
      !    (modeled after Marat's cloud.f90).

      ! generate initial guess based on above calculation of qsat
      dtabs = + fac_cond*MAX(0.,qt(k) - qsat1) &
           / ( 1. + fac_cond*dtqsatw( tabs1, 0.01*pres(k) ) ) ! convert to hPa
      tabs1 = tabs1 + dtabs
      niter = 1

      ! convergence threshold: min of 0.01K and latent heating due to
      !    condensation of 1% of saturation mixing ratio.
      thresh = MIN(0.01, 0.01*fac_cond*qsat1)

      ! iterate while temperature increment > thresh and niter < maxiter
      do while((ABS(dtabs).GT.thresh) .AND. (niter.lt.maxiter))

        qsat1 = qsatw(tabs1, 0.01*pres(k) ) ! saturation mixing ratio, convert pressure to hPa

        fff = tabs(k) - tabs1 + fac_cond*MAX(0.,qt(k) - qsat1)
        dfff = 1. + fac_cond*dtqsatw( tabs1, 0.01*pres(k) ) ! convert to hPa
        dtabs = fff/dfff
        tabs1 = tabs1 + dtabs

        niter = niter + 1

      end do

      qc(k) = MAX( 0.,tabs1 - tabs(k) )/fac_cond ! cloud liquid mass mixing ratio
      qt(k) = qt(k) - qc(k) ! This now holds the water vapor mass mixing ratio.
      tabs(k) = tabs1 ! update temperature.
      
      if(niter.gt.maxiter-1) write(*,*) 'Reached iteration limit in satadj_liquid'

    end if ! qt_in > qsat

  end do ! k = 1,nzm

end subroutine satadj_liquid

!-----------------------------------------------------------------------
! Supply function that computes total water in a domain:
!
real(8) function total_water()

  real(8) tmp
  integer i,j,k,m

  total_water = 0.
  do m=1,nmicro_fields
   if(flag_wmass(m).eq.1) then
    do k=1,nzm
      tmp = 0.
      do j=1,ny
        do i=1,nx
          tmp = tmp + micro_field(i,j,k,m)
        end do
      end do
      total_water = total_water + tmp*adz(k)*dz*rho(k)
    end do
   end if
  end do

end function total_water

!-----------------------------------------------------------------------
! Supply function that computes the horizontally-averaged total water path in a domain:
!
subroutine total_waters(twp)

  real(8) tmp, twp(1+niso)
  integer i,j,k,m, nn

  m = 1

  twp(m) = 0.
  do k=1,nzm
    tmp = 0.
    do j=1,ny
      do i=1,nx
        tmp = tmp + micro_field(i,j,k,iqv) + micro_field(i,j,k,iqcl) + micro_field(i,j,k,iqr)
      end do
    end do
    if(doicemicro) then
      do j=1,ny
        do i=1,nx
          tmp = tmp + micro_field(i,j,k,iqci) + micro_field(i,j,k,iqs) + micro_field(i,j,k,iqg)
        end do
      end do
    end if
    twp(m) = twp(m) + tmp*adz(k)*dz*rho(k)
  end do

  if(doisotopes) then
    do nn = 1,niso
      m = m + 1
      twp(m) = 0.
      do k=1,nzm
        tmp = 0.
        do j=1,ny
          do i=1,nx
            tmp = tmp + micro_field(i,j,k,iqv_iso(nn)) &
                 + micro_field(i,j,k,iqcl_iso(nn)) + micro_field(i,j,k,iqr_iso(nn))
          end do
        end do
        if(doicemicro) then
          do j=1,ny
            do i=1,nx
              tmp = tmp + micro_field(i,j,k,iqci_iso(nn)) &
                   + micro_field(i,j,k,iqs_iso(nn)) + micro_field(i,j,k,iqg_iso(nn))
            end do
          end do
        end if
        twp(m) = twp(m) + tmp*adz(k)*dz*rho(k)
      end do
    end do
  end if

  twp(:) = twp(:)/float(nx*ny)

end subroutine total_waters

logical function micro_provides_reffc()
  micro_provides_reffc = douse_reffc
end function micro_provides_reffc

logical function micro_provides_reffi()
  micro_provides_reffi = douse_reffi
end function micro_provides_reffi

function Get_reffc() ! liquid water
  real, dimension(nx,ny,nzm) :: Get_reffc
  Get_reffc = reffc
end function Get_reffc

function Get_reffi() ! ice
  real, dimension(nx,ny,nzm) :: Get_reffi
  Get_reffi = reffi
end function Get_reffi

function Get_nca() ! aerosol
  real, pointer, dimension(:,:,:) :: Get_nca
  Get_nca = 0.
end function Get_nca

end module microphysics



