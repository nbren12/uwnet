module sgs

! This is a template subgrid-scale (SGS) parameterization module that should be followed to
! implement any new SGS closure/parameterization. It is also useful to use the origial SAM SGS 
! parameterization (1.5-order TKE/Smagorinsky) in SGS_TKE directory as an example.
! Marat Khairoutdinov, 2012

!Instructions:

! First, create a new directory in SRC that will contain ALL source files (and data) that
! represent you new SGS parameterization. You SHOULD NOT touch any of the files
! in the SRC directory; otherwise, it would defit the whole purpose of having a SGS interface.

! Edit the Build script to set environmental variable SGS_DIR to be the name of your SGS directory,
! so that the compiler could have correct path to it.

! include this grid information:

use grid, only: nx, ny, nz, nzm, & ! grid dimensions; nzm=nz-1 - # of levels for all scalars
                dimx1_s,dimx2_s,dimy1_s,dimy2_s ! actual scalar-array dimensions

!----------------------------------------------------------------------
! Required definitions:
! The following definitions (variables) are required:
! The example of 1.5-order TKE closure will be used.  This closure uses 1 prognostic field
! (transported around the domain) and two diagnostic fields (eddy diffusivity and conductivity).
! THose diagnostic fields have boundaries that are one grid point wider in x and y directions than 
! the domain size (to be able to compute derivative). The code in SRC will make sure that
! prognostic variables are advected and boundaries of diagnostic are updated.
!
! Set the total number of prognostic fields:

integer, parameter :: nsgs_fields = 1   ! total number of prognostic sgs vars

!!! prognostic scalar fields (that need to be advected arround the grid):
! Note that all prognostic scalar fields should have the same spatial  dimensions as below:

real sgs_field(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm, nsgs_fields)

!!! sgs diagnostic variables that need to exchange boundary information (via MPI):

integer, parameter :: nsgs_fields_diag = 2   ! total number of diagnostic sgs vars

! The following dimensions of the diagnostic fields are set in stone; don;t change them!
! diagnostic fields' boundaries:
integer, parameter :: dimx1_d=0, dimx2_d=nxp1, dimy1_d=1-YES3D, dimy2_d=nyp1

real sgs_field_diag(dimx1_d:dimx2_d, dimy1_d:dimy2_d, nzm, nsgs_fields_diag)

logical, parameter:: advect_sgs = .true. ! advect prognostics
logical, parameter:: do_sgsdiag_bound = .true.  ! exchange boundaries for diagnostics fields

! SGS prognostic and diagnostic 3D fields that output by default as 3D snapshots (if set to 1).
integer, parameter :: flag_sgs3Dout(nsgs_fields) = (/0/)
integer, parameter :: flag_sgsdiag3Dout(nsgs_fields_diag) = (/0,0/)

! those prognostic fields' fluxes should be present:
real fluxbsgs (nx, ny, 1:nsgs_fields) ! surface fluxes 
real fluxtsgs (nx, ny, 1:nsgs_fields) ! top boundary fluxes 

!!! these arrays may be needed for sgs output statistics:

real sgswle(nz,1:nsgs_fields)  ! resolved vertical flux
real sgswsb(nz,1:nsgs_fields)  ! SGS vertical flux
real sgsadv(nz,1:nsgs_fields)  ! tendency due to vertical advection
real sgslsadv(nz,1:nsgs_fields)  ! tendency due to large-scale vertical advection
real sgsdiff(nz,1:nsgs_fields)  ! tendency due to vertical diffusion

!------------------------------------------------------------------
! internal (optional) definitions:

! make aliases for prognostic variables, so that you could use them by name 
! rather than generic sgs_field:

real tke(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm)   ! SGS TKE
equivalence (tke(dimx1_s,dimy1_s,1),sgs_field(dimx1_s,dimy1_s,1,1))

! Similarly, you can make aliases for diagnostic variables:

real tk  (dimx1_d:dimx2_d, dimy1_d:dimy2_d, nzm) ! SGS eddy viscosity
real tkh (dimx1_d:dimx2_d, dimy1_d:dimy2_d, nzm) ! SGS eddy conductivity
equivalence (tk(dimx1_d,dimy1_d,1), sgs_field_diag(dimx1_d, dimy1_d,1,1))
equivalence (tkh(dimx1_d,dimy1_d,1), sgs_field_diag(dimx1_d, dimy1_d,1,2))


CONTAINS
! The subroutines below are all required as they are called from the main body of SAM.
! if you don't need a certain required subroutine/function, just leave it's body blank,
! but definition should present for the compiler not to complain.

! Required microphysics subroutines and function:

!----------------------------------------------------------------------
!!! Read microphysics options from prm (namelist) file
! 
  ! read any options in from prm file here using a namelist named
  !   after your SGS package, e.g. SGS_3 
  ! Note, for all namelist parameters you should always specify default value in case
  ! it is not explicitly set in prm namelist file.


subroutine sgs_setparm()

  use grid, only: case
  implicit none

  integer ierr, ios, ios_missing_namelist, place_holder

  NAMELIST /SHS_3/ &
       param_w, & ! Some parameter param_w
       param_t    ! Some parameter param_t

  NAMELIST /BNCUIODSBJCB/ place_holder

  param_w = 3.5 ! default value
  param_t = 1.5 ! default value

  !----------------------------------
  !  Read namelist for microphysics options from prm file:
  !------------
  open(55,file='./'//trim(case)//'/prm', status='old',form='formatted')

  read (UNIT=55,NML=BNCUIODSBJCB,IOSTAT=ios_missing_namelist)
  rewind(55) !note that one must rewind before searching for new namelists

  read (55,SGS_TKE,IOSTAT=ios)


  if (ios.ne.0) then
     !namelist error checking
     if(ios.ne.ios_missing_namelist) then
        write(*,*) '****** ERROR: bad specification in SGS_TKE namelist'
        call task_abort()
     end if
  end if
  close(55)

end subroutine sgs_setparm

!----------------------------------------------------------------------
!!! Initialize sgs:

! This subroutine is called by SAM at the beginning of each run, initial or restart.

subroutine sgs_init()


  if(nrestart.eq.0) then

     sgs_field = 0.
     sgs_field_diag = 0.

     fluxbsgs = 0.
     fluxtsgs = 0.

  end if

  if(masterproc) then
        write(*,*) 'Prognostic SGS Closure version 3'
  end if

  sgswle = 0.
  sgswsb = 0.
  sgsadv = 0.
  sgsdiff = 0.
  sgslsadv = 0.

! add your code from here:

end subroutine sgs_init

!----------------------------------------------------------------------
!!! make some initial noise in sgs:
! in case you need to initialize your fields at the beginnnig of each run using value of 
! perturbation type, here is your place:
!
subroutine setperturb_sgs(ptype)

integer, intent(in) :: ptype

select case (ptype)

  case(0)


  case(1)

  case(2)

  case(3)   ! gcss wg1 smoke-cloud case

  case(4)  ! gcss wg1 arm case

  case(5)  ! gcss wg1 BOMEX case

  case(6)  ! GCSS Lagragngian ASTEX

  case default

end select

end subroutine setperturb_sgs

!----------------------------------------------------------------------
!!! Estimate Courant number limit for SGS
! Here you need to return the value of the courant number needed for stability
! of your scheme. Courant number is defined as dt/dt_info, where dt is model
! timestep, and dt_info, the minimum time for information propagation ovet the
! grid cell. For example, for advection in x direction dt_info=dx/dt; for
! diffusion with diffusion coefficient dt_info = 0.5*dx^2/tk. If you don;t know
! the courant number for your SGS scheme, just return cfl=0.
!

subroutine kurant_sgs(cfl)

cfl = 0.

end subroutine kurant_sgs


!----------------------------------------------------------------------
!!! compute sgs tendences for velocity components, that is dudt, dvdt, and dwdt
! It is guarantied, that the velocities arrays are filled including 1 extra point
! in both x and y directisions. For example, not only u(1:nx,1:ny,nzm), but also
! u(0:nx+1,0:ny+1,nzm). For 2D case, the size of all arrays in y direction is 1.

subroutine sgs_mom()



end subroutine sgs_mom

!----------------------------------------------------------------------
!!! compute sgs tendences for all SAM's prognostic scalars (see vars.f90) as well as
! SGS prognostic scalars. SAM only advects all prognostic scalars using resolved velocity.
! SGS contribution is your responsibility. 
! It is guarantied, that all prognostic scalars' arrays are filled including 1 extra layer
! in both x and y directions. For example, not only t(1:nx,1:ny,nzm) is known, but also
! t(0:nx+1,0:ny+1,nzm).. For 2D case, the size of all arrays in y direction is 1.

!
subroutine sgs_scalars()


end subroutine sgs_scalars

!----------------------------------------------------------------------
!!! compute sgs processes (beyond advection):
! Here is where all the computations for updating your SGS model on the current time step 
! are done (except for SGS effects on momentum and prognostic scalars that are done elsewhere).
!
subroutine sgs_proc()


end subroutine sgs_proc

!----------------------------------------------------------------------
!!! Diagnose arrays nessesary for dynamical core and statistics:
! Could use this one to make some usefull diagnostics done at the end of each time step.

subroutine sgs_diagnose()

end subroutine sgs_diagnose


!----------------------------------------------------------------------
!!!! Collect microphysics history statistics (vertical profiles)
! This subroutine is called right after the main statistics subroutine is called.
! If you wanna output some horizontally averaged profiles, here is the place to compute those.

subroutine sgs_statistics()
  

end subroutine sgs_statistics

!----------------------------------------------------------------------
! called when stepout() called
! Output minimum and maximum values of your arrays for printout at regular intervals
! as specified by nprint parameter:

subroutine sgs_print()


end subroutine sgs_print

!----------------------------------------------------------------------
!!! Initialize the list of sgs statistics 
! Look at implementation of microphysics for specific examples

subroutine sgs_hbuf_init(namelist,deflist,unitlist,status,average_type,count,sgscount)
character(*) namelist(*), deflist(*), unitlist(*)
integer status(*),average_type(*),count,sgscount

end subroutine sgs_hbuf_init


end module sgs



