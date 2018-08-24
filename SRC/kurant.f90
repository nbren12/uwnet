
subroutine kurant

use vars
use sgs, only: kurant_sgs_level_by_level
use params, only: ncycle_max, ncycle_min, ncycle0, cfl_safety_factor

implicit none

!bloss: Revised kurant routine that uses tighter CFL constraint and the ability to 
!  provide a safety factor to keep runs with larger timesteps away from the possiblity
!  of encoutering instability.  Based in part on a version of kurant.f90 developed by
!  Peter Caldwell while he was at UW.

integer i, j, k, ncycle1(1),ncycle2(1), ncycleold
real wm(nz)
real um(nz)
real vm(nz)

real cfl_adv_z(nzm), cfl_sgs_z(nzm), cfl_cond

! Note that cfl_sgs is a CFL-like number ~max|tkh|*(dt/dx^2 + dt/dy^2 + dt/dz^2) 
!   for diffusion at a given level.  note that it has been modified to account
!   for scaling of diffusivity in horizontal

!PMC-set equal to maxcfl and pass to MPI fn to keep the fn from
!reading/writing to same variable at once (messes up C, not sure about f90)
real maxcfl1(5), maxcfl2(5), maxcfl

!bloss: CFL limits for third-order adams-bashforth with second-order advection and diffusion
real, parameter :: cfl_adv_max = 0.72, cfl_sgs_max = 0.5

call t_startf ('kurant')

ncycleold = ncycle
ncycle = 1
	
do k = 1,nzm
 um(k) = MAXVAL(ABS(u(:,:,k)))
 vm(k) = MAXVAL(ABS(v(:,:,k)))
 wm(k) = MAXVAL(ABS(w(:,:,k)))
end do
u_max=MAXVAL(um(:))
w_max=MAXVAL(wm(:))

do k=1,nzm
  cfl_adv_z(k) = um(k)*dt/dx + YES3D*vm(k)*dt/dy + max(wm(k),wm(k+1))*dt/(dz*adzw(k))
end do

call kurant_sgs_level_by_level(cfl_sgs_z)

!bloss: compute stability constraint assuming elliptical stability region, so that 
!   the stability limits for diffusion and advection can be combined as follows.
!   Note that the max CFL for advective and diffusive stability are defined above
!   and depend on the time-stepping scheme and order of the momentum (and scalar)
!   advection.  My impression is that momentum advection will usually be the tighter 
!   constraint in SAM, but this may not always be true.

cfl_cond = SQRT( MAXVAL( (cfl_adv_z(:)/cfl_adv_max)**2 &
                       + (cfl_sgs_z(:)/cfl_sgs_max)**2 ) )

! choose number of pieces to break dt into.  The cfl_safety_factor 
!   should be >1, and larger values will give a more conservative actual 
!   timestep dtn = dt/ncycle.
ncycle = max(ncycle_min, &  ! specified minimum number of substeps (default=1)
             ncycleold-1, & ! number of substeps may fall by at most one per timestep
             ceiling(cfl_cond*cfl_safety_factor) ) ! CFL constraint w/safety factor

!bloss: special conditions for beginning of run if trying to run with large time steps
if(nstep.eq.1) ncycle = max(ncycle,ncycle0)

if(dompi) then
  ncycle1(1)=ncycle
  call task_max_integer(ncycle1,ncycle2,1)
  ncycle=ncycle2(1)
end if

! PMC - will actually read "MAX CFL = " # which is actually sum of directional cfls
!       also, will ignore fact that diffusivity could also be constraining dt...
!       Ideas how to improve this output diagnostic?
maxcfl = cfl_cond/float(ncycle)*cfl_adv_max ! scale by CFLmax since cfl_cond=CFL/CFL_max. 
cfl_adv = MAXVAL(cfl_adv_z)/float(ncycle)
cfl_sgs = MAXVAL(cfl_sgs_z)/float(ncycle)

if(dompi) then
  maxcfl1(1) = maxcfl !PMC repeat for input to task_max_real
  maxcfl1(2) = cfl_adv
  maxcfl1(3) = cfl_sgs
  maxcfl1(4) = u_max
  maxcfl1(5) = w_max
  call task_max_real(maxcfl1,maxcfl2,5) !bloss
  maxcfl = maxcfl2(1)
  cfl_adv = maxcfl2(2)
  cfl_sgs = maxcfl2(3)
  u_max = maxcfl2(4)
  w_max = maxcfl2(5)
end if

if(ncycle.gt.ncycle_max) then
   if(masterproc) print *,'the number of required substeps ', ncycle, &
        ' exceeds the maxmimum allowed ', ncycle_max
   if(masterproc) print *,'Outputting 3D field for diagnostic purposes...'
   save3Dbin = .true.
   call write_fields3D() !bloss add output of 3D fields for diagnostics
   call task_abort()
end if

call t_stopf ('kurant')

end subroutine kurant	
