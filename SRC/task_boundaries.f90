
        subroutine task_boundaries(flag)
        	
!  These routines exchanges overlaping information for various subdomains to
!  be able to calculate various quantities in common /com3d/
!  near the subdomain boundaries.

use vars
use microphysics
use sgs
use params, only: dotracers, dosgs
use tracers
implicit none

integer flag,i

if(flag.eq.0) then

 call task_exchange(u,dimx1_u,dimx2_u,dimy1_u,dimy2_u,nzm,1,1,1,1,1)
 call task_exchange(v,dimx1_v,dimx2_v,dimy1_v,dimy2_v,nzm,1,1,1,1,2)
 ! use w at the top level  - 0s anyway - to exchange the sst boundaries (for
 ! surface fluxes call
 w(1:nx,1:ny,nz) = sstxy(1:nx,1:ny)
 call task_exchange(w,dimx1_w,dimx2_w,dimy1_w,dimy2_w,nz,1,1,1,1,3)	
 sstxy(0:nx,1-YES3D:ny) = w(0:nx,1-YES3D:ny,nz)
 w(0:nx+1,1-YES3D:ny+YES3D,nz) = 0. ! fill it back with 0s

endif

if(flag.eq.2) then

 call task_exchange(u,dimx1_u,dimx2_u,dimy1_u,dimy2_u,nzm,2,3,2+NADV,2+NADV,1)
 call task_exchange(v,dimx1_v,dimx2_v,dimy1_v,dimy2_v,nzm,2+NADV,2+NADV,2,3,2)
 call task_exchange(w,dimx1_w,dimx2_w,dimy1_w,dimy2_w,nz,2+NADV,2+NADV,2+NADV,2+NADV,3)	

 call task_exchange(t,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,3+NADVS,3+NADVS,3+NADVS,3+NADVS,4)
 do i = 1,nsgs_fields
    if(dosgs.and.advect_sgs) &
     call task_exchange(sgs_field(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm, &
                                                           3+NADVS,3+NADVS,3+NADVS,3+NADVS,4+i)
 end do
 do i = 1,nmicro_fields
    if(   i.eq.index_water_vapor             &
     .or. docloud.and.flag_precip(i).ne.1    &
     .or. doprecip.and.flag_precip(i).eq.1 ) &
     call task_exchange(micro_field(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm, &
                                3+NADVS,3+NADVS,3+NADVS,3+NADVS,4+nsgs_fields+nsgs_fields_diag+i)
 end do
 if(dotracers) then
   do i=1,ntracers
     call task_exchange(tracer(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm, &
                   3+NADVS,3+NADVS,3+NADVS,3+NADVS,4+nsgs_fields+nsgs_fields_diag+nmicro_fields+i)
   end do
 end if


endif


if(flag.eq.3) then

 call task_exchange(t,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,1,1,1,1,4)
 do i = 1,nsgs_fields
    if(dosgs.and.advect_sgs) &
     call task_exchange(sgs_field(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,1,1,1,1,4+i)
 end do
 do i = 1,nmicro_fields
    if(   i.eq.index_water_vapor             &
     .or. docloud.and.flag_precip(i).ne.1    &
     .or. doprecip.and.flag_precip(i).eq.1 ) &
     call task_exchange(micro_field(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm, &
                                                             1,1,1,1,4+nsgs_fields+nsgs_fields_diag+i)
 end do
 if(dotracers) then
   do i=1,ntracers
     call task_exchange(tracer(:,:,:,i),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm, &
                              1,1,1,1,4+nsgs_fields+nsgs_fields_diag+nmicro_fields+i)
   end do
 end if

end if

if(flag.eq.4) then

 do i = 1,nsgs_fields_diag
    if(dosgs) &
     call task_exchange(sgs_field_diag(:,:,:,i),dimx1_d,dimx2_d,dimy1_d,dimy2_d,nzm, &
                   1+dimx1_d,dimx2_d-nx,YES3D+dimy1_d,1-YES3D+dimy2_d-ny,4+nsgs_fields+i)
 end do

end if


end
	
	
