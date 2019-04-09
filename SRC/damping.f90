
subroutine damping()

!  "Spange"-layer damping at the domain top region

use vars
use microphysics, only: micro_field, index_water_vapor
implicit none

real tau_min	! minimum damping time-scale (at the top)
real tau_max    ! maxim damping time-scale (base of damping layer)
real damp_depth ! damping depth as a fraction of the domain height
parameter(tau_min=600., tau_max=10800., damp_depth=0.3)
real tau(nzm)   
integer i, j, k, n_damp

call t_startf ('damping')

if(tau_min.lt.2*dt) then
   print*,'Error: in damping() tau_min is too small!'
   call task_abort()
end if

do k=nzm,1,-1
 if(z(nzm)-z(k).lt.damp_depth*z(nzm)) then 
   n_damp=nzm-k+1
 endif
end do

do k=nzm,nzm-n_damp,-1
 tau(k) = tau_min *(tau_max/tau_min)**((z(nzm)-z(k))/(z(nzm)-z(nzm-n_damp)))
 tau(k)=1./tau(k)
end do

do k = nzm, nzm-n_damp, -1
   do j=1,ny
    do i=1,nx
      dudt(i,j,k,na)= dudt(i,j,k,na)-(u(i,j,k)-u0(k)) * tau(k)
      dvdt(i,j,k,na)= dvdt(i,j,k,na)-(v(i,j,k)-v0(k)) * tau(k)
      dwdt(i,j,k,na)= dwdt(i,j,k,na)-w(i,j,k) * tau(k)
!      t(i,j,k)= t(i,j,k)-dtn*(t(i,j,k)-t0(k)) * tau(k)
!      micro_field(i,j,k,index_water_vapor)= micro_field(i,j,k,index_water_vapor)- &
!                                    dtn*(qv(i,j,k)+qcl(i,j,k)+qci(i,j,k)-q0(k)) * tau(k)
    end do! i 
   end do! j
end do ! k

call t_stopf('damping')
end subroutine damping
