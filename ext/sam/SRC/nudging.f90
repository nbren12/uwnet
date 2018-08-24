subroutine nudging()
	
use vars
use params
use microphysics, only: micro_field, index_water_vapor
implicit none

real coef, coef1
integer i,j,k
	
call t_startf ('nudging')

tnudge = 0.
qnudge = 0.
unudge = 0.
vnudge = 0.

coef = 1./tauls

if(donudging_uv) then
    do k=1,nzm
      if(z(k).ge.nudging_uv_z1.and.z(k).le.nudging_uv_z2) then
        unudge(k)=unudge(k) - (u0(k)-ug0(k))*coef
        vnudge(k)=vnudge(k) - (v0(k)-vg0(k))*coef
        do j=1,ny
          do i=1,nx
             dudt(i,j,k,na)=dudt(i,j,k,na)-(u0(k)-ug0(k))*coef
             dvdt(i,j,k,na)=dvdt(i,j,k,na)-(v0(k)-vg0(k))*coef
          end do
        end do
      end if
    end do
endif

coef = 1./tautqls

if(donudging_tq.or.donudging_t) then
    coef1 = dtn / tautqls
    do k=1,nzm
      if(z(k).ge.nudging_t_z1.and.z(k).le.nudging_t_z2) then
        tnudge(k)=tnudge(k) -(t0(k)-tg0(k)-gamaz(k))*coef
        do j=1,ny
          do i=1,nx
             t(i,j,k)=t(i,j,k)-(t0(k)-tg0(k)-gamaz(k))*coef1
          end do
        end do
      end if
    end do
endif

if(donudging_tq.or.donudging_q) then
    coef1 = dtn / tautqls
    do k=1,nzm
      if(z(k).ge.nudging_q_z1.and.z(k).le.nudging_q_z2) then
        qnudge(k)=qnudge(k) -(q0(k)-qg0(k))*coef
        do j=1,ny
          do i=1,nx
             micro_field(i,j,k,index_water_vapor)=micro_field(i,j,k,index_water_vapor)-(q0(k)-qg0(k))*coef1
          end do
        end do
      end if
    end do
endif

call t_stopf('nudging')

end subroutine nudging
