    
subroutine upperbound

use vars
use params
use microphysics, only: micro_field, index_water_vapor, is_water_vapor, nmicro_fields, mk0, mklsadv
implicit none
real coef, coefxy
integer i,j,k,n
real, parameter ::  tau_nudging = 3600.
real a,b,c,d,e

call t_startf ('upperbound')

coefxy = 1./float(nx*ny)/dtn

if(dolargescale.AND.(.NOT.doExtrapolate_UpperBound)) then

!
! if there is an "observed" sounding - nudge two highest levels to it
! to avoid problems with the upper boundary.
!

  coef = dtn / tau_nudging
  do k=nzm-1,nzm
    do j=1,ny
      do i=1,nx
         !bloss: add tendency to *NUDGE so that it is included in the
         !  vertically-integrated energy and moisture budgets.
         tnudge(k) = tnudge(k) - (t(i,j,k)-tg0(k)-gamaz(k))*coef*coefxy
         qnudge(k) = qnudge(k) - (micro_field(i,j,k,index_water_vapor)-qg0(k))*coef*coefxy

         t(i,j,k)=t(i,j,k)-(t(i,j,k)-tg0(k)-gamaz(k))*coef
         micro_field(i,j,k,index_water_vapor)=micro_field(i,j,k,index_water_vapor)- &
                              (micro_field(i,j,k,index_water_vapor)-qg0(k))*coef
      end do
    end do
  end do

else

!  otherwise, preserve the vertical gradients:
! MK: change starting from v. 6.8.3: limit gradients
  coef = dz*adz(nzm)
  gamt0=max(0.1e-3,(t0(nzm-1)-t0(nzm-2))/(z(nzm-1)-z(nzm-2)))
!bloss  gamq0=min(0.,(q0(nzm-1)-q0(nzm-2))/(z(nzm-1)-z(nzm-2)))
  do j=1,ny
   do i=1,nx
     !bloss: add tendency to *NUDGE so that it is included in the
     !  vertically-integrated energy and moisture budgets.
     tnudge(nzm) = tnudge(nzm) + (t(i,j,nzm-1)+gamt0*coef-t(i,j,nzm))*coefxy
!bloss     qnudge(nzm) = qnudge(nzm) + (max(0.,micro_field(i,j,nzm-1,index_water_vapor)+gamq0*coef) &
!bloss                                     - micro_field(i,j,nzm,index_water_vapor))*coefxy

     t(i,j,nzm)=t(i,j,nzm-1)+gamt0*coef
!bloss     micro_field(i,j,nzm,index_water_vapor)=max(0.,micro_field(i,j,nzm-1,index_water_vapor)+gamq0*coef)
   end do    
  end do 

!bloss: Vertical gradients of water vapor at the model top can lead to trouble
!   in long simulations.  A long-time RCE simulation with large-scale vertical
!   motion at the top of the model had cloud appearing in the top model level.
!   To avoid this, relax the water vapor in the top model level to the avgerage
!   of the four model levels below.
  coef = dtn / tau_nudging
  gamq0=0.25*SUM(q0(nzm-4:nzm-1))
  do j=1,ny
   do i=1,nx
     !bloss: add tendency to *NUDGE so that it is included in the
     !  vertically-integrated energy and moisture budgets.
     qnudge(nzm) = qnudge(nzm) + coef*( gamq0 - micro_field(i,j,nzm,index_water_vapor) )*coefxy

     micro_field(i,j,nzm,index_water_vapor)= micro_field(i,j,nzm,index_water_vapor) &
          + ( gamq0 - micro_field(i,j,nzm,index_water_vapor) )*coef
   end do    
  end do 

  !bloss: add extrapolation for other water vapor variables (e.g., water isotopologues)
  !  We'd like to make the treatment of isotopic water as close to that of standard water
  !  as we can to prevent spurious fractionation.
  coef = dtn / tau_nudging
  do n = 1,nmicro_fields
    if( is_water_vapor(n) .AND. (n.ne.index_water_vapor) ) then
      gamq0=0.25*SUM(mk0(nzm-4:nzm-1,n))
      do j=1,ny
        do i=1,nx
          !bloss: add tendency to *NUDGE so that it is included in the
          !  vertically-integrated energy and moisture budgets.
          mklsadv(nzm,n) = mklsadv(nzm,n) + coef*( gamq0 - micro_field(i,j,nzm,n) )*coefxy

          micro_field(i,j,nzm,n)= micro_field(i,j,nzm,n) &
               + ( gamq0 - micro_field(i,j,nzm,n) )*coef
        end do
      end do
    end if
  end do

!
! make gradient accross two highest scalar lavels the same preserving the mass-weighted integral
! (experimental, not proven to work yet)
!
!   a = adz(nzm)/adz(nzm-1)
!   d = adzw(nzm-1)*rho(nzm-1)/(adzw(nzm-1)*rhow(nzm-1)+adzw(nzm)*rhow(nzm))
!   e = adzw(nzm)*rhow(nzm)/(adzw(nzm-1)*rhow(nzm-1)+adzw(nzm)*rhow(nzm))
!   b = 1./(a + d)
!   a = a*b
!   c = b*e
!   do j=1,ny
!    do i=1,nx
!      t(i,j,nzm)=a*t(i,j,nzm)+b*t(i,j,nzm-1)-c*t(i,j,nzm-2)
!      t(i,j,nzm-1)=d*t(i,j,nzm)+e*t(i,j,nzm-2)
!      micro_field(i,j,nzm,index_water_vapor)=max(0., &
!              a*micro_field(i,j,nzm,index_water_vapor)+b*micro_field(i,j,nzm-1,index_water_vapor) &
!              -c*micro_field(i,j,nzm-2,index_water_vapor))
!      micro_field(i,j,nzm-1,index_water_vapor)=d*micro_field(i,j,nzm,index_water_vapor)+ &
!               e*micro_field(i,j,nzm-2,index_water_vapor)
!    end do    
!   end do 


           
end if

call t_stopf('upperbound')

end   
