subroutine advect_all_scalars()

  use vars
  use microphysics
  use sgs
  use tracers
  use params, only: dotracers
  implicit none
  real dummy(nz)
  integer k
  logical do_poslimit
  logical compute_variance_stats


!---------------------------------------------------------
!      advection of scalars :

     do_poslimit = .false.
     compute_variance_stats = .true.
     call advect_scalar(t,tadv,twle,t2leadv,t2legrad,twleadv,do_poslimit,compute_variance_stats)
    
!
!    Advection of microphysics prognostics:
!

     do_poslimit = .true.
     do k = 1,nmicro_fields
        if(k.eq.index_water_vapor) then
          ! transport water-vapor variable no metter what
          compute_variance_stats = .true.
          call advect_scalar(micro_field(:,:,:,k),mkadv(:,k),mkwle(:,k), &
               q2leadv,q2legrad,qwleadv,do_poslimit,compute_variance_stats)
        elseif ((docloud.and.flag_precip(k).ne.1)   & 
             .or.(doprecip.and.flag_precip(k).eq.1) ) then
          compute_variance_stats = .false.
          call advect_scalar(micro_field(:,:,:,k),mkadv(:,k),mkwle(:,k), &
               dummy,dummy,dummy,do_poslimit,compute_variance_stats)
        end if
     end do

!
!    Advection of sgs prognostics:
!

     if(dosgs.and.advect_sgs) then
       do_poslimit = .true. !bloss: Is positivity a good assumption here??
       compute_variance_stats = .false.
       do k = 1,nsgs_fields
           call advect_scalar(sgs_field(:,:,:,k),sgsadv(:,k),sgswle(:,k),dummy,dummy,dummy, &
                do_poslimit,compute_variance_stats)
       end do
     end if


!
!   Precipitation fallout:
!
    if(doprecip) then

       total_water_prec = total_water_prec + total_water()

       call micro_precip_fall()

       total_water_prec = total_water_prec - total_water()


    end if

 ! advection of tracers:

     if(dotracers) then
       do_poslimit = .true. !bloss: Is positivity a good assumption here??
       compute_variance_stats = .false.
        do k = 1,ntracers
         call advect_scalar(tracer(:,:,:,k),tradv(:,k),trwle(:,k),dummy,dummy,dummy, &
                do_poslimit,compute_variance_stats)
        end do

     end if

end subroutine advect_all_scalars
