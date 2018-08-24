  subroutine cldefrint(m,n,landfrac,tlayer,rel,rei,psurface, player,landm,icefrac,snowh)
!-----------------------------------------------------------------------
!
! interface for cldefr to work with isccp simulator calls and CAM3 radiation
!
!-----------------------------------------------------------------------
    use parkind, only: kind_rb
    use grid, only: nzm
    use cam_rad_parameterizations, only : &
        computeRe_Liquid, computeRe_Ice
    implicit none
!------------------------------Parameters-------------------------------

! Input arguments
!
  integer m,n, k
  real landfrac(1)
  real icefrac(1)       ! Ice fraction
  real psurface(1)      ! Surface pressure
  real tlayer(1,nzm)   ! Temperature
  real player(1,nzm)   !  Midpoint pressures
  real landm(1)         ! Land fraction
  real snowh(1)      ! snow depth, water equivalent (meters)

!
! Output arguments
!
  real rel(nzm)   ! Liquid effective drop size (microns)
  real rei(nzm)   ! Ice effective drop size (microns)

  do k = 1,nzm
    rel(k) = real( computeRe_Liquid( real(tlayer(1,k), kind_rb), &
                                     real(landm(1), kind_rb), &
                                     real(icefrac(1), kind_rb), &
                                     real(snowh(1), kind_rb) ) )
    rei(k) = real( computeRe_Ice( real(tlayer(1,k), kind_rb) ) )
  end do

  return
  end
 
