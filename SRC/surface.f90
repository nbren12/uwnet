subroutine surface()
	
use vars
use params
use microphysics, only: micro_field, index_water_vapor
implicit none
	
real qvs(0:nx,1-YES3D:ny),t_s, q_s, u_h0
real taux0, tauy0, xlmo
real diag_ustar, coef, coef1
real fluxq0_coef, ustar0, u10n
real ws0
integer i,j
real(8) buffer(2), buffer1(2)

real dummy1, dummy2

! LES mode: 

call t_startf ('surface')


if(.not.SFC_FLX_FXD) then
  
  if(sstxy(1,1).le.-100.) then
     print*,'surface: sst is undefined. Quitting...'
     call task_abort()
  end if

  if(OCEAN) then

    if(LES) then

       call oceflx(pres(1),u0(1)+ug, v0(1)+vg,t0(1)-gamaz(1), q0(1),t0(1),z(1),&
                          sstxy(1,1)+t00, fluxt0, fluxq0, taux0, tauy0, q_s, &
                          fluxq0_coef, ustar0, u10n)
       if(SFC_TAU_FXD) then
         u_h0 = max(1.,sqrt((u0(1)+ug)**2+(v0(1)+vg)**2))
         taux0 = -(u0(1)+ug)/u_h0*tau0*rhow(1)
         tauy0 = -(v0(1)+vg)/u_h0*tau0*rhow(1)
       else
         tau0=sqrt( taux0**2 +  tauy0**2)/rhow(1)
       end if

       if(doFixedWindSpeedForSurfaceFluxes) then
         call oceflx(pres(1),WindSpeedForFluxes,0.,t0(1)-gamaz(1), q0(1),t0(1),z(1),&
                          sstxy(1,1)+t00, fluxt0, fluxq0, dummy1, dummy2, q_s, &
                          fluxq0_coef, ustar0, u10n)
       end if

       fluxbt(:,:) = fluxt0
       fluxbq(:,:) = fluxq0
       fluxbu(:,:) = taux0/rhow(1)
       fluxbv(:,:) = tauy0/rhow(1)
       ! extra parameters for surface fluxes of water isotopologues
       fluxbq_coef(:,:) = fluxq0_coef
       qsat_surf(:,:) = q_s
       ustar(:,:) = ustar0
       u10arr(:,:) = u10n !bloss(2018-02): For aerosol surface flux in M2005_PA

    end if ! LES

    if(CEM) then

       qvs(0:nx,1-YES3D:ny) = micro_field(0:nx,1-YES3D:ny,1,index_water_vapor)

       do j=1,ny
         do i=1,nx


           if(doFixedWindSpeedForSurfaceFluxes) then
             ! fixed wind speed
             call oceflx(pres(1),WindSpeedForFluxes, 0., &
                  t(i,j,1)-gamaz(1),qv(i,j,1),t(i,j,1),z(1), &
                  sstxy(i,j)+t00, fluxt0, fluxq0, taux0, tauy0, q_s, &
                  fluxq0_coef, ustar0, u10n)
           else
             ! normal call
             call oceflx(pres(1),0.5*(u(i+1,j,1)+u(i,j,1))+ug, &
                  0.5*(v(i,j+YES3D,1)+v(i,j,1))+vg, &
                  t(i,j,1)-gamaz(1),qv(i,j,1),t(i,j,1),z(1), &
                  sstxy(i,j)+t00, fluxt0, fluxq0, taux0, tauy0, q_s, &
                  fluxq0_coef, ustar0, u10n)
           end if

           fluxbt(i,j) = fluxt0
           fluxbq(i,j) = fluxq0
           ! extra parameters for surface fluxes of water isotopologues
           fluxbq_coef(i,j) = fluxq0_coef
           qsat_surf(i,j) = q_s
           ustar(i,j) = ustar0
           u10arr(i,j) = u10n

           call oceflx(pres(1),u(i,j,1)+ug, &
                       0.25*(v(i-1,j+YES3D,1)+v(i-1,j,1)+v(i,j+YES3D,1)+v(i,j,1))+vg, &
                       0.5*(t(i-1,j,1)+t(i,j,1))-gamaz(1),0.5*(qvs(i-1,j)+qvs(i,j)), &
                       0.5*(t(i-1,j,1)+t(i,j,1)),z(1), &
                       0.5*(sstxy(i-1,j)+sstxy(i,j))+t00, fluxt0, fluxq0, taux0, tauy0, q_s, &
                       fluxq0_coef, ustar0, u10n)
           if(SFC_TAU_FXD) then
             u_h0 = max(1.,sqrt((u(i,j,1)+ug)**2+ &
                     (0.25*(v(i-1,j+YES3D,1)+v(i-1,j,1)+v(i,j+YES3D,1)+v(i,j,1))+vg)**2))
             taux0 = -(u(i,j,1)+ug)/u_h0*tau0*rhow(1)
           end if
           fluxbu(i,j) = taux0/rhow(1)

           call oceflx(pres(1),0.25*(u(i+1,j-YES3D,1)+u(i,j-YES3D,1)+u(i+1,j,1)+u(i,j,1))+ug, &
                       v(i,j,1)+vg, &
                       0.5*(t(i,j-YES3D,1)+t(i,j,1))-gamaz(1),0.5*(qvs(i,j-YES3D)+qvs(i,j)), &
                       0.5*(t(i,j-YES3D,1)+t(i,j,1)),z(1), &
                       0.5*(sstxy(i,j-YES3D)+sstxy(i,j))+t00, fluxt0, fluxq0, taux0, tauy0, q_s, &
                       fluxq0_coef, ustar0, u10n)
           if(SFC_TAU_FXD) then
             u_h0 = max(1.,sqrt( &
                       (0.25*(u(i+1,j-YES3D,1)+u(i,j-YES3D,1)+u(i+1,j,1)+u(i,j,1))+ug)**2+ &
                       (v(i,j,1)+vg)**2))
             tauy0 = -(v(i,j,1)+vg)/u_h0*tau0*rhow(1)
           end if
           fluxbv(i,j) = tauy0/rhow(1)


         end do
       end do
	
    end if ! CEM

  end if ! OCEAN


  if(LAND) then

       if(doFixedWindSpeedForSurfaceFluxes) then
         write(*,*) 'Fixed Wind Speed not yet implemented for land fluxes'
         STOP 'in surface.f90'
       end if

            if(LES) then    

               coef = (1000./pres0)**(rgas/cp)
               coef1 = (1000./pres(1))**(rgas/cp)
               t_s = (sstxy(1,1)+t00)*coef
               q_s = soil_wetness*qsatw(sstxy(1,1)+t00,pres(1))
               call landflx(pres(1),(t0(1)-gamaz(1))*coef1, t_s,     &
                      q0(1), q_s, u0(1)+ug, v0(1)+vg, z(1), z0,      &
                      fluxt0, fluxq0, taux0, tauy0, xlmo)
               if(SFC_TAU_FXD) then
                 u_h0 = max(1.,sqrt((u0(1)+ug)**2+(v0(1)+vg)**2))
                 taux0 = -(u0(1)+ug)/u_h0*tau0*rhow(1)
                 tauy0 = -(v0(1)+vg)/u_h0*tau0*rhow(1)
               else
                 tau0=sqrt( taux0**2 +  tauy0**2)/rhow(1)
               end if

               fluxbt(:,:) = fluxt0
               fluxbq(:,:) = fluxq0
               fluxbu(:,:) = taux0/rhow(1)
               fluxbv(:,:) = tauy0/rhow(1)

            end if ! LES

            if(CEM) then

              coef = (1000./pres0)**(rgas/cp)
              coef1 = (1000./pres(1))**(rgas/cp)
              qvs(0:nx,1-YES3D:ny) = micro_field(0:nx,1-YES3D:ny,1,index_water_vapor)

              do j=1,ny  
               do i=1,nx

               t_s = (sstxy(i,j)+t00)*coef
               q_s = soil_wetness*qsatw(sstxy(i,j)+t00,pres(1))
               call landflx(pres(1),(t(i,j,1)-gamaz(1))*coef1, t_s,   &
                      qv(i,j,1), q_s, 0.5*(u(i+1,j,1)+u(i,j,1))+ug,     &
                        0.5*(v(i,j+YES3D,1)+v(i,j,1))+vg, z(1), z0,        &
                      fluxt0, fluxq0, taux0, tauy0, xlmo)
               fluxbt(i,j) = fluxt0
               fluxbq(i,j) = fluxq0

               t_s = (0.5*(sstxy(i-1,j)+sstxy(i,j))+t00)*coef
               q_s = soil_wetness*qsatw(0.5*(sstxy(i-1,j)+sstxy(i,j))+t00,pres(1))
               call landflx(pres(1),(0.5*(t(i-1,j,1)+t(i,j,1))-gamaz(1))*coef1, t_s,   &
                      0.5*(qvs(i-1,j)+qvs(i,j)), q_s, u(i,j,1)+ug,     &
                        0.25*(v(i-1,j+YES3D,1)+v(i-1,j,1)+v(i,j+YES3D,1)+v(i,j,1))+vg, &
                       z(1), z0, fluxt0, fluxq0, taux0, tauy0, xlmo)
               if(SFC_TAU_FXD) then
                   u_h0 = max(1.,sqrt((u(i,j,1)+ug)**2+ &
                        (0.25*(v(i-1,j+YES3D,1)+v(i-1,j,1)+v(i,j+YES3D,1)+v(i,j,1))+vg)**2))
                   taux0 = -(u(i,j,1)+ug)/u_h0*tau0*rhow(1)
               end if
               fluxbu(i,j) = taux0/rhow(1)

               t_s = (0.5*(sstxy(i,j-YES3D)+sstxy(i,j))+t00)*coef
               q_s = soil_wetness*qsatw(0.5*(sstxy(i,j-YES3D)+sstxy(i,j))+t00,pres(1))
               call landflx(pres(1),(0.5*(t(i,j-YES3D,1)+t(i,j,1))-gamaz(1))*coef1, t_s,   &
                      0.5*(qvs(i,j-YES3D)+qvs(i,j)), q_s,  &
                      0.25*(u(i+1,j-YES3D,1)+u(i,j-YES3D,1)+u(i+1,j,1)+u(i,j,1))+ug,     &
                      v(i,j,1)+vg, &
                      z(1), z0, fluxt0, fluxq0, taux0, tauy0, xlmo)
               if(SFC_TAU_FXD) then
                  u_h0 = max(1.,sqrt( &
                       (0.25*(u(i+1,j-YES3D,1)+u(i,j-YES3D,1)+u(i+1,j,1)+u(i,j,1))+ug)**2+ &
                       (v(i,j,1)+vg)**2))
                  tauy0 = -(v(i,j,1)+vg)/u_h0*tau0*rhow(1)
               end if
               fluxbv(i,j) = tauy0/rhow(1)

               end do
              end do

            end if ! CEM


  end if ! LAND

end if! .not.SFC_FLX_FXD



if(SFC_FLX_FXD) then

  u_h0 = max(1.,sqrt((u0(1)+ug)**2+(v0(1)+vg)**2))

  if(.not.SFC_TAU_FXD) then
    if(OCEAN) z0 = 0.0001  ! for LAND z0 should be set in namelist (default z0=0.035)

    tau0 = diag_ustar(z(1),  &
                bet(1)*(fluxt0+epsv*(t0(1)-gamaz(1))*fluxq0),u_h0,z0)**2  

  end if ! .not.SFC_TAU_FXD

  if(LES) then
    taux0 = -(u0(1)+ug)/u_h0*tau0
    tauy0 = -(v0(1)+vg)/u_h0*tau0
    fluxbu(:,:) = taux0
    fluxbv(:,:) = tauy0
  else
    fluxbu(:,:) = -(u(1:nx,1:ny,1)+ug)/u_h0*tau0
    fluxbv(:,:) = -(v(1:nx,1:ny,1)+vg)/u_h0*tau0
  end if

  fluxbt(:,:) = fluxt0
  fluxbq(:,:) = fluxq0

end if ! SFC_FLX_FXD

!
! Homogenize the surface scalar fluxes if needed for sensitivity studies
!
   if(dosfchomo) then

	fluxt0 = 0.
	fluxq0 = 0.
	do j=1,ny
         do i=1,nx
	   fluxt0 = fluxt0 + fluxbt(i,j)
	   fluxq0 = fluxq0 + fluxbq(i,j)
         end do
        end do
	fluxt0 = fluxt0 / float(nx*ny)
	fluxq0 = fluxq0 / float(nx*ny)
        if(dompi) then
            buffer(1) = fluxt0
            buffer(2) = fluxq0
            call task_sum_real8(buffer,buffer1,2)
	    fluxt0 = buffer1(1) /float(nsubdomains)
	    fluxq0 = buffer1(2) /float(nsubdomains)
        end if ! dompi
	fluxbt(:,:) = fluxt0
	fluxbq(:,:) = fluxq0

   end if

shf_xy(:,:) = shf_xy(:,:) + fluxbt(:,:) * dtfactor
lhf_xy(:,:) = lhf_xy(:,:) + fluxbq(:,:) * dtfactor

call t_stopf ('surface')

end




! ----------------------------------------------------------------------
!
! DISCLAIMER : this code appears to be correct but has not been
!              very thouroughly tested. If you do notice any
!              anomalous behaviour then please contact Andy and/or
!              Bjorn
!
! Function diag_ustar:  returns value of ustar using the below 
! similarity functions and a specified buoyancy flux (bflx) given in
! kinematic units
!
! phi_m (zeta > 0) =  (1 + am * zeta)
! phi_m (zeta < 0) =  (1 - bm * zeta)^(-1/4)
!
! where zeta = z/lmo and lmo = (theta_rev/g*vonk) * (ustar^2/tstar)
!
! Ref: Businger, 1973, Turbulent Transfer in the Atmospheric Surface 
! Layer, in Workshop on Micormeteorology, pages 67-100.
!
! Code writen March, 1999 by Bjorn Stevens
!
! Code corrected 8th June 1999 (obukhov length was wrong way up,
! so now used as reciprocal of obukhov length)

      real function diag_ustar(z,bflx,wnd,z0)

      implicit none
      real, parameter      :: vonk =  0.4   ! von Karmans constant
      real, parameter      :: g    = 9.81   ! gravitational acceleration
      real, parameter      :: am   =  4.8   !   "          "         "
      real, parameter      :: bm   = 19.3   !   "          "         "
      real, parameter      :: eps  = 1.e-10 ! non-zero, small number

      real, intent (in)    :: z             ! height where u locates
      real, intent (in)    :: bflx          ! surface buoyancy flux (m^2/s^3)
      real, intent (in)    :: wnd           ! wind speed at z
      real, intent (in)    :: z0            ! momentum roughness height

      integer :: iterate
      real    :: lnz, klnz, c1, x, psi1, zeta, rlmo, ustar

      lnz   = log(z/z0) 
      klnz  = vonk/lnz              
      c1    = 3.14159/2. - 3.*log(2.)

      ustar =  wnd*klnz
      if (bflx /= 0.0) then 
        do iterate=1,4
          rlmo   = -bflx * vonk/(ustar**3 + eps)   !reciprocal of
                                                   !obukhov length
          zeta  = z*rlmo
          if (zeta > 0.) then
            ustar =  vonk*wnd  /(lnz + am*zeta)
          else
            x     = sqrt( sqrt( 1.0 - bm*zeta ) )
            psi1  = 2.*log(1.0+x) + log(1.0+x*x) - 2.*atan(x) + c1
            ustar = wnd*vonk/(lnz - psi1)
          end if
        end do
      end if

      diag_ustar = ustar

      return
      end function diag_ustar
! ----------------------------------------------------------------------

