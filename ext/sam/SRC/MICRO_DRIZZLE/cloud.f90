
subroutine cloud

!  Condensation of cloud water/cloud ice.

use vars
use microphysics
use micro_params
use params

implicit none

integer i,j,k, kb, kc, kmax
real dtabs, tabs1
real fac1  
real fff,dfff,qsatt,dqsat
real lstarn,lstarp
real coef,dqi,lat_heat,vt_cl, coef_cl
real qiu, qic, qid, tmp_theta, tmp_phi
real fz(nx,ny,nz)
integer niter

fac1 = fac_cond
kmax=0

call t_startf ('cloud')

do k = 1, nzm
 do j = 1, ny
  do i = 1, nx

    q(i,j,k)=max(0.,q(i,j,k))


! Initail guess for temperature assuming no cloud water/ice:


    tabs(i,j,k) = t(i,j,k)-gamaz(k)
    tabs1=tabs(i,j,k)+fac1*qp(i,j,k)

    tabs1=tabs(i,j,k)+fac_cond*qp(i,j,k)
    qsatt = qsatw(tabs1,pres(k))

!  Test if condensation is possible:


    if(q(i,j,k).gt.qsatt) then

      niter=0
      dtabs = 100.
      do while(abs(dtabs).gt.0.01.and.niter.lt.10)
	lstarn=fac_cond
	lstarp=fac_cond
	qsatt=qsatw(tabs1,pres(k))
	dqsat=dtqsatw(tabs1,pres(k))
	fff = tabs(i,j,k)-tabs1+lstarn*(q(i,j,k)-qsatt)+lstarp*qp(i,j,k)
	dfff=-lstarn*dqsat-1.
	dtabs=-fff/dfff
	niter=niter+1
	tabs1=tabs1+dtabs
      end do   

      qsatt = qsatt + dqsat * dtabs
      qn(i,j,k) = max(0.,q(i,j,k)-qsatt)
      kmax = max(kmax,k)

    else

      qn(i,j,k) = 0.

    endif

    tabs(i,j,k) = tabs1
    qp(i,j,k) = max(0.,qp(i,j,k)) ! just in case

  end do
 end do
end do

!
! Take into account sedimentation of cloud water which may be important for stratocumulus case.
! Parameterization of sedimentation rate is taken from GCSS WG1 DYCOMS2_RF2 case, and base on
! Rogers and Yau, 1989

coef_cl = 1.19e8*(3./(4.*3.1415*rho_water*Nc0*1.e6))**(2./3.)*exp(5.*log(sigmag)**2)

fz = 0.

! Compute cloud ice flux (using flux limited advection scheme, as in
! chapter 6 of Finite Volume Methods for Hyperbolic Problems by R.J.
! LeVeque, Cambridge University Press, 2002).
do k = 1,kmax
   ! Set up indices for x-y planes above and below current plane.
   kc = min(nzm,k+1)
   kb = max(1,k-1)
   ! CFL number based on grid spacing interpolated to interface i,j,k-1/2
   coef = dtn/(0.5*(adz(kb)+adz(k))*dz)
   do j = 1,ny
      do i = 1,nx
         ! Compute cloud water density in this cell and the ones above/below.
         ! Since cloud ice is falling, the above cell is u (upwind),
         ! this cell is c (center) and the one below is d (downwind).
         
         qiu = rho(kc)*qn(i,j,kc)
         qic = rho(k) *qn(i,j,k) 
         qid = rho(kb)*qn(i,j,kb)
         
         ! Ice sedimentation velocity depends on ice content. The fiting is
         ! based on the data by Heymsfield (JAS,2003). -Marat
         ! 0.1 m/s low bound was suggested by Chris Bretherton
         vt_cl = coef_cl*(qic+1.e-12)**(2./3.)
         
         ! Use MC flux limiter in computation of flux correction.
         ! (MC = monotonized centered difference).
         if (qic.eq.qid) then
            tmp_phi = 0.
         else
            tmp_theta = (qiu-qic)/(qic-qid)
            tmp_phi = max(0.,min(0.5*(1.+tmp_theta),2.,2.*tmp_theta))
         end if
         
         ! Compute limited flux.
         ! Since falling cloud ice is a 1D advection problem, this
         ! flux-limited advection scheme is monotonic.
         fz(i,j,k) = -vt_cl*(qic - 0.5*(1.-coef*vt_cl)*tmp_phi*(qic-qid))
      end do
   end do
end do
fz(:,:,nz) = 0.

do k=1,kmax
   coef=dtn/(dz*adz(k)*rho(k))
   do j=1,ny
      do i=1,nx
         ! The cloud ice increment is the difference of the fluxes.
         dqi=coef*(fz(i,j,k)-fz(i,j,k+1))
         ! Add this increment to both non-precipitating and total water.
         qn(i,j,k) = qn(i,j,k) + dqi
         q(i,j,k)  = q(i,j,k)  + dqi
         ! Include this effect in the total moisture budget.
         qifall(k) = qifall(k) + dqi
         
         ! The latent heat flux induced by the falling cloud water enters
         ! the liquid-ice static energy budget in the same way as the
         ! precipitation.   
         lat_heat  = fac_cond*dqi
         ! Add divergence of latent heat flux to liquid-ice static energy.
         t(i,j,k)  = t(i,j,k)  - lat_heat
         ! Add divergence to liquid-ice static energy budget.
         tlatqi(k) = tlatqi(k) - lat_heat
      end do
   end do
end do

call t_stopf ('cloud')

end subroutine cloud

