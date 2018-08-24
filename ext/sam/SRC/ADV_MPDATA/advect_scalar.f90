
subroutine advect_scalar (f,fadv,flux,f2leadv,f2legrad,fwleadv,do_poslimit,doit)
 	
!     positively definite monotonic advection with non-oscillatory option

use grid
use vars, only: u, v, w, rho, rhow
use params, only: docolumn

implicit none

real f(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm)
real flux(nz), fadv(nz)
real f2leadv(nzm),f2legrad(nzm),fwleadv(nzm)
logical do_poslimit !bloss: added for compatibility with SELPPM
logical doit

real df(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm)
real f0(nzm),df0(nzm),fff(nz),factor
real coef
integer i,j,k

if(docolumn) then
  flux = 0.
  return
end if

call t_startf ('advect_scalars')

if(dostatis) then
	
 df(:,:,:) = f(:,:,:)

endif

if(RUN3D) then
  call advect_scalar3D(f, u, v, w, rho, rhow, flux)
else
  call advect_scalar2D(f, u, w, rho, rhow, flux)	  
endif


if(dostatis) then

  do k=1,nzm
    fadv(k)=0.
    do j=1,ny
     do i=1,nx
      fadv(k)=fadv(k)+f(i,j,k)-df(i,j,k)
     end do
    end do
  end do

end if

if(dostatis.and.doit) then
	
  call stat_varscalar(f,df,f0,df0,f2leadv)
  call stat_sw2(f,df,fwleadv)


!  Compute advection flux of variance
 

  do k=1,nzm
    do j=dimy1_s,dimy2_s
     do i=dimx1_s,dimx2_s
      df(i,j,k) = (df(i,j,k)-df0(k))**2
     end do
    end do
  end do

  coef = max(1.e-10,maxval(df(dimx1_s:dimx2_s, dimy1_s:dimy2_s, 1:nzm)))
  df(:,:,:) = df(:,:,:) / coef
  if(RUN3D) then
   call advect_scalar3D(df, u, v, w, rho, rhow, fff)
  else
   call advect_scalar2D(df, u, w, rho, rhow, fff)	  
  endif
  df(:,:,:) = df(:,:,:) * coef

  factor=dz/(nx*ny*dtn)
  do k = 1,nzm
    fff(k)=fff(k) * factor
  end do
  fff(nz)=0.
  do k = 1,nzm
    f2legrad(k) = f2leadv(k)
    f2leadv(k)=-(fff(k+1)-fff(k))/(dz*adz(k)*rho(k))	 
    f2legrad(k)=f2legrad(k)-f2leadv(k)
  end do
endif

call t_stopf ('advect_scalars')

end subroutine advect_scalar

