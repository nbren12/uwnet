subroutine advect_scalar( f, fadv, flux, f2leadv, f2legrad, fwleadv, do_poslimit, doit )
 	
! Driver routine for Blossey-Durran advection.
! For a description of the scheme (3rd order PPM with selective flux correction), see
!  Peter N. Blossey, Dale R. Durran 2008. Selective monotonicity preservation in 
!     scalar advection, Journal of Computational Physics, Vol. 227, pp. 5160-5183, 
!     doi:10.1016/j.jcp.2008.01.043

  !bloss: this routine seems agnostic to the choice of advection scheme.

!	At this point, 
!	u = u * rho * dtn / dx
!	v = v * rho * dtn / dy
!	w = w * rho * dtn / dz
!	Division by adz has not been performed.

	use grid
	use vars, only: u, v, w, rho, rhow
        use params, only: docolumn
	
	implicit none
	
	!	input
	real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm), intent(inout) :: f
	real, dimension(nz), intent(out) :: flux, fadv
	real, dimension(nzm), intent(out) :: f2leadv, f2legrad, fwleadv
	logical, intent(in) :: do_poslimit, doit
	
	!	local
	real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm) :: df
	real, dimension(nzm) :: f0, df0, f_ref
	real, dimension(nz) :: fff
	real :: factor, coef, tmpsum
	integer :: i, j, k
	
	
	if(docolumn) then
		flux = 0.
		return
	endif
	
	call t_startf ('advect_scalars')
	
        ! check to see if there are any non-zero scalar values.
        if(f(1,1,1).eq.0.) then
          tmpsum = 0.
          do k = 1,nzm
            tmpsum = tmpsum + SUM(f(:,:,k)*f(:,:,k))
            if(tmpsum.gt.0.) EXIT
          end do

          if(tmpsum.eq.0.) then
            ! without any non-zero values, return.
            flux = 0.
            f2leadv = 0.
            f2legrad = 0.
            call t_stopf ('advect_scalars')
            return
          end if
        endif

        !bloss: placeholder for future input of reference scalar profiles.
        !  This will let the model work with open boundary conditions at the top
        !  or bottom.
        f_ref(:) = 0.
!!$        if((MAXVAL(w(:,:,1)).gt.0.).OR.(MINVAL(w(:,:,nz)).lt.0.)) then
!!$          write(*,*) '**** Error in ADV_SELPPM/advect_scalar'
!!$          write(*,*) ' Input appropriate reference scalar profile to enable'
!!$          write(*,*) ' inflow conditions at the top/bottom boundaries'
!!$          call task_abort()
!!$        end if
          
          

	if (dostatis) then
		df(:,:,:) = f(:,:,:)
	endif
	
	if (RUN3D) then
		call advect_scalar3D(f, u, v, w, rho, rhow, flux, f_ref, do_poslimit)
	else
		call advect_scalar2D(f, u, w, rho, rhow, flux, f_ref, do_poslimit)	  
	endif
	
	if (dostatis) then
		do k = 1, nzm
			fadv(k) = 0.
			do j = 1, ny
				do i = 1, nx
					fadv(k) = fadv(k) + f(i,j,k) - df(i,j,k)
				enddo
			enddo
		enddo
	endif
	
	if (dostatis.and.doit) then
		call stat_varscalar( f, df, f0, df0, f2leadv )
		call stat_sw2( f, df, fwleadv )
		
		! Compute advection flux of variance
		do k = 1, nzm
			do j = dimy1_s, dimy2_s
				do i = dimx1_s, dimx2_s
					df(i,j,k) = ( df(i,j,k) - df0(k) ) ** 2
				enddo
			enddo
		enddo
		coef = max( 1.e-10, maxval( df(dimx1_s:dimx2_s,dimy1_s:dimy2_s,1:nzm) ) )
		df(:,:,:) = df(:,:,:) / coef

		if (RUN3D) then
			call advect_scalar3D(df, u, v, w, rho, rhow, fff, f_ref, do_poslimit)
		else
			call advect_scalar2D(df, u, w, rho, rhow, fff, f_ref, do_poslimit)	  
		endif
		df(:,:,:) = df(:,:,:) * coef
		factor=dz/(nx*ny*dtn)
		do k = 1,nzm
			fff(k)=fff(k) * factor
		enddo
		fff(nz)=0.
		do k = 1,nzm
			f2legrad(k) = f2leadv(k)
			f2leadv(k)=-(fff(k+1)-fff(k))/(dz*adz(k)*rho(k))	 
			f2legrad(k)=f2legrad(k)-f2leadv(k)
		enddo
	endif
	
	call t_stopf ('advect_scalars')

end subroutine advect_scalar
