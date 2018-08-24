subroutine advect_scalar( f, fadv, flux, f2leadv, f2legrad, fwleadv, do_poslimit, doit )
 	
!	5th order ultimate-macho advection scheme
! Yamaguchi, T., D. A. Randall, and M. F. Khairoutdinov, 2011:
! Cloud Modeling Tests of the ULTIMATE-MACHO Scalar Advection Scheme.
! Monthly Weather Review. 139, pp.3248-3264


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
	logical, intent(in) :: do_poslimit !bloss: added for compatibility with SELPPM
	logical, intent(in) :: doit
	
	!	local
	real, dimension(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm) :: df
	real, dimension(nzm) :: f0, df0
	real, dimension(nz) :: fff
	real :: factor, coef
	integer :: i, j, k, kmin, kmax
	
	
	if(docolumn) then
		flux = 0.
		return
	endif
	
	call t_startf ('advect_scalars')
	
        ! If the upper and bottom parts of the domain are all zeros,
        !   don't bother computing fluxes there.  Only compute
        !   between kmin and kmax
        if((f(1,1,nzm).ne.0.).OR.compute_advection_everywhere) then
          ! if there are non-zeros at the top level (as for water vapor, energy)
          !   compute fluxes for the whole domain.
          kmin = 1
          kmax = nzm
        else
          ! check for non-zeros, starting from the top of the domain.
          kmax = -1
          do k = nzm,1,-1
            if(MAXVAL(ABS(f(:,:,k))).ne.0.) then
              kmax = k
              EXIT
            end if
          end do

          kmin = MIN(nzm,kmax+2)
          do k = 1,kmax
            if(MAXVAL(ABS(f(:,:,k))).ne.0.) then
              kmin = k
              EXIT
            end if
          end do

!bloss          write(*,*) 'kmin = ', kmin, ' kmax = ', kmax
        end if

        if(kmax.lt.0) then
          ! without any non-zero values, return.
          flux = 0.
          f2leadv = 0.
          f2legrad = 0.
          call t_stopf ('advect_scalars')
          return
        endif

	if (dostatis) then
		df(:,:,:) = f(:,:,:)
	endif
	
	if (RUN3D) then
		call advect_scalar3D(f, u, v, w, rho, rhow, flux, kmin, kmax)
	else
		call advect_scalar2D(f, u, w, rho, rhow, flux, kmin, kmax)	  
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

                kmin = 1
                kmax = nzm
		if (RUN3D) then
			call advect_scalar3D(df, u, v, w, rho, rhow, fff, kmin, kmax)
		else
			call advect_scalar2D(df, u, w, rho, rhow, fff, kmin, kmax)	  
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
