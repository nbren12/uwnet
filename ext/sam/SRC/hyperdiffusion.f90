module hyperdiffusion
contains
  subroutine hyper_diffuse
    use vars
    use microphysics, only: micro_field, index_water_vapor
    use params, only: docolumn, dowallx, dowally, khyp
    implicit none

    real rdx2,rdy2,rdz2,rdz,rdx25,rdy25, rdx16, rdy16
    real rdx21,rdy21,rdx251,rdy251,rdz25
    real dxy,dxz,dyx,dyz,dzx,dzy

    integer i,j,k,ic,ib,jb,jc,kc,kcu
    real tkx, tky, tkz, rhoi, iadzw, iadz
    real fu(0:nx,0:ny,nz),fv(0:nx,0:ny,nz),fw(0:nx,0:ny,nz)
    real  tsrc

    print *, 'hyperdiffusion.f90: diffusing momentum khyp=', khyp

    rdx2=1./(dx*dx)
    rdy2=1./(dy*dy)

    rdx25=0.25*rdx2
    rdy25=0.25*rdy2

    rdx16 = rdx25*rdx25
    rdy16 = rdy25*rdy25

    ! standard value for 250m was 0.1e8
    ! khyp = 1.5e4
    ! make grid-scale hyperdiffusive Reynold's number order 1 with velocity scale
    ! U ~ 50 m/s
    ! khyp = 0.5 * dx**4 /  (dx / 50.0)
    ! This value of hyperdiffusion recommended by Jablonowski in
    ! The Pros and Cons of Diffusion, Filters and Fixers in Atmospheric General Circulation Models
    ! near Eq. 13.50

    dxy=dx/dy
    dxz=dx/dz
    dyx=dy/dx
    dyz=dy/dz

    !-----------------------------------------
    if(dowallx) then

       if(mod(rank,nsubdomains_x).eq.0) then
          do k=1,nzm
             do j=1,ny
                v(0,j,k) = v(1,j,k)
                w(0,j,k) = w(1,j,k)
                t(0,j,k) = t(1,j,k)
             end do
          end do
       end if
       if(mod(rank,nsubdomains_x).eq.nsubdomains_x-1) then
          do k=1,nzm
             do j=1,ny
                v(nx+1,j,k) = v(nx,j,k)
                w(nx+1,j,k) = w(nx,j,k)
                t(nx+1,j,k) = t(nx,j,k)
             end do
          end do
       end if

    end if

    if(dowally) then

       if(rank.lt.nsubdomains_x) then
          do k=1,nzm
             do i=1,nx
                u(i,1-YES3D,k) = u(i,1,k)
                w(i,1-YES3D,k) = w(i,1,k)
                t(i,1-YES3D,k) = t(i,1,k)
             end do
          end do
       end if
       if(rank.gt.nsubdomains-nsubdomains_x-1) then
          do k=1,nzm
             do i=1,nx
                u(i,ny+YES3D,k) = u(i,ny,k)
                w(i,ny+YES3D,k) = w(i,ny,k)
                t(i,ny+YES3D,k) = t(i,ny,k)
             end do
          end do
       end if

    end if


    !  Add hyperdiffusive terms to the momentum, temperature and scalars

    do k=1,nzm
       do j = 1,ny
          do i=1,nx
             dudt(i,j,k,na) =  dudt(i,j,k,na) - khyp * (rdx16 *  &
                  (u(i-2, j, k) - 4*u(i-1,j,k) + 6*u(i,j,k) - 4*u(i+1,j,k) + u(i+2,j,k)) + &
                  rdy16* &
                  (u(i, j-2, k) - 4*u(i,j-1,k) + 6*u(i,j,k) - 4*u(i,j+1,k) + u(i,j+2,k)) )

             dvdt(i,j,k,na) =  dvdt(i,j,k,na) - khyp * (rdx16 *  &
                  (v(i-2, j, k) - 4*v(i-1,j,k) + 6*v(i,j,k) - 4*v(i+1,j,k) + v(i+2,j,k)) + &
                  rdy16* &
                  (v(i, j-2, k) - 4*v(i,j-1,k) + 6*v(i,j,k) - 4*v(i,j+1,k) + v(i,j+2,k)) )
          end do
       end do
    end do

    ! scalar
    call hyper_diffuse_scalar(micro_field(:,:,:,index_water_vapor))
    call hyper_diffuse_scalar(t)


    do k=2,nzm
       do j = 1,ny
          do i=1,nx
             dwdt(i,j,k,na) =  dwdt(i,j,k,na) - khyp * (rdx16 *  &
                  (w(i-2, j, k) - 4*w(i-1,j,k) + 6*w(i,j,k) - 4*w(i+1,j,k) + w(i+2,j,k)) + &
                  rdy16* &
                  (w(i, j-2, k) - 4*w(i,j-1,k) + 6*w(i,j,k) - 4*w(i,j+1,k) + w(i,j+2,k)) )
          end do
       end do
    end do



  end subroutine hyper_diffuse

  subroutine hyper_diffuse_scalar(t)
    use grid
    use params, only: docolumn, dowallx, dowally, khyp
    real, intent(inout) :: t(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm)

    ! local variables
    real tsrc, rdx16, rdy16, rdx2, rdy2, rdx25, rdy25

    rdx2=1./(dx*dx)
    rdy2=1./(dy*dy)

    rdx25=0.25*rdx2
    rdy25=0.25*rdy2

    rdx16 = rdx25*rdx25
    rdy16 = rdy25*rdy25

    if(dowallx) then

       if(mod(rank,nsubdomains_x).eq.0) then
          do k=1,nzm
             do j=1,ny
                t(0,j,k) = t(1,j,k)
             end do
          end do
       end if
       if(mod(rank,nsubdomains_x).eq.nsubdomains_x-1) then
          do k=1,nzm
             do j=1,ny
                t(nx+1,j,k) = t(nx,j,k)
             end do
          end do
       end if

    end if

    if(dowally) then

       if(rank.lt.nsubdomains_x) then
          do k=1,nzm
             do i=1,nx
                t(i,1-YES3D,k) = t(i,1,k)
             end do
          end do
       end if
       if(rank.gt.nsubdomains-nsubdomains_x-1) then
          do k=1,nzm
             do i=1,nx
                t(i,ny+YES3D,k) = t(i,ny,k)
             end do
          end do
       end if

    end if


    !  Add hyperdiffusive terms to the momentum, temperature and scalars
    do k=1,nzm
       do j = 1,ny
          do i=1,nx
             tsrc = - khyp * (rdx16 *  &
                  (t(i-2, j, k) - 4*t(i-1,j,k) + 6*t(i,j,k) - 4*t(i+1,j,k) + t(i+2,j,k)) + &
                  rdy16* &
                  (t(i, j-2, k) - 4*t(i,j-1,k) + 6*t(i,j,k) - 4*t(i,j+1,k) + t(i,j+2,k)) )
             t(i,j,k) = t(i,j,k) + tsrc * dtn
          end do
       end do
    end do

  end subroutine hyper_diffuse_scalar

end module hyperdiffusion
