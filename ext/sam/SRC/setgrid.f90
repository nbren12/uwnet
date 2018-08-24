subroutine setgrid

! Initialize vertical grid

use vars	
use params

implicit none
	
real latit, long
integer i,j,it,jt,k, kmax

if(nrestart.eq.0) then

 
 if(doconstdz) then

    z(1) = 0.5*dz
    do k=2,nz
     z(k)=z(k-1)+dz
    end do

 else

    open(8,file='./'//trim(case)//'/grd',status='old',form='formatted') 
    do k=1,nz     
      read(8,fmt=*,end=111) z(k)
      kmax=k
    end do
    goto 222
111 do k=kmax+1,nz
     z(k)=z(k-1)+(z(k-1)-z(k-2))
    end do
222 continue
    close (8)

 end if 

end if
 	
if(.not.doconstdz) dz = 0.5*(z(1)+z(2))

do k=2,nzm
   adzw(k) = (z(k)-z(k-1))/dz
end do
adzw(1) = 1.
adzw(nz) = adzw(nzm)
adz(1) = 1.
do k=2,nzm-1
   adz(k) = 0.5*(z(k+1)-z(k-1))/dz
end do
adz(nzm) = adzw(nzm)
zi(1) = 0.
do k=2,nz
   zi(k) = zi(k-1) + adz(k-1)*dz
end do

do k=1,nzm
  gamaz(k)=ggr/cp*z(k)
end do

if(dofplane) then

  if(fcor.eq.-999.) fcor = 4*pi/86400.*sin(latitude0*pi/180.)
  fcorz = 0.
  if(docoriolisz) fcorz =  sqrt(4.*(2*pi/(3600.*24.))**2-fcor**2)
  fcory(:) = fcor
  fcorzy(:) = fcorz
  longitude(:,:) = longitude0

else

 call task_rank_to_index(rank,it,jt)

 do j=0,ny
     latit=latitude0+dy*(j+jt-(ny_gl+YES3D-1)/2-1)*2.5e-8*360.
     fcory(j)= 4.*pi/86400.*sin(latit*pi/180.)
     fcorzy(j) = 0.
     if(j.ne.0.and.docoriolisz) fcorzy(j) = sqrt(4.*(2*pi/(3600.*24.))**2-fcory(j)**2)
 end do

end if ! dofplane

if (doradlat) then
 call task_rank_to_index(rank,it,jt)
 do j=1,ny
     latitude(:,j) = latitude0+dy*(j+jt-(ny_gl+YES3D-1)/2-1)*2.5e-8*360.
  end do
else
  latitude(:,:) = latitude0
end if

if (doradlon) then
 call task_rank_to_index(rank,it,jt)
 do i=1,nx
     longitude(i,:) = longitude0+dx/cos(latitude0*pi/180.)* &
                            (i+it-nx_gl/2-1)*2.5e-8*360.
  end do
else
  longitude(:,:) = longitude0
end if


end
