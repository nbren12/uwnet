
! Monin-Obukhov Similarity
! Coded by Marat Khairoutdinov (C) 2003


subroutine landflx(p0, th, ts, qh, qs, uh, vh, h, z0, shf, lhf, taux, tauy, xlmo)

use params, only: epsv
implicit none

! Input:

real p0   ! surface pressure, mb
real th   ! pot. temperature at height h
real ts   ! pot. Temperature at z0
real qh   ! vapor at height h
real qs   ! vapor at z0
real uh   ! zonal wind at height h
real vh   ! merid wind at height h
real h    ! height h
real z0   ! friction height
        
! Output:

real shf   ! sensible heat flux (K m/s)
real lhf   ! latent heat flux (m/s)
real taux  ! zonal surface stress (N/m2)
real tauy  ! merid surface stress (N/m2)
real xlmo  ! Monin-Obukhov length

real r, x, pii, zody, vel
real a, b, c, d, ustar, tstar
real xm, xh, xsi, xsi1, xsi2, dxsi, fm, fh
integer iter
real gm1, gh1, fm1, fh1

gm1(x)=(1.-15.*x)**0.25
gh1(x)=sqrt(1.-9.*x)/0.74
fm1(x)=2.*alog((1.+x)/2.)+alog((1.+x*x)/2.)-2.*atan(x)+pii
fh1(x)=2.*alog((1.+0.74*x)/2.)

pii=acos(-1.)/2.
zody=alog(h/z0)

vel = sqrt(max(0.5,uh**2+vh**2))
r=min(0.25,9.81/ts*(th*(1+epsv*qh)-ts*(1.+epsv*qs))*h/vel**2)
iter=0

 
if(r.lt.0.) then 

        xsi=0.
	iter=iter+1
 	xm=gm1(xsi)
	xh=gh1(xsi)
	fm=zody-fm1(xm)
	fh=0.74*(zody-fh1(xh))
	xsi1=r/fh*fm**2
	dxsi=xsi-xsi1
	xsi=xsi1

        xsi=-abs(xsi)
	iter=iter+1
 	xm=gm1(xsi)
	xh=gh1(xsi)
	fm=zody-fm1(xm)
	fh=0.74*(zody-fh1(xh))
	xsi1=r/fh*fm**2
	dxsi=xsi-xsi1
	xsi=xsi1

        xsi=-abs(xsi)
	iter=iter+1
 	xm=gm1(xsi)
	xh=gh1(xsi)
	fm=zody-fm1(xm)
	fh=0.74*(zody-fh1(xh))
	xsi1=r/fh*fm**2
	dxsi=xsi-xsi1
	xsi=xsi1

else
  	a=4.8*4.8*r-1.00*6.35
	b=(2.*r*4.8-1.00)*zody
	c=r*zody**2
	d=sqrt(b*b-4*a*c)
	xsi1=(-b+d)/a/2.
	xsi2=(-b-d)/a/2.
	xsi=amax1(xsi1,xsi2)
	fm=zody+4.8*xsi
	fh=1.00*(zody+7.8*xsi)
!  	a=4.7*4.7*r-0.74*6.35
!	b=(2.*r*4.7-0.74)*zody
!	c=r*zody**2
!	d=sqrt(b*b-4*a*c)
!	xsi1=(-b+d)/a/2.
!	xsi2=(-b-d)/a/2.
!	xsi=amax1(xsi1,xsi2)
!	fm=zody+4.7*xsi
!	fh=0.74*(zody+6.35*xsi)
end if

 
shf=0.4**2/fm/fh*vel*(ts-th)
lhf=0.4**2/fm/fh*vel*(qs-qh)
taux=-0.4**2/fm/fm*vel*uh*(p0*100./287./ts)
tauy=-0.4**2/fm/fm*vel*vh*(p0*100./287./ts)
      
ustar = 0.4/fm*vel
tstar = 0.4/fh*(th-ts)
if(xsi.ge.0.) then
   xsi = max(1.e-5,xsi)
else
   xsi = min(-1.e-5,xsi)
end if
xlmo = h/xsi

return
end
