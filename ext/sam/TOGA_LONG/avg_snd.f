integer, parameter :: nz = 40
real(8) z1(nz),p1(nz),t1(nz),q1(nz),u1(nz),v1(nz)
real(8) z(nz),p(nz),t(nz),q(nz),u(nz),v(nz)
real(8) pres01, pres0
time = 0.
nt = 0
z=0.
p=0.
u=0.
v=0.
w=0.
t=0.
q=0.
pres0=0.
time = 0.
read*
do while(.true.)
  read(unit=5,fmt=*,end=111) time1,nn,pres01
  pres0 = pres0+pres01
  time = time + time1
  do k=1,nz
   read*,z1(k),p1(k),t1(k),q1(k),u1(k),v1(k)
   z(k) = z(k) + z1(k)
   p(k) = p(k) + p1(k)
   t(k) = t(k) + t1(k)
   q(k) = q(k) + q1(k)
   u(k) = u(k) + u1(k)
   v(k) = v(k) + v1(k)
  end do
  nt=nt+1
end do
111 pres0=pres0/nt
time=time/nt
z = z/nt
p = p/nt
t = t/nt
q = q/nt
u = u/nt
v = v/nt
w = w/nt
print*,time,nn,pres0
do k=1,nz
   write(*,'(7g14.5)')z(k),p(k),t(k),q(k),u(k),v(k)
end do

end


