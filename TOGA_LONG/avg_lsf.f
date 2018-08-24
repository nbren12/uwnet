integer, parameter :: nz = 40
real z1(nz),p1(nz),dtdt1(nz),dqdt1(nz),u1(nz),v1(nz),w1(nz)
real z(nz),p(nz),dtdt(nz),dqdt(nz),u(nz),v(nz),w(nz)
time = 0.
nt = 0
z=0.
p=0.
u=0.
v=0.
w=0.
dtdt=0.
dqdt=0.
pres0=0.
read*
do while(.true.)
  read(unit=5,fmt=*,end=111) time1,nn,pres01
  pres0 = pres0+pres01
  time = time + time1
  do k=1,nz
   read*,z1(k),p1(k),dtdt1(k),dqdt1(k),u1(k),v1(k),w1(k)
   z(k) = z(k) + z1(k)
   p(k) = p(k) + p1(k)
   dtdt(k) = dtdt(k) + dtdt1(k)
   dqdt(k) = dqdt(k) + dqdt1(k)
   u(k) = u(k) + u1(k)
   v(k) = v(k) + v1(k)
   w(k) = w(k) + w1(k)
  end do
  nt=nt+1
end do
111 continue
pres0=pres0/nt
time=time/nt
z = z/nt
p = p/nt
dtdt = dtdt/nt
dqdt = dqdt/nt
u = u/nt
v = v/nt
w = w/nt
print*,time,nn,pres0
do k=1,nz
!   write(*,'(7g14.5)')z(k),p(k),dtdt(k),dqdt(k),0.,0.,w(k)
   write(*,'(7g14.5)')z(k),p(k),dtdt(k),dqdt(k),u(k),v(k),w(k)
end do

end

