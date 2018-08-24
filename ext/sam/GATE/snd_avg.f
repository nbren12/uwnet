! program to average sounding file.
! to use :  a.out < snd > snd_avg
character(80) header
real, dimension(100) :: z,p,t,q,u,v
read(*,'(a80)') header 
p0=0.
z=0.
p=0.
t=0.
q=0.
u=0.
v=0.
m=0
do 
  read(*,fmt=*,end=111) day,n,p01
  p0=p0+p01
  do k=1,n
   read*,z1,p1,t1,q1,u1,v1 
   z(k) = z(k) + z1
   p(k) = p(k) + p1
   t(k) = t(k) + t1
   q(k) = q(k) + q1
   u(k) = u(k) + u1
   v(k) = v(k) + v1
  end do
  m=m+1
end do
111 continue
z=z/m
p=p/m
t=t/m
q=q/m
u=u/m
v=v/m
p0=p0/m
print*, header
write(*,'(f6.0,i5,f10.3)') 0.,n,p0
write(*,'(6f10.4)') (z(k),p(k),t(k),q(k),u(k),v(k),k=1,n)
write(*,'(f6.0,i5,f10.3)') 1000.,n,p0
write(*,'(6f10.4)') (z(k),p(k),t(k),q(k),u(k),v(k),k=1,n)

end


