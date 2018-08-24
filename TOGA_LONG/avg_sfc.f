real(8) sst,sst0,LH,LH0,SH,SH0
sst0=0.
LH0=0.
SH0=0.
n=0
read(5,*)
do while(.true.)
  read(5,fmt=*,end=111) a,sst,SH,LH
  sst0=sst0+sst
  LH0 = LH0+LH
  SH0 = SH0+SH
  n = n+1
end do
111 sst0=sst0/dble(n)
LH0=LH0/dble(n)
SH0=SH0/dble(n)
print*,sst0,SH0,LH0,0.
end
