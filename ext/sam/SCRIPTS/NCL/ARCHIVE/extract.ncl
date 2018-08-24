begin

f=addfile("ARM9707_512x64_new_X_rad.nc","r")
qr = f->RADQR
p = f->p
nz = dimsizes(p)
print("nz="+nz)
nt = dimsizes(qr(:,0))
print("nt="+nt)
time = f->time

do i=0,nt-1
 print(time(i)+",  "+nz)
 do j = 0,nz-1
   print(p(j)+"     "+(qr(i,j)/86400.))
 end do
end do

end
