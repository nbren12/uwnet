dz = 5.
z=0.
zi=795.
div = 3.75e-6

do while(z.lt.1600.)
  u= 3.+4.3*z/1000.
  v=-9.+5.6*z/1000.
  write(*,'(7f11.5)') z,-999.,0.,0.,u,v,-div*z
  z = z+5.
end do

end
