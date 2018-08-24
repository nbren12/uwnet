dz = 5.
z=0.
zi=795.

do while(z.lt.1600.)
  if(z.le.zi) then
    tp=288.3
    qt=9.45
  else
    tp=295+(z-zi)**0.333333
    qt=5.-3*(1.-exp((zi-z)/500.))
  endif
  u= 3.+4.3*z/1000.
  v=-9.+5.6*z/1000.
  write(*,'(6f9.3)') z,-999.,tp,qt,u,v
  z = z+5.
end do

end
