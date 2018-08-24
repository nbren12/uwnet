        z1=4000.
        z2=7763.
        z3=18000.
        z4=22000.
        z5=27000.
        dz1=50.
        dz2=1200.
        dz3=500.
        dz4=300.
        z=0.5*dz1
        dz=dz1
        n=1
        do while(z.lt.30000.)
         print*,z,n,dz
         dz=z
         if(z.gt.z1.and.z.le.z2) then
          z=z+dz1+(dz2-dz1)*(z-z1)/(z2-z1)
         else if(z.gt.z2.and.z.le.z3) then
          z=z+dz2
         else if(z.gt.z3.and.z.le.z4) then
          z=z+dz2+(dz3-dz2)*(z-z3)/(z4-z3)
         else if(z.gt.z4.and.z.le.z5) then
          z=z+dz3+(dz4-dz3)*(z-z4)/(z5-z4)
         else if(z.gt.z5) then
          z=z+dz4
         else
          z=z+dz1
         end if
         dz=z-dz
         n=n+1
        end do
        end

