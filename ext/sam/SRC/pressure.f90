subroutine pressure

! call a pressure solver

use grid
implicit none

call t_startf ('pressure')

if(RUN3D) then
 if(mod(nx_gl,nsubdomains).ne.0.or.mod(ny_gl,nsubdomains).ne.0) then
  call pressure_orig
 else
  call pressure_big
 end if
else
  call pressure_orig
end if

call t_stopf ('pressure')

end
