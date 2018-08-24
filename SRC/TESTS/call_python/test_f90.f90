program call_python
  use, intrinsic :: iso_c_binding
  implicit none
  interface
     subroutine hello_world() bind (c)
     end subroutine hello_world
  end interface

  integer i

  print *, "Calling python"
  do i=1,10
     print *, i
     call hello_world()
  end do

end program call_python
