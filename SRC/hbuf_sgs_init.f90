! Wrapper call to avoid circular dependency between hbuffer and SGS modules

subroutine hbuf_sgs_init(namelist,deflist,unitlist,status,average_type,count,trcount)
  use sgs 
  implicit none
  character(*) namelist(*), deflist(*), unitlist(*)
  integer status(*), average_type(*), count, trcount

  call sgs_hbuf_init(namelist,deflist,unitlist,status,average_type,count,trcount)

end
