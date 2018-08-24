! Wrapper call to avoid circular dependency between hbuffer and microphysics modules

subroutine hbuf_micro_init(namelist,deflist,unitlist,status,average_type,count,trcount)
   use microphysics
   implicit none
   character(*) namelist(*), deflist(*), unitlist(*)
   integer status(*),average_type(*),count,trcount

   call micro_hbuf_init(namelist,deflist,unitlist,status,average_type,count,trcount)

end
