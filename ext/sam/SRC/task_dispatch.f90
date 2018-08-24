
subroutine task_dispatch(buff,tag)
	
! dispathes the messages according to the field sent.

use vars
use microphysics
use sgs
use tracers
implicit none
	
real buff(*)	! buff for sending data
integer tag	! tag of the message
integer field   ! id of field
	
field = tag/100000
	
if(field.eq.1) then        

  call task_assign_bnd(u,dimx1_u,dimx2_u,dimy1_u,dimy2_u,nzm,buff,tag)

elseif(field.eq.2) then        

  call task_assign_bnd(v,dimx1_v,dimx2_v,dimy1_v,dimy2_v,nzm,buff,tag)

elseif(field.eq.3) then        

  call task_assign_bnd(w,dimx1_w,dimx2_w,dimy1_w,dimy2_w,nz,buff,tag)	

elseif(field.eq.4) then        

  call task_assign_bnd(t,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,buff,tag)

elseif(field.gt.4.and.field.le.4+nsgs_fields) then

  call task_assign_bnd(sgs_field(:,:,:,field-4),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,buff,tag)	 

elseif(field.gt.4+nsgs_fields.and.field.le.4+nsgs_fields+nsgs_fields_diag) then

  call task_assign_bnd(sgs_field_diag(:,:,:,field-4-nsgs_fields), &
                                                dimx1_d,dimx2_d,dimy1_d,dimy2_d,nzm,buff,tag)	 

elseif(field.gt.4+nsgs_fields+nsgs_fields_diag.and.field.le.4+nsgs_fields+nsgs_fields_diag+nmicro_fields) then

  call task_assign_bnd(micro_field(:,:,:,field-4-nsgs_fields-nsgs_fields_diag), &
                                                dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,buff,tag)	 

elseif(field.gt.4+nsgs_fields+nsgs_fields_diag+nmicro_fields.and. &
                        field.le.4+nsgs_fields+nsgs_fields_diag+nmicro_fields+ntracers) then

  call task_assign_bnd(tracer(:,:,:,field-4-nsgs_fields-nsgs_fields_diag-nmicro_fields),&
                                                dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm,buff,tag)	 

end if
	
end
	     
	     
