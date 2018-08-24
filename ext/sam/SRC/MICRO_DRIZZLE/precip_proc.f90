  
subroutine precip_proc

use vars
use microphysics
use micro_params
use params

implicit none

integer i,j,k
real auto, accr, dq, qsat
real qcc
real df(dimx1_s:dimx2_s, dimy1_s:dimy2_s, nzm)
real f0(nzm),df0(nzm)
      
call t_startf ('precip_proc')
     
if(dostatis) then
        
  do k=1,nzm
    do j=dimy1_s,dimy2_s
     do i=dimx1_s,dimx2_s
      df(i,j,k) = q(i,j,k)
     end do
    end do
  end do
         
endif


 
do k=1,nzm
 qpsrc(k)=0.
 qpevp(k)=0.
 do j=1,ny
  do i=1,nx	  
	  
!-------     Autoconversion/accretion

   if(qn(i,j,k)+qp(i,j,k).gt.0.) then


         if(qn(i,j,k).gt.0.) then
 
           qcc = qn(i,j,k)

           auto = 1350.*qcc**1.47/Nc0**1.79   ! Linearized drizzle autoconversion
                                                !(Khairoutdinov and Kogan 2000)
           accr = 67.*qcc**0.15*qp(i,j,k)**1.15 ! Linearized accretion

           qcc = qcc/(1.+dtn*(auto+accr))
           auto = auto*qcc
           accr = accr*qcc
           dq = min(qn(i,j,k),dtn*(auto+accr))
           qp(i,j,k) = qp(i,j,k) + dq
           conp(i,j,k) = conp(i,j,k) + max(0.,dq-dtn*accr)*coefconpmax ! all new drizzle drops have rd_min size
           q(i,j,k) = q(i,j,k) - dq
           qn(i,j,k) = qn(i,j,k) - dq
           qpsrc(k) = qpsrc(k) + dq

         elseif(qp(i,j,k).gt.qp_threshold.and.qn(i,j,k).eq.0.) then

           qsat = qsatw(tabs(i,j,k),pres(k))
           dq = dtn * evapr1(k) * (qp(i,j,k)*conp(i,j,k)**2)**0.333 * (q(i,j,k) /qsat-1.) 
           dq = max(-0.5*qp(i,j,k),dq)
           conp(i,j,k) = conp(i,j,k) + dq/qp(i,j,k)*conp(i,j,k)
           qp(i,j,k) = qp(i,j,k) + dq
           q(i,j,k) = q(i,j,k) - dq
           qpevp(k) = qpevp(k) + dq

         else

           q(i,j,k) = q(i,j,k) + qp(i,j,k)
           qpevp(k) = qpevp(k) - qp(i,j,k)
           qp(i,j,k) = 0.
           conp(i,j,k) = 0.

         endif

    endif

    if(qp(i,j,k).lt.0.) then
      qp(i,j,k)=0.
      conp(i,j,k) = 0.
    end if

    conp(i,j,k) = max(qp(i,j,k)*coefconpmin, conp(i,j,k))  ! keep conp reasonable

  end do
 enddo
enddo
    


if(dostatis) then
                  
  call stat_varscalar(q,df,f0,df0,q2leprec)
  call setvalue(qwleprec,nzm,0.)
  call stat_sw2(q,df,qwleprec)

endif

call t_stopf ('precip_proc')

end subroutine precip_proc

