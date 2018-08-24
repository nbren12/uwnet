
subroutine buoyancy()

use vars
use params
implicit none
	
integer i,j,k,kb
real betu, betd, du(nx,ny,nz,3)

if(docolumn) return

call t_startf ('buoyancy')

if(dostatis) then

  do k=1,nzm
    do j=1,ny
      do i=1,nx
         du(i,j,k,3)=dwdt(i,j,k,na)
      end do
    end do
  end do

endif

do k=2,nzm	
 kb=k-1
 betu=adz(kb)/(adz(k)+adz(kb))
 betd=adz(k)/(adz(k)+adz(kb))
 do j=1,ny
  do i=1,nx

   dwdt(i,j,k,na)=dwdt(i,j,k,na) +  &
      bet(k)*betu* &
     ( tabs0(k)*(epsv*(qv(i,j,k)-qv0(k))-(qcl(i,j,k)+qci(i,j,k)-qn0(k)+qpl(i,j,k)+qpi(i,j,k)-qp0(k))) &
       +(tabs(i,j,k)-tabs0(k))*(1.+epsv*qv0(k)-qn0(k)-qp0(k)) ) &
    + bet(kb)*betd* &
     ( tabs0(kb)*(epsv*(qv(i,j,kb)-qv0(kb))-(qcl(i,j,kb)+qci(i,j,kb)-qn0(kb)+qpl(i,j,kb)+qpi(i,j,kb)-qp0(kb))) &
       +(tabs(i,j,kb)-tabs0(kb))*(1.+epsv*qv0(kb)-qn0(kb)-qp0(kb)) )  

  end do ! i
 end do ! j
end do ! k

if(dostatis) then

  do k=1,nzm
    do j=1,ny
      do i=1,nx
        du(i,j,k,1)=0.
        du(i,j,k,2)=0.
        du(i,j,k,3)=dwdt(i,j,k,na)-du(i,j,k,3)
      end do
    end do
  end do

  call stat_tke(du,tkelebuoy)
  call stat_mom(du,momlebuoy)
  call setvalue(twlebuoy,nzm,0.)
  call setvalue(qwlebuoy,nzm,0.)
  call stat_sw1(du,twlebuoy,qwlebuoy)

endif

call t_stopf ('buoyancy')
 
end subroutine buoyancy


