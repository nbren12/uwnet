subroutine stepout(nstatsteps)

use vars
use rad, only: qrad
use sgs, only: tk
use tracers
use microphysics, only: micro_print, micro_statistics
use sgs, only: sgs_print, sgs_statistics
use stat_moments, only: compmoments
use movies, only: mvmovies
use params
use hbuffer
use instrument_diagnostics, only : isccp_write, modis_write, misr_write, zero_instr_diag
implicit none	
	
integer i,j,k,ic,jc,nstatsteps
real div, divmax, divmin
real rdx, rdy, rdz, coef
integer im,jm,km
real wmax, qnmax(1), qnmax1(1)
real(8) buffer(5), buffer1(5)

if(mod(nstep,nstatis).eq.0) then
      call statistics()
      call micro_statistics()
      call sgs_statistics()
end if


if(mod(nstep,nmovie).eq.0.and.nstep.ge.nmoviestart &
                                   .and.nstep.le.nmovieend) then
      call mvmovies()
endif

if(mod(nstep,nstatmom).eq.0.and.nstep.ge.nstatmomstart &
                                   .and.nstep.le.nstatmomend) then
      call compmoments()
endif


if(mod(nstep,nstat).eq.0) then
  if(masterproc) print *,'Writting statistics:nstatsteps=',nstatsteps

  call t_startf ('stat_out')

  ! write out instrument simulator data and compute domain means for statistics
  !bloss(2016-02-06): This needs to be called before hbuf_avg for the simulator
  !  cloud fraction to be recorded at the first statistics output.
  call isccp_write()
  call modis_write()
  call misr_write()

  call hbuf_average(nstatsteps)
  call hbuf_write(nstatsteps)
  call hbuf_flush()	  
  nstatsteps = 0

  call zero_instr_diag()

  call t_stopf ('stat_out')

endif
if(mod(nstep,nstat*(1+nrestart_skip)).eq.0.or.nstep.eq.nstop.or.nelapse.eq.0) then

  call write_all() ! save restart file

end if



call t_startf ('2D_out')

if(mod(nstep,nsave2D).eq.0.and.nstep.ge.nsave2Dstart &
                                   .and.nstep.le.nsave2Dend) then
  call write_fields2D()
endif

if(.not.save2Davg.or.nstep.eq.nsave2Dstart-nsave2D) call stat_2Dinit(0) ! argument of 0 means storage terms for statistics are preserved

call t_stopf ('2D_out')

call t_startf ('3D_out')
if(mod(nstep,nsave3D).eq.0.and.nstep.ge.nsave3Dstart.and.nstep.le.nsave3Dend ) then
  ! determine if the maximum cloud water exceeds the threshold
  ! value to save 3D fields:
  qnmax(1)=0.
  do k=1,nzm
    do j=1,ny
      do i=1,nx
         qnmax(1) = max(qnmax(1),qcl(i,j,k))
         qnmax(1) = max(qnmax(1),qci(i,j,k))
      end do
    enddo
  enddo
  if(dompi) then
     call task_max_real(qnmax(1),qnmax1(1),1)
     qnmax(1) = qnmax1(1)
  end if
  if(qnmax(1).ge.qnsave3D) then 
    call write_fields3D()
  end if
endif
call t_stopf ('3D_out')

!------------------------------------------------------------------------------
!------------------------------------------------------------------------------
! Print stuff out:

call t_startf ('print_out')

if(masterproc) write(*,'(a,i10,a,i3,a,f6.3,a,f6.3)') 'NSTEP = ',nstep,'   NCYCLE=',ncycle, &
                  '  CFL_adv=',cfl_adv,'  CFL_sgs=',cfl_sgs

if(mod(nstep,nprint).eq.0) then
	

 divmin=1.e20
 divmax=-1.e20
	 
 rdx = 1./dx
 rdy = 1./dy

 wmax=0.
 do k=1,nzm
  coef = rho(k)*adz(k)*dz
  rdz = 1./coef
  if(ny.ne.1) then
   do j=1,ny-1*YES3D
    jc = j+1*YES3D
    do i=1,nx-1
     ic = i+1
     div = (u(ic,j,k)-u(i,j,k))*rdx + (v(i,jc,k)-v(i,j,k))*rdy + &
		  (w(i,j,k+1)*rhow(k+1)-w(i,j,k)*rhow(k))*rdz
     divmax = max(divmax,div)
     divmin = min(divmin,div)
     if(w(i,j,k).gt.wmax) then
	wmax=w(i,j,k)
	im=i
	jm=j
	km=k
     endif
    end do
   end do
  else
    j = 1
    do i=1,nx-1
    ic = i+1
     div = (u(ic,j,k)-u(i,j,k))*rdx +(w(i,j,k+1)*rhow(k+1)-w(i,j,k)*rhow(k))*rdz
     divmax = max(divmax,div)
     divmin = min(divmin,div)
     if(w(i,j,k).gt.wmax) then
	wmax=w(i,j,k)
	im=i
	jm=j
	km=k
     endif
    end do
  endif
 end do

 if(dompi) then
   buffer(1) = total_water_before
   buffer(2) = total_water_after
   buffer(3) = total_water_evap
   buffer(4) = total_water_prec
   buffer(5) = total_water_ls
   call task_sum_real8(buffer, buffer1,5)
   total_water_before = buffer1(1)
   total_water_after = buffer1(2)
   total_water_evap = buffer1(3)
   total_water_prec = buffer1(4)
   total_water_ls = buffer1(5)
 end if

!print*,rank,minval(u(1:nx,1:ny,:)),maxval(u(1:nx,1:ny,:))
!print*,rank,'min:',minloc(u(1:nx,1:ny,:))
!print*,rank,'max:',maxloc(u(1:nx,1:ny,:))

!if(masterproc) then

!print*,'--->',tk(27,1,1)
!print*,'tk->:'
!write(6,'(16f7.2)')((tk(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'p->:'
!write(6,'(16f7.2)')((p(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'u->:'
!write(6,'(16f7.2)')((u(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'v->:'
!write(6,'(16f7.2)')((v(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'w->:'
!write(6,'(16f7.2)')((w(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'qcl:'
!write(6,'(16f7.2)')((qcl(i,1,k)*1000.,i=16,31),k=nzm,1,-1)
!print*,'qpl->:'
!write(6,'(16f7.2)')((qpl(i,1,k)*1000.,i=16,31),k=nzm,1,-1)
!print*,'qcl:->'
!write(6,'(16f7.2)')((qci(i,1,k)*1000.,i=16,31),k=nzm,1,-1)
!print*,'qpl->:'
!write(6,'(16f7.2)')((qpi(i,1,k)*1000.,i=16,31),k=nzm,1,-1)
!print*,'qrad->:'
!write(6,'(16f7.2)')((qrad(i,1,k)*3600.,i=16,31),k=nzm,1,-1)
!print*,'qv->:'
!write(6,'(16f7.2)')((qv(i,1,k)*1000.,i=16,31),k=nzm,1,-1)
!print*,'t->:'
!write(6,'(16f7.2)')((t(i,1,k),i=1,16),k=nzm,1,-1)
!print*,'tabs->:'
!write(6,'(16f7.2)')((tabs(i,1,k),i=16,31),k=nzm,1,-1)

!end if

!--------------------------------------------------------
 if(masterproc) then
	
    print*,'DAY = ',day	
    write(6,*) 'NSTEP=',nstep
    write(6,*) 'div:',divmax,divmin
    if(.not.dodynamicocean) write(6,*) 'SST=',tabs_s 
    write(6,*) 'surface pressure=',pres0

 endif

 call fminmax_print('u:',u,dimx1_u,dimx2_u,dimy1_u,dimy2_u,nzm)
 call fminmax_print('v:',v,dimx1_v,dimx2_v,dimy1_v,dimy2_v,nzm)
 call fminmax_print('w:',w,dimx1_w,dimx2_w,dimy1_w,dimy2_w,nz)
 call fminmax_print('p:',p,0,nx,1-YES3D,ny,nzm)
 call fminmax_print('t:',t,dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm)
 call fminmax_print('tabs:',tabs,1,nx,1,ny,nzm)
 call fminmax_print('qv:',qv,1,nx,1,ny,nzm)
 if(dosgs) call sgs_print()
 if(docloud) then
   call fminmax_print('qcl:',qcl,1,nx,1,ny,nzm)
   call fminmax_print('qci:',qci,1,nx,1,ny,nzm)
   call micro_print()
 end if
 if(doprecip) then
   call fminmax_print('qpl:',qpl,1,nx,1,ny,nzm)
   call fminmax_print('qpi:',qpi,1,nx,1,ny,nzm)
 end if
 if(dolongwave.or.doshortwave) call fminmax_print('qrad(K/day):',qrad*86400.,1,nx,1,ny,nzm)
 if(dotracers) then
   do k=1,ntracers
     call fminmax_print(trim(tracername(k))//':',tracer(:,:,:,k),dimx1_s,dimx2_s,dimy1_s,dimy2_s,nzm)
   end do
 end if
 call fminmax_print('shf:',fluxbt*cp*rhow(1),1,nx,1,ny,1)
 call fminmax_print('lhf:',fluxbq*lcond*rhow(1),1,nx,1,ny,1)
 call fminmax_print('uw:',fluxbu,1,nx,1,ny,1)
 call fminmax_print('vw:',fluxbv,1,nx,1,ny,1)
 call fminmax_print('sst:',sstxy+t00,0,nx,1-YES3D,ny,1)

 total_water_before = total_water_before/float(nx_gl*ny_gl)
 total_water_after = total_water_after/float(nx_gl*ny_gl)
 total_water_evap = total_water_evap/float(nx_gl*ny_gl)
 total_water_prec = total_water_prec/float(nx_gl*ny_gl)
 total_water_ls = total_water_ls/float(nx_gl*ny_gl)
 
 if(masterproc) then
   
   print*,'total water budget:'
   write(*,991) total_water_before !'before (mm):    ',total_water_before
   write(*,992) total_water_after !'after (mm) :    ',total_water_after
   write(*,993) total_water_evap !'evap (mm/day):  ',total_water_evap
   write(*,994) total_water_prec !'prec (mm/day):  ',total_water_prec
   write(*,995) total_water_ls !'ls (mm/day):    ',total_water_ls
   write(*,996) (total_water_after-(total_water_before+total_water_evap+total_water_ls-total_water_prec))
   991 format(' before (mm):       ',F16.11)
   992 format(' after (mm):        ',F16.11)
   993 format(' evaporation (mm):  ',F16.11)
   994 format(' precipitation (mm):',F16.11)
   995 format(' large-scale (mm):  ',F16.11)
   996 format(' Imbalance (mm)    ',F16.11)
   print*,' imbalance (rel error):', &
     (total_water_after-(total_water_before+total_water_evap+total_water_ls-total_water_prec))/total_water_after
   print*,'evap (mm/day):',total_water_evap/dt*86400.
   print*,'prec (mm/day):',total_water_prec/dt*86400.
   print*,'ls (mm/day):',total_water_ls/dt*86400.
   print*,'imbalance (mm/day)', &
     (total_water_after-total_water_before-total_water_evap-total_water_ls+total_water_prec)/dt*86400.

 end if



end if ! (mod(nstep,nprint).eq.0)

call t_stopf ('print_out')

end
