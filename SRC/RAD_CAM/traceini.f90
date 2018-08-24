	subroutine tracesini()


! Initialize trace gaz vertical profiles


        use grid
        use rad
        use params, only: notracegases

	implicit none

	integer nzls,nzls0
	parameter (nzls0=500)
	real zls(nzls0)
	real pls(nzls0)
	real o3ls(nzls0)
	real n2ols(nzls0)
	real ch4ls(nzls0)
	real cfc11ls(nzls0)
	real cfc12ls(nzls0)

	real zz(nzm)
	integer k,m,n

	do k=1,nzm
	  zz(nz-k)=z(k)
	end do

	open(88,file=trim(rundatadir)//'/trc', status='old',form='formatted')

	read(88,*) nzls
	read(88,*)
	do k=1,nzls
         read(88,*) zls(k),pls(k),o3ls(k),n2ols(k), &
					ch4ls(k),cfc11ls(k),cfc12ls(k)
	 zls(k)=zls(k)*1000.
	end do
	close (88)
        if(notracegases) then
           o3ls = o3ls*0.01
           n2ols = 0.
           ch4ls = 0.
	   cfc11ls = 0.
           cfc12ls = 0.
        end if
        n=1
	do k=1,nzm
	  if(zz(k).ge.zls(1)) then
           o3(k)=o3ls(1)
           n2o(k)=n2ols(1)
           ch4(k)=ch4ls(1)
           cfc11(k)=cfc11ls(1)
           cfc12(k)=cfc11ls(1)
	  elseif (zz(k).le.zls(nzls)) then
           o3(k)=o3ls(nzls)
           n2o(k)=n2ols(nzls)
           ch4(k)=ch4ls(nzls)
           cfc11(k)=cfc11ls(nzls)
           cfc12(k)=cfc11ls(nzls)
	  else 
	   do m=n,nzls-1
            if(zz(k).gt.zls(m)) then
              n=m-1
	      goto 111
	    endif
	   end do
 111	   continue
	   o3(k)=o3ls(n)+(zz(k)-zls(n))/(zls(n+1)-zls(n))* &
						(o3ls(n+1)-o3ls(n))
	   n2o(k)=n2ols(n)+(zz(k)-zls(n))/(zls(n+1)-zls(n))* &
						(n2ols(n+1)-n2ols(n))
	   ch4(k)=ch4ls(n)+(zz(k)-zls(n))/(zls(n+1)-zls(n))* &
						(ch4ls(n+1)-ch4ls(n))
	   cfc11(k)=cfc11ls(n)+(zz(k)-zls(n))/(zls(n+1)-zls(n))* &
					(cfc11ls(n+1)-cfc11ls(n))
	   cfc12(k)=cfc12ls(n)+(zz(k)-zls(n))/(zls(n+1)-zls(n))* &
					(cfc12ls(n+1)-cfc12ls(n))
	  endif
	end do
	if(rank.eq.0) then

	print*,'CEM.trace: number of levels=',nzls
	print*,'gas traces vertical profiles (g/g):'
	print*,'  z    o3   n2o   ch4   cfc11   cfc12'
	do k=nzm,1,-1
	 write(6,'(6g12.4)') zz(k),o3(k),n2o(k),ch4(k),cfc11(k),cfc12(k)
	end do
	print*,'done...'

	endif
	
	return
	end
