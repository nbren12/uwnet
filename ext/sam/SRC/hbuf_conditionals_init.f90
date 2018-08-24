subroutine hbuf_conditionals_init(count,trcount)
  use vars, only: ncondavg, condavgname, condavglongname, &
       dowtg_blossey_etal_JAMES2009
  use rad, only: do_output_clearsky_heating_profiles
  implicit none

  ! Initialize the list of UW statistics variables written in statistics.f90
  integer count,trcount, n

  if(do_output_clearsky_heating_profiles) then
    call add_to_namelist(count,trcount,'RADQRCLW', &
         'Clearsky longwave heating rate','K/d',0)
    call add_to_namelist(count,trcount,'RADQRCSW', &
         'Clearsky shortwave heating rate','K/d',0)
  end if

  if(dowtg_blossey_etal_JAMES2009) then
    call add_to_namelist(count,trcount,'WWTG', &
         'Large-scale W induced by weak temperature gradient approx','m/s',0)
  end if

  if(dowtg_blossey_etal_JAMES2009) then
    call add_to_namelist(count,trcount,'WOBSREF', &
         'Reference Large-scale W Before Modifications by WTG/Scaling','m/s',0)
  end if

  !bloss: setup to add an arbitrary number of conditional statistics
  do n = 1,ncondavg

     !bloss: add all of the conditional statistics here, so that they don't
     !  have to be added to the lst file
     call add_to_namelist(count,trcount,TRIM(condavgname(n)), &
          TRIM(condavglongname(n))//' Fraction',' ',0)
     call add_to_namelist(count,trcount,'W'//TRIM(condavgname(n)), &
          'Mean W in '//TRIM(condavglongname(n)),'m/s',n)
     call add_to_namelist(count,trcount,'U'//TRIM(condavgname(n)), &
          'Mean U in '//TRIM(condavglongname(n)),'m/s',n)
     call add_to_namelist(count,trcount,'V'//TRIM(condavgname(n)), &
          'Mean V in '//TRIM(condavglongname(n)),'m/s',n)
     call add_to_namelist(count,trcount,'MSE'//TRIM(condavgname(n)), &
          'Mean moist static energy in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'DSE'//TRIM(condavgname(n)), &
          'Mean dry static energy in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'TL'//TRIM(condavgname(n)), &
          'Mean liquid-ice static energy in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'TA'//TRIM(condavgname(n)), &
          'Mean TABS in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'TV'//TRIM(condavgname(n)), &
          'Mean THETAV in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'TV'//TRIM(condavgname(n))//'A', &
          'Mean THETAV anomaly in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'QT'//TRIM(condavgname(n)), &
          'Mean QT in '//TRIM(condavglongname(n)),'g/kg',n)
     call add_to_namelist(count,trcount,'QN'//TRIM(condavgname(n)), &
          'Mean QN in '//TRIM(condavglongname(n)),'g/kg',n)
     !bloss: these conditional averages are now computed inside the microphysics
     !         routines.
     !bloss        call add_to_namelist(count,trcount,'QC'//TRIM(condavgname(n)), &
     !bloss             'Mean QC in '//TRIM(condavglongname(n)),'g/kg',n)
     !bloss        call add_to_namelist(count,trcount,'QI'//TRIM(condavgname(n)), &
     !bloss             'Mean QI in '//TRIM(condavglongname(n)),'g/kg',n)
     call add_to_namelist(count,trcount,'QP'//TRIM(condavgname(n)), &
          'Mean QP in '//TRIM(condavglongname(n)),'g/kg',n)
     call add_to_namelist(count,trcount,'S'//TRIM(condavgname(n)), &
          'Mean scalar in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'W'//TRIM(condavgname(n))//'A', &
          'W in '//TRIM(condavglongname(n))//' averaged over the whole domain','m/s',0)
     call add_to_namelist(count,trcount,'TLW'//TRIM(condavgname(n)), &
          'TLW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'Km/s',0)
     call add_to_namelist(count,trcount,'TVW'//TRIM(condavgname(n)), &
          'TVW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'Km/s',0)
     call add_to_namelist(count,trcount,'SW'//TRIM(condavgname(n)), &
          'SW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'Km/s',0)
     call add_to_namelist(count,trcount,'QTW'//TRIM(condavgname(n)), &
          'QTW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'g/kg m/s',0)
     call add_to_namelist(count,trcount,'QCW'//TRIM(condavgname(n)), &
          'QCW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'g/kg m/s',0)
     call add_to_namelist(count,trcount,'QIW'//TRIM(condavgname(n)), &
          'QIW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'g/kg m/s',0)

     !bloss: frozen moist static energy statistics
     call add_to_namelist(count,trcount,'HF'//TRIM(condavgname(n)), &
          'Mean Frozen MSE in '//TRIM(condavglongname(n)),'K',n)
     call add_to_namelist(count,trcount,'HF'//TRIM(condavgname(n))//'A', &
          'Mean Frozen MSE anomaly in '//TRIM(condavglongname(n)),'K',n)

     !bloss: velocity anomalies
     call add_to_namelist(count,trcount,'U'//TRIM(condavgname(n))//'A', &
          'Mean U anomaly in '//TRIM(condavglongname(n)),'m/s',n)
     call add_to_namelist(count,trcount,'V'//TRIM(condavgname(n))//'A', &
          'Mean V anomaly in '//TRIM(condavglongname(n)),'m/s',n)

     !bloss: pressure gradients
     call add_to_namelist(count,trcount,'UPGF'//TRIM(condavgname(n)), &
          'Zonal pressure gradient in '//TRIM(condavglongname(n)),'m/s2',n)
     call add_to_namelist(count,trcount,'VPGF'//TRIM(condavgname(n)), &
          'Meridional pressure gradient in '//TRIM(condavglongname(n)),'m/s2',n)
     call add_to_namelist(count,trcount,'WPGF'//TRIM(condavgname(n)), &
          'Vertical pressure gradient in '//TRIM(condavglongname(n)),'m/s2',n)

     !bloss: momentum statistics
     call add_to_namelist(count,trcount,'UW'//TRIM(condavgname(n)), &
          'UW in '//TRIM(condavglongname(n)),'m2/s2',n)
     call add_to_namelist(count,trcount,'VW'//TRIM(condavgname(n)), &
          'VW in '//TRIM(condavglongname(n)),'m2/s2',n)
     call add_to_namelist(count,trcount,'UWSB'//TRIM(condavgname(n)), &
          'Subgrid UW in '//TRIM(condavglongname(n)),'m2/s2',n)
     call add_to_namelist(count,trcount,'VWSB'//TRIM(condavgname(n)), &
          'Subgrid VW in '//TRIM(condavglongname(n)),'m2/s2',n)

     !bloss: UW-added mass flux weighted statistics
     call add_to_namelist(count,trcount,'MF'//TRIM(condavgname(n)), &
          'Mass flux in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFH'//TRIM(condavgname(n)), &
          'RHO*W*HF in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFH'//TRIM(condavgname(n))//'A', &
          'RHO*W*HF anomaly in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFTL'//TRIM(condavgname(n)), &
          'RHO*W*TL in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFTL'//TRIM(condavgname(n))//'A', &
          'RHO*W*TL anomaly in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFTV'//TRIM(condavgname(n)), &
          'RHO*W*TV in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFTV'//TRIM(condavgname(n))//'A', &
          'RHO*W*TV anomaly in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'K kg/m2/s',0)
     call add_to_namelist(count,trcount,'MFQT'//TRIM(condavgname(n)), &
          'RHO*W*QT in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'g/m2/s',0)
     call add_to_namelist(count,trcount,'MFQT'//TRIM(condavgname(n))//'A', &
          'RHO*W*QT anomaly in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'g/m2/s',0)
     call add_to_namelist(count,trcount,'RUW'//TRIM(condavgname(n)), &
          'RHOUW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'kg/m/s2',0)
     call add_to_namelist(count,trcount,'RVW'//TRIM(condavgname(n)), &
          'RHOVW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'kg/m/s2',0)
     call add_to_namelist(count,trcount,'RWW'//TRIM(condavgname(n)), &
          'RHOWW in '//TRIM(condavglongname(n))//' averaged over the whole domain', &
          'kg/m/s2',0)
  end do ! n = 1,ncondavg

end

subroutine add_to_namelist(count,trcount,varname,varlongname,varunits,varavg)
  use hbuffer, only: namelist,deflist,unitlist,status,average_type
  implicit none

  ! add variable to namelist
  integer count, trcount, ntr, n, varstatus, varavg
  character(*) varname
  character(*) varlongname
  character(*) varunits

  count = count + 1
  trcount = trcount + 1
  namelist(count) = trim(varname)
  deflist(count) = trim(varlongname)
  unitlist(count) = trim(varunits)
  status(count) = 1
  average_type(count) = varavg

end subroutine add_to_namelist
