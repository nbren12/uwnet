; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of files:
 nfiles = 3

 nrunning =10 

 filename = new((/nfiles/),string)
 Case = new((/nfiles/),string)

 ; set file names:
; filename(0) = "./../../OUT_STAT/RCEOBS_32x28sp_4km_20s_SAMCCN_C100.nc"
; filename(1) = "./../../OUT_STAT/RCEOBS_32x28sp_4km_20s_SAMCCN_C100_KK.nc"
; filename(2) = "./../../OUT_STAT/RCEOBS_32x28sp_4km_20s_SAMCCN_C1000.nc"
; filename(3) = "./../../OUT_STAT/RCEOBS_32x28sp_4km_20s_SAMCCN_C1000_KK.nc"
; ; Case names:
; Case(0) = "CCN100"
; Case(1) = "CCN100KK"
; Case(2) = "CCN1000"
; Case(3) = "CCN1000KK"

 filename(0) = "./../../OUT_STAT/RCE_128x128x64MK_3km_12s_300K_newbuo.nc"
 filename(1) = "./../../OUT_STAT/RCE_128x128x64MK_3km_12s_300K_oldbuo_damp.nc"
 filename(2) = "./../../OUT_STAT/RCE_128x128x64MK_3km_12s_303K_oldbuo_damp.nc"


 Case(0) = "300K newbuo"
 Case(1) = "300K oldbuo damp"
 Case(2) = "303K oldbuo damp"


 ; Set time interval (in hours) for time averaged vertical profile plots:

; time_start = 365.
; time_stop = 405.
time_start = 90.
time_stop = 100.

 ; define threshold of total water (in g/m3) to defone PBL top:
 qt_threshold = 6.5

 ; Define maximum height (in m) plotted:

 z_max = 30000.

 ; Time evolution plots: Set min and max values:
 field_names = (/"SST","CLDSHD","ISCCPHGH","ISCCPMID","ISCCPLOW","MODISREL","LHF","SHF","PREC","LWCF","SWCF","LWNS", "SWNS", "LWNT","SWNT","SFC","TOP","PW", "CWP","IWP","PWP","WMAX","W2MAX"/)
 lows = (/         290., 0.,     0.,          0.,         0.,      0.,        0.,  0.,    0.,     0., -80.,   0.,   0.,     100.,  0.,   -200., -200., 0.,   0.,   0., 0.,    0.,    0. /)
 highs = (/        310., 1.,     1.,            1.,       1.,     25.,       300., 40.,  10.,    60.,  0.,   140.,  300.,   300.,  500.,  200.,  200., 60.,  140.,   80., 120.,   100.,  1.5 /)

 ; Vertical profile plots. Set min and max values:
 field_names_z = (/"MSE",   "QT", "QN", "CLD", "RADQR", "RELH", "TABS",  "PRECIP", "TKE", "W2", "dTOBS","dQOBS"/)
 lows_z = (/         320.,   0.,   0.,   0.,   -2.,      0.,    -2.,       -2.,      0.,    0.,   -10., -3./)
 highs_z = (/        360.,   15.,  0.1,  0.5,  1.,      100.,   2.,          2.,     5.0,   0.5,  10.,   3. /)
        
 colors = (/"blue","green","red","magenta","purple","cyan","brown","blueviolet"/)
 dashes = (/1,2,3,4,5,6,7,8/)
;===========================================================================

 fin = new((/nfiles/),file) 
 do i=0,nfiles-1
   fin(i)     = addfile (filename(i) , "r")             ; add file
 end do

wks = gsn_open_wks("ps","ps")
    
      wgt=new((/nrunning/),float)
      wgt = 1./nrunning 
;************************************************ 
; plotting parameters
;************************************************

 res                   = True                       ; plot mods desired
 resl                   = True                       ; plot mods desired

; res@tiMainString      = fld@long_name             ; add title
 res@trYReverse        = False                       ; reverse Y-axis
; res@xyDashPatterns    = 1                         ; choose dash patterns
 res@xyLineThicknessF = 3.0
 res@gsnStringFontHeightF = 0.015
 res@tmXBLabelFontHeightF = 0.015                 ; resize tick labels
 res@tmYLLabelFontHeightF = 0.015
 res@tiXAxisFontHeightF   = 0.015
 res@tiMainFontHeightF   = 0.02
 res@tiYAxisFontHeightF   = 0.015
 res@vpWidthF            = 0.75           ; change aspect ratio of plot
 res@vpHeightF           = 0.5
 res@tmXMajorGrid      = True
 res@tmYMajorGrid      = True
 res@tmXMajorGridThicknessF = 0.5
 res@tmYMajorGridThicknessF = 0.5
 res@tmXMinorGridLineDashPattern = 1
 res@tmYMinorGridLineDashPattern = 1


 resl@gsLineThicknessF  = 3.                    ; line thicker

 res@gsnDraw     = False
 res@gsnFrame    = False


 res@xyExplicitLegendLabels = Case

 plot =new((/nfiles*6/),graphic)

 do k=0,dimsizes(field_names)-1
    print(""+field_names(k))
   do i=0,nfiles-1
     fi = fin(i)
     res@xyLineColor    =  colors(i)
     res@xyDashPattern    =   dashes(i)
     resl@gsLineDashPattern = dashes(i)                    ; solid line
     resl@gsLineColor       = colors(i)                 ; line color
     res@trYMinF = lows(k)
     res@trYMaxF = highs(k)
     res@tiXAxisString = "Days"
     res@pmLegendOrthogonalPosF = -0.8-0.2*i
     if(field_names(k).eq."SFC") then
        fld = fi->SWNS - fi->LWNS - fi->LHF - fi->SHF
        res@tiYAxisString = "W/m2"
        res@tiMainString = "Net Sfc Flux"
     else
     if(field_names(k).eq."TOP") then
        fld = fi->SWNT - fi->LWNT 
        res@tiYAxisString = "W/m2"
        res@tiMainString = "Net TOA Flux"
      else
     if(field_names(k).eq."W2MAX") then
        fld = fi->time
        fld = 0.
        fld =  dim_max(fi->W2)
        res@tiYAxisString = "m2/s2"
        res@tiMainString = "Maximum Vertical Velocity Variance"
     else
     if(field_names(k).eq."LWCF") then
        if(i.eq.0) then
          fld = fi->LWNTC - fi->LWNT 
        else
          fld = fi->LWNTOAC - fi->LWNTOA 
        end if
        res@tiYAxisString = "W/m2"
        res@tiMainString = "Longwave Cloud Forcing"
     else
     if(field_names(k).eq."SWCF") then
        if(i.eq.0) then
          fld = fi->SWNT - fi->SWNTC 
        else
          fld = fi->SWNTOA - fi->SWNTOAC 
        end if
        res@tiYAxisString = "W/m2"
        res@tiMainString = "Shortwave Cloud Forcing"
     else
     if(field_names(k).eq."PWP") then
        fld = fi->RWP+fi->SWP+fi->GWP
        res@tiYAxisString = "g/m2"
        res@tiMainString = "Precipitating water (rain+snow+graupel) path"
      else
        res@tiYAxisString = fi->$field_names(k)$@units
        res@tiMainString = fi->$field_names(k)$@long_name
        fld = fi->$field_names(k)$(:)
      end if
      end if
      end if
      end if
      end if
      end if
      if(field_names(k).eq."SST") then
        fld = fld+300.
      end if
      fld = wgt_runave(fld,wgt,0)
      if(i.eq.0) then
         plot(0)  = gsn_csm_xy (wks,fi->time(:),fld,res)
      else
         plot(i)  = gsn_add_polyline (wks,plot(0),fi->time(:),fld,resl)
      end if
      delete(fld)
   end do
   draw (plot(0))
   yy=0.78
   if(nfiles.gt.1) then
     do i=0,nfiles-1
      res@txFontHeightF   = 0.01
      res@txJust   = "BottomLeft"
      resl@gsLineDashPattern = dashes(i)                    ; solid line
      resl@gsLineColor       = colors(i)                 ; line color
      gsn_polyline_ndc(wks,(/0.25,0.30/),(/yy-0.02*i,yy-0.02*i/),resl)
      gsn_text_ndc(wks,Case(i),0.31,yy-0.02*i,res)
     end do
   end if
   frame(wks)

 end do

 plots =new(6,graphic)

 iplot=0
 do k=0,dimsizes(field_names_z)-1
   res@trYMinF = 0.
   res@trYMaxF = max(fi->z({:z_max}))
   res@trXMinF = lows_z(k)
   res@trXMaxF = highs_z(k)
   do i=0,nfiles-1
     fi = fin(i)
     time0=fi->time(0)*0.
     res@xyLineColor    =  colors(i)
     res@xyDashPattern    =   dashes(i)
     resl@gsLineDashPattern = dashes(i)                    ; solid line
     resl@gsLineColor       = colors(i)                 ; line color
     res@tiYAxisString = "height, m"
     if(field_names_z(k).eq."RADFLUX") then
        fld = fi->RADLWDN({time_start+time0:time_stop+time0},{:z_max})
        fld = fld - fi->RADLWUP({time_start+time0:time_stop+time0},{:z_max})
        res@tiXAxisString = "W/m2"
        res@tiMainString = "Net Longwave Radiative Flux"
     else
     if(field_names_z(k).eq."dTOBS") then
        fld = fi->TABS({time_start+time0:time_stop+time0},{:z_max})
        fld = fld - fi->TABSOBS({time_start+time0:time_stop+time0},{:z_max})
        res@tiXAxisString = "K"
        res@tiMainString = "T - Tobs"
     else
     if(field_names_z(k).eq."dQOBS") then
        fld = fi->QV({time_start+time0:time_stop+time0},{:z_max})
        fld = fld - fi->QVOBS({time_start+time0:time_stop+time0},{:z_max})
        res@tiXAxisString = "g/kg"
        res@tiMainString = "Q - Qobs"
     else
        fld = fi->$field_names_z(k)$({time_start+time0:time_stop+time0},{:z_max})
        res@tiXAxisString = fi->$field_names_z(k)$@units
        res@tiMainString = fi->$field_names_z(k)$@long_name
     end if
     end if
     end if
     if(i.eq.0) then
       res@trXMinF = min(fld({z|:z_max},time|:))
       if(res@trXMinF.gt.0.) then
           res@trXMinF = 0.95*res@trXMinF
       else
           res@trXMinF = 1.1*res@trXMinF
       end if
       if(res@trXMaxF.gt.0.) then
           res@trXMaxF = 1.1*res@trXMaxF
       else
           res@trXMaxF = 0.95*res@trXMaxF
       end if
       res@trXMaxF = max(fld({z|:z_max},time|:))
       plot(iplot)  = gsn_csm_xy (wks,dim_avg(fld({z|:z_max},time|:)),fi->z({:z_max}), res)
       print(field_names_z(k)+"  "+res@trXMinF+"  "+res@trXMaxF)
     else
       print(iplot+"   "+i+"   "+(6+i+iplot*(nfiles-1)))
       plot(5+i+iplot*(nfiles-1))  = gsn_add_polyline (wks,plot(iplot),dim_avg(fld({z|:z_max},time|:)),fi->z({:z_max}), resl)
     end if
     delete (fld)
   end do
   plots(iplot)=plot(iplot)
   iplot=iplot+1
   if(iplot.eq.6) then
     iplot=0
     resP = True
     if(nfiles.gt.1) then
      yy = 0.95
      do i=0,nfiles-1
       res@txFontHeightF   = 0.01
       res@txJust   = "BottomLeft"
       resl@gsLineDashPattern = dashes(i)                    ; solid line
       resl@gsLineColor       = colors(i)                 ; line color
       gsn_polyline_ndc(wks,(/0.8,0.85/),(/yy-0.02*i,yy-0.02*i/),resl)
       gsn_text_ndc(wks,Case(i),0.86,yy-0.02*i,res)
      end do
     end if
     gsn_panel(wks,plots,(/3,2/),False)
   end if
 end do




end

