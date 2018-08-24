; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of files:
 nfiles = 2

 nrunning = 1

 filename = new((/nfiles/),string)

 ; set file names:
filename(0) = "./../../OUT_STAT/CLIMATE_32x28sp_4km_20s_M_RRTMR_240.nc"
filename(1) = "./../../OUT_STAT/CLIMATE_32x28sp_4km_20s_M_RRTMR_240_C500.nc"

;filename(0) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_M_CAMNR_270.nc"
;filename(1) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_M_CAMR_270.nc"
;filename(2) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_M_RRTMNR_270.nc"
;filename(3) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_M_RRTMR_270.nc"
;filename(4) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_S_CAM_270.nc"
;filename(5) = "./../../OUT_STAT/SOLAR_32x28sp_4km_20s_S_RRTM_270.nc"

 Case = new((/nfiles/),string)

 ; Case names:
 Case(0) = "CCN100"
 Case(1) = "CCN500"
; Case(0) = "M_CAMNR"
; Case(1) = "M_CAMR"
; Case(2) = "M_RRTMNR"
; Case(3) = "M_RRTMR"
; Case(4) = "S_CAM"
; Case(5) = "S_RRTM"


 ; Set time interval (in days) for time averaged vertical profile plots:

 time_start = 150.
 time_stop = 200.

 ; define threshold of total water (in g/m3) to defone PBL top:
 qt_threshold = 6.5

 ; Define maximum height (in m) plotted:

 z_max = 5000.

 ; Time evolution plots: Set min and max values:
 field_names = (/"SST","CLDSHD","LHF","SHF","PW","CWP","WMAX","PREC","TKE","W2MAX","LWNT","SWNT","ISCCPTOT","MODISTOT","MISRTOT","MODISREL","MODISREI","ISCCPPTOP","MODISPTOP","MISRZTOP","ISCCPLOW","MODISLOW","ISCCPHGH","MODISHGH"/)
 lows = (/         295., 0.,      0.,   0.,   0.,  0.,    0.,   0.,    0.,   0.,    250., 0.,    0., 0., 0., 0.,  0.,   0.,     0.,   0. , 0., 0., 0., 0./)
 highs = (/        305., 1.,     250., 30.,  50., 100.,   20.,  10.,  4000., 0.6,  350., 1200., 1., 1., 1., 20., 400., 1000., 1000., 15., 1.,1.,1.,1./)

 ; Vertical profile plots. Set min and max values:
 field_names_z = (/"MSE",   "QT", "QN",  "CLD","THETA", "RELH", "W2", "TNUDGE", "TVFLUX","QNUDGE","PRECIP","RADQR","QC","NC","QR","QCCLD","NCCLD","WCLD"/)
 lows_z = (/         300.,   0.,   0.,    0.,   290.,   0.,     0.,    -5.,     -10.,     -10. ,   0.,      -5.,    0.,   0.,  0.,  0., 0.,   0./)
 highs_z = (/        350.,   20.,  0.3,  1.,  310.,  100.,    0.5,     5.,      30.,     10.,     5.,     5.,   0.1,  20.,  0.02, 2.0, 100., 1. /)
        
 colors = (/"black","blue","green","red","blue","green"/)
 dashes = (/0,0,0,0,1,1/)
;===========================================================================

 fin = new((/nfiles/),file) 
 do i=0,nfiles-1
   fin(i)     = addfile (filename(i) , "r")             ; add file
 end do

wks = gsn_open_wks("ps","ps")
    
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
 res@vpWidthF            = 0.8           ; change aspect ratio of plot
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
 fld_plot = new((/nfiles,1000/),float)
 x_plot = fld_plot

 do k=0,dimsizes(field_names)-1
   fld_plot@_FillValue = 999.
   fld_plot = 999.
   res@xyLineColors    =  colors
   res@xyDashPatterns    =   dashes
   res@trYMinF = lows(k)
   res@trYMaxF = highs(k)
   res@tiXAxisString = "Hours"
   res@pmLegendOrthogonalPosF = -0.4
   do i=0,nfiles-1
     fi = fin(i)
     if(field_names(k).eq."TKE") then
        z = fi->z
        adz = 0.5*(z(0)+z(1))
        fld = fi->time
        fld = 0.
        do m=1,dimsizes(z)-2
          fld = fld + fi->TKE(:,m)*adz*fi->RHO(:,m)
          adz = 0.5*(z(m+1)-z(m-1))
        end do
        res@tiYAxisString = "kg/m/s2"
        res@tiMainString = "Vertically Integrated TKE"
        delete(z)
     else
     if(field_names(k).eq."Zi") then
        fld = fi->time
        fld = 0.
        z = fi->z
        do n=0,dimsizes(fld)-1
          fld(n) = 0.
          do m=dimsizes(z)-2,1,1
            if(fi->QT(n,m).gt.qt_threshold) then
              fld(n) = z(m-1)-(fi->QT(n,m-1)-fi->QT(n,m))/(z(m-1)-z(m))*(qt_threshold-fi->QT(n,m))
              break
            end if
          end do
        end do
        res@tiYAxisString = "m"
        res@tiMainString = "PBL Top"
        delete(z)
      else
     if(field_names(k).eq."W2MAX") then
        fld = fi->time
        fld = 0.
        fld =  dim_max(fi->W2)
        res@tiYAxisString = "m2/s2"
        res@tiMainString = "Maximum Vertical Velocity Variance"
      else
        res@tiYAxisString = fi->$field_names(k)$@units
        res@tiMainString = fi->$field_names(k)$@long_name
        fld = fi->$field_names(k)$(:)
        if(isatt(fld,"_FillValue")) then
         assignFillValue(fld,fld_plot)
         fld_plot(i,:) = fld@_FillValue
        end if
      end if
      end if
      end if
      nnn = dimsizes(fi->time)
      fld_plot(i,:nnn-1) = fld(:) 
      x_plot(i,:nnn-1) = fi->time(:) 
      delete(fld)
   end do
   nrunning = 24
   wgt=new((/nrunning/),float)
   wgt = 1./nrunning
   fld_plot = wgt_runave(fld_plot,wgt,0)
   plot(0)  = gsn_csm_xy (wks,x_plot,fld_plot,res)
   draw (plot(0))
   yy=0.7
   if(nfiles.gt.1) then
     do i=0,nfiles-1
      res@txFontHeightF   = 0.01
      res@txJust   = "BottomLeft"
      resl@gsLineDashPattern = dashes(i)                    ; solid line
      resl@gsLineColor       = colors(i)                 ; line color
      gsn_polyline_ndc(wks,(/0.8,0.85/),(/yy-0.02*i,yy-0.02*i/),resl)
      gsn_text_ndc(wks,Case(i),0.86,yy-0.02*i,res)
     end do
   end if
   frame(wks)

 end do

 plots =new(6,graphic)

 iplot=0
 do k=0,dimsizes(field_names_z)-1
   do i=0,nfiles-1
     fi = fin(i)
     res@xyLineColor    =  colors(i)
     res@xyDashPattern    =   dashes(i)
     resl@gsLineDashPattern = dashes(i)                    ; solid line
     resl@gsLineColor       = colors(i)                 ; line color
     res@trYMinF = 0.
     res@trYMaxF = max(fi->z({:z_max}))
     res@trXMinF = lows_z(k)
     res@trXMaxF = highs_z(k)
     res@tiYAxisString = "height, m"
     if(field_names_z(k).eq."RADFLUX") then
        fld = fi->RADLWDN({time_start:time_stop},{:z_max})
        fld = fld - fi->RADLWUP({time_start:time_stop},{:z_max})
        fld = fld + fi->RADSWDN({time_start:time_stop},{:z_max})
        fld = fld - fi->RADSWUP({time_start:time_stop},{:z_max})
        res@tiXAxisString = "W/m2"
        res@tiMainString = "Net Radiative Flux"
     else
        fld = fi->$field_names_z(k)$({time_start:time_stop},{:z_max})
        res@tiXAxisString = fi->$field_names_z(k)$@units
        res@tiMainString = fi->$field_names_z(k)$@long_name
     end if
     if(i.eq.0) then
       plot(iplot)  = gsn_csm_xy (wks,dim_avg(fld(z|:,time|:)),fi->z({:z_max}), res)
     else
       print(iplot+"   "+i+"   "+(6+i+iplot*(nfiles-1)))
       plot(5+i+iplot*(nfiles-1))  = gsn_add_polyline (wks,plot(iplot),dim_avg(fld(z|:,time|:)),fi->z({:z_max}), resl)
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

