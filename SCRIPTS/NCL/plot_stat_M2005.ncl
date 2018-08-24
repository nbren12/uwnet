; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
 ; set number of files:
 nfiles = 4

 filename = new((/nfiles/),string)

 ; set file names:
 filename(0) = "./../../OUT_STAT/BOMEX_1280x104_hires_50m_2s_CCN100_12h.nc"
 filename(1) = "./../../OUT_STAT/BOMEX_32x28sp_4000m_20s_CCN100_12h.nc"
 filename(2) = "./../../OUT_STAT/BOMEX_64x28sp_2000m_20s_CCN100_12h.nc"
 filename(3) = "./../../OUT_STAT/BOMEX_256x28sp_500m_20s_CCN100_12h.nc"

 Case = new((/nfiles/),string)

 ; Case names:
 Case(0) = "HiRes"
 Case(1) = "LoRes4000"
 Case(2) = "LoRes2000"
 Case(3) = "LoRes500"


 ; Set time interval (in hours) for time averaged vertical profile plots:

 time_start = 185.
 time_stop = 190.

 ; define threshold of total water (in g/m3) to defone PBL top:
 qt_threshold = 6.5

 ; Define maximum height (in m) plotted:

 z_max = 5000.

 ; Time evolution plots: Set min and max values:
 field_names = (/"SST","CLDSHD","LHF","SHF","PW","CWP","WMAX","PREC","TKE","W2MAX"/)
 lows = (/         299., 0.,     0.,   0.,   40.,  0.,    0.,   0.,    0.,   0. /)
 highs = (/        303., 1.,     250., 30.,  70., 300.,   70.,  50.,  100000., 2.5 /)

 ; Vertical profile plots. Set min and max values:
 field_names_z = (/"MSE","QT",    "QN",  "CLD","COR","QP",  "RELH", "W2", "TLFLUX", "TVFLUX","QTFLUX","RADFLUX","QC","QI","QR","QG","QS","QP","NC","NI","NR","NG","NS","NC","QCCOR","QICOR","QRCOR","QGCOR","QSCOR","QPCOR","NCCOR","NICOR","NRCOR","NGCOR","NSCOR","NCCOR","QSADV","QSLSADV","QSMPHY","QSSED","QSFLX","QSSDFLX"/)
 lows_z = (/         320.,   0.,   0.,    0.,   0.,     0.,   0.,     0.,  -200.,      -10.,     -50.,    0. ,    0.,0.,0.,0.,0.,0.,0., 0., 0., 0., 0., 0.,0.,0.,0.,0.,0.,0.,0., 0., 0., 0., 0., 0.,-10.,-10.,-10.,-10.,-100.,-200./)
 highs_z = (/        350.,   20.,  0.1,  1.,   0.2,  0.3, 100.,    0.5,  100.,        30.,      300.,   300. , 0.3,0.3,0.3,0.3,0.3,0.3,40., 1., 0.01, 0.01, 0.01, 20.,0.9,0.9,0.9,0.9,0.9,0.9,200., 1., 0.05, 0.01, 0.01, 400.,10.,10.,10.,10.,100.,0./)
        
 colors = (/"black","blue","red","green","red"/)
 dashes = (/0,1,2,3/)
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

 do k=0,dimsizes(field_names)-1
   do i=0,nfiles-1
     fi = fin(i)
     res@xyLineColor    =  colors(i)
     res@xyDashPattern    =   dashes(i)
     res@gsLineDashPattern = dashes(i)                    ; solid line
     res@gsLineColor       = colors(i)                 ; line color
     res@trYMinF = lows(k)
     res@trYMaxF = highs(k)
     res@tiXAxisString = "Hours"
     res@pmLegendOrthogonalPosF = -0.4-0.2*i
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
      end if
      end if
      end if
      if(i.eq.0) then
         plot(0)  = gsn_csm_xy (wks,fi->time(:),fld,res)
      else
         plot(i)  = gsn_polyline (wks,plot(0),fi->time(:),fld,res)
      end if
      delete(fld)
   end do
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

