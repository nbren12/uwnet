; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "open_wks.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
 filename = "./../../../OUT_STAT/GATE_IDEAL_S_2048x2048x256_100m_2s.nc"

zmax = 18000.


;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")


 f     = addfile (filename , "r")             ; add file

;************************************************
; plotting parameters
;************************************************

 res                   = True                       ; plot mods desired

; res@tiMainString      = fld@long_name             ; add title
 res@trYReverse        = False                       ; reverse Y-axis
; res@xyDashPatterns    = 1                         ; choose dash patterns
 res@pmLegendDisplayMode    = "Always"
 res@pmLegendSide           = "Top"                 ; Change location of 
 res@pmLegendParallelPosF   = .90                   ; move units right
 res@pmLegendOrthogonalPosF = -0.8                  ; more neg = down
 
 res@pmLegendWidthF         = 0.12                  ; Change width and
 res@pmLegendHeightF        = 0.25                  ; height of legend.
 res@lgLabelFontHeightF     = .02                   ; change font height
 res@lgPerimOn              = False                 ; no box around
 res@xyLineThicknessF = 3.0 

 res@xyExplicitLegendLabels = (/"0","4","8","12","16","20"/)

 times = (/0.1,4.,8.,12.,16.,20./)

 fld = f->QT(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
print(fld&time)
 ;res@trXMinF  = min(fld)                   ; min value on y-axis
 ;res@trXMaxF  =  max(fld)                  ; max value on y-axis
 data = new((/dimsizes(times),dimsizes(fld&z)/),float)

 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 res@xyLineColors    =   (/1,2,3,4,5,6,7/) 
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot
 print(fld&z*1000.+"  "+data(5,:))

 fld = f->MSE(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->SSE(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->CLD(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->PRECIP(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->QPEVP(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->QTFLUX(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->TLFLUX(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot



end

