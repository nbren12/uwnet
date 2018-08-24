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

time_start = 10./24.
time_end = 24./24.
zmax = 18000.

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")


 f     = addfile (filename , "r")             ; add file

;************************************************
; plotting parameters
;************************************************

 res                   = True                       ; plot mods desired

 res@trYReverse        = False                       ; reverse Y-axis
 res@pmLegendDisplayMode    = "Always"
 res@pmLegendSide           = "Top"                 ; Change location of 
 res@pmLegendParallelPosF   = .80                   ; move units right
 res@pmLegendOrthogonalPosF = -0.4                  ; more neg = down
 
 res@pmLegendWidthF         = 0.12                  ; Change width and
 res@pmLegendHeightF        = 0.15                  ; height of legend.
 res@lgLabelFontHeightF     = .015                   ; change font height
 res@lgPerimOn              = False                 ; no box around
 res@xyLineThicknessF = 4.0 
 res@xyDashPatterns    = 0                         ; choose dash patterns

 fields = (/"BUOYA","PRESSTR","ADVTR","SHEAR","DISSIP","RESID"/)
 res@xyExplicitLegendLabels = (/"BUOY","PRES","ADV","SHEAR","DISS","SUM"/)
 res@trXMinF  = -0.5                   ; min value on y-axis
 res@trXMaxF  =  0.5                  ; max value on y-axis



 fld = f->TL({time_start:time_end},{0:zmax})
 data = new((/dimsizes(fields),dimsizes(fld&z)/),float)
 sss = data(0,:)
 sss = 0.

 do i = 0, dimsizes(fields)-2
   fld = f->$fields(i)$({time_start:time_end},{0:zmax})
   fld&z = fld&z * 0.001
   fld&time = fld&time * 24.
   data(i,:) = dim_avg(fld(z|:,time|:))*f->RHO(0,{0:zmax})*1000.
   sss = sss + data(i,:)
 end do
 data(dimsizes(fields)-1,:) = sss
 res@xyLineColors    =   (/1,4,3,8,2,7/) 
 res@tiMainString      = "tke_budget"
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot



end

