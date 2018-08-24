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
 res@pmLegendParallelPosF   = .20                   ; move units right
 res@pmLegendOrthogonalPosF = -0.4                  ; more neg = down
 
 res@pmLegendWidthF         = 0.12                  ; Change width and
 res@pmLegendHeightF        = 0.15                  ; height of legend.
 res@lgLabelFontHeightF     = .016                   ; change font height
 res@lgPerimOn              = False                 ; no box around
 res@xyLineThicknessF = 3.0 

 field_names = (/"TTEND","QTEND","RADQR"/)
 res@xyExplicitLegendLabels = (/"T_TEND","Q_TEND","RAD"/)

 data = new((/dimsizes(field_names),dimsizes(f->z({0:zmax}))/),float)

 do i=0,dimsizes(field_names)-1
   data(i,:) = f->$field_names(i)$(0,{0:zmax})
 end do
 res@xyLineColors    =   (/1,2,4,3,5,6,7/) 
 plot  = gsn_csm_xy (wks,data,data&z*0.001,res) ; create plot


field_names = (/"U","V","QT"/)
 res@xyExplicitLegendLabels = (/"U","V","Q"/)

 data = new((/dimsizes(field_names),dimsizes(f->z({0:zmax}))/),float)

 do i=0,dimsizes(field_names)-1
   data(i,:) = f->$field_names(i)$(0,{0:zmax})
 end do
 res@xyLineColors    =   (/1,2,4,3,5,6,7/)
 plot  = gsn_csm_xy (wks,data,data&z*0.001,res) ; create plot


end

