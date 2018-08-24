; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "open_wks.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
; filename = "./LBA_160x160x64_1000m.nc"
 filename = "./LBA_1536x1536x256.nc"

wks = open_wks("NCGM")
;wks = open_wks("X11")
;wks = open_wks("PS")


 f     = addfile (filename , "r")             ; add file

 fld = f->SSE(:,{0:18000})
 fld1 = f->MSE(:,0)
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 fld1&time = fld1&time * 24.

;************************************************
; plotting parameters
;************************************************

 res                   = True                       ; plot mods desired
 res@gsnFrame        = False                   ; don't advance frame yet

; res@tiMainString      = fld@long_name             ; add title
 res@trYReverse        = False                       ; reverse Y-axis
; res@xyDashPatterns    = 1                         ; choose dash patterns

;  res@trXMinF  = min(fld)                   ; min value on y-axis
;  res@trXMaxF  =  max(fld)                  ; max value on y-axis
  res@trXMinF  = -20.                   ; min value on y-axis
  res@trXMaxF  =  20.                  ; max value on y-axis

 times = (/0.5,1.,1.5,2.,2.5,3,3.5,4.,4.5,5.,5.5,6./)

 do i=0,dimsizes(times)-1
   res@xyLineColors    =   i+2 
   fff = fld({times(i)},:) 
   fff = fld1({times(i)}) -fff
   plot  = gsn_csm_xy (wks,fff,fld&z,res) ; create plot
;   plot  = gsn_csm_xy (wks,fld({times(i)},:),fld&z,res) ; create plot
 end do

 draw(wks)
 frame(wks)


end

