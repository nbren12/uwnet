
; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "open_wks.ncl"

begin
;=============================================
; data processing
;=============================================
; filename = "./LBA_160x160x64_1000m.nc"
 filename = "./LBA_1536x1536x256.nc"

wks = open_wks("NCGM")
;wks = open_wks("X11")
;wks = open_wks("PS")
 gsn_define_colormap(wks,"rainbow+white+gray")        ; choose color map

 
 f     = addfile (filename , "r")             ; add file

 fld = f->QT2(:,{0:1000})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
; ===========================
; create plot
; ===========================

 res                 = True                    ; plot mods desired
 res@gsnDraw         = False                   ; don't draw yet
 res@gsnFrame        = False                   ; don't advance frame yet

 res@cnFillOn        = True                    ; turn on color
 res@cnLinesOn       = False
 res@gsnSpreadColors = True                    ; use full colormap

 res@vpWidthF            = 0.7           ; change aspect ratio of plot
 res@vpHeightF           = 0.4
; res@lbLabelStride         = 20
 res@lbBoxLinesOn   = False

; res@cnLevelSpacingF      =   0.2
 res@gsnXAxisIrregular2Linear = True
 res@gsnYAxisIrregular2Linear = True

 plot = gsn_csm_contour(wks, fld(z|:,time|:), res) 
 plot = ZeroLineContour(plot)

 draw(wks)
 frame(wks)
end

