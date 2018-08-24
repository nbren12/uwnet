
; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "open_wks.ncl"

begin
;=============================================
; data processing
;=============================================
 filename = "../../OUT_STAT/LG_ASTEX_128x128x466_35m_1s.nc"

height_max=3000.
time_beg=165.
time_end=166.6

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")
;wks  = gsn_open_wks("pdf","") 

 gsn_define_colormap(wks,"rainbow+white+gray")        ; choose color map

 los = (/0.,0.,50./)
 his = (/0.6,0.04,100./)
; fields = (/"QN","QP","TVFLUX","CLD"/)
 fields = (/"CLD","QN","QP","PRECIP","THETAV","QT","RELH","TVFLUX","TKE","QT2","TL2","RADQR"/)
; fields = (/"CLD","QN","RELH"/)

 
 f     = addfile (filename , "r")             ; add file

; ===========================
 nplotsx = 1
 nplotsy = 3
nplots = nplotsx*nplotsy

 plot = new(nplotsx*nplots,graphic)

 res                 = True                    ; plot mods desired
 res@gsnFrame        = False                   ; don't advance frame yet
 res@gsnDraw         = False

 res@cnFillOn        = True                    ; turn on color
 res@cnLinesOn       = False
 res@gsnSpreadColors = True                    ; use full colormap

 res@lbLabelStride         = 5
 res@lbBoxLinesOn   = False
 res@lbOrientation   = "vertical"
 res@vpWidthF            = 1.0           ; change aspect ratio of plot
 res@vpHeightF           = 0.3


 res@gsnXAxisIrregular2Linear = True
 res@gsnYAxisIrregular2Linear = True
 pres             = True                    ; panel mods desired
 pres@gsnMaximize = True                    ; fill the page
 pres@gsnPaperOrientation = "auto"

 do i=0,dimsizes(fields)-1

   fld1 = f->$fields(i)$
   fld = fld1({time_beg:time_end},{0:height_max})
   fld&z = fld&z * 0.001
   fld&time = (fld&time - fld&time(0)) * 24.
  res@cnLevelSelectionMode    = "AutomaticLevels"   ; manual levels
;  res@cnLevelSelectionMode    = "ManualLevels"   ; manual levels
;  res@cnMinLevelValF          =  los(i)             ; min level
;  res@cnMaxLevelValF          = his(i)              ; max level
;  res@cnLevelSpacingF         = (his(i)-los(i))/25. ; interval

   plot(i%nplots) = gsn_csm_contour(wks, fld(z|:,time|:), res) 
   print(" "+i%nplots)

   if((i+1)%nplots.eq.0.or.i.eq.dimsizes(fields)-1) then
     gsn_panel(wks,plot,(/nplotsy,nplotsx/),pres)           ; draw: 2-across, 2-down
   end if

 end do

end

