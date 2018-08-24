
; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "open_wks.ncl"

begin
;=============================================
; data processing
;=============================================
; filename = "./LBA_320x320x64_500m.nc"
; filename = "./LBA_40x40x64_4000m.nc"
; filename = "./LBA_160x160x64_1000m.nc"
; filename = "./LBA_1536x1536x256.nc"
; filename = "./LBA_512x512x128_250m_5s_.nc"
; filename = "./LBA_512x512x128_250m_5s_wet.nc"
; filename = "./LBA_512x512x128_250m_5s_dry.nc"
; filename = "./LBA_512x512x128_250m_5s_nowind.nc"
; filename = "./LBA_512x512x128_250m_5s_sfc_x1.5.nc"
; filename = "./LBA_512x512x128_250m_5s_norainevap.nc"
 filename ="~/DYCOMS_RF02_128x128x96_rev_scalelin.nc"

height_max=1500.

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")
 gsn_define_colormap(wks,"rainbow+white+gray")        ; choose color map

 los = (/0.,0.,20.,0./)
 his = (/1.0,1.0,100.,1.5/)
 fields = (/"CLD","QN","RELH","TKE"/)

 
 f     = addfile (filename , "r")             ; add file

; estimate the PBL top as minimum of buoyancy flux:


BUOY = f->TVFLUX({1./24:},{0:height_max})
ZPBL = f->time({1./24:})
ZW2 = f->time({1./24:})
z = f->z 
time = f->time({1./24:})*24.
W2 = f->W2({1./24:},{0:height_max})

do k=0,dimsizes(time)-1
 ZPBL(k) = z(minind(BUOY(k,:)))
end do
do k=0,dimsizes(time)-1
 zz = z({ZPBL(k):height_max})
 ZW2(k) = zz(maxind(W2(k,{ZPBL(k):height_max})))
 delete(zz)
 if(ZW2(k)-ZPBL(k).lt.100.) then
   ZW2(k) = -999.
 end if
end do
ZPBL(:) = ZPBL(:)*0.001
ZW2@_FillValue = -999.
ZW2(:) = ZW2(:)*0.001
z = z * 0.001

print(ZPBL+"  "+ZW2)

; ===========================
 nplotsx = 1
 nplotsy = 4
nplots = nplotsx*nplotsy

 plot = new(nplotsx*nplots,graphic)
 poly = new(nplotsx*nplots,graphic)
 poly1 = new(nplotsx*nplots,graphic)

 res                 = True                    ; plot mods desired
 res@gsnFrame        = False                   ; don't advance frame yet

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
   fld = fld1(:,{0:height_max})
   fld&z = fld&z * 0.001
   fld&time = fld&time * 24.
;   res@cnLevelSpacingF = (max(fld)-min(fld))/50.
  res@cnLevelSelectionMode    = "ManualLevels"   ; manual levels
  res@cnMinLevelValF          =  los(i)             ; min level
  res@cnMaxLevelValF          = his(i)              ; max level
  res@cnLevelSpacingF         = (his(i)-los(i))/25. ; interval
   res@gsnDraw         = False
   res@gsnFrame         = False


   res@gsnDraw         = False
   res@gsnFrame         = False
   plot(i%nplots) = gsn_csm_contour(wks, fld(z|:,time|:), res) 
   polyres                   = True
   polyres@gsLineThicknessF     = 5.          ; polymarker size
   polyres@gsLineColor     = "White"
   poly(i%nplots)    = gsn_add_polyline(wks,plot(i%nplots),time,ZPBL,polyres)
   if(fields(i).eq."W2") then
   polyres@gsLineColor     = "Red"
   poly1(i%nplots)    = gsn_add_polyline(wks,plot(i%nplots),time,ZW2,polyres)
   end if

   print(" "+i%nplots)

   if((i+1)%nplots.eq.0.or.i.eq.dimsizes(fields)-1) then
     gsn_panel(wks,plot,(/nplotsy,nplotsx/),pres)           ; draw: 2-across, 2-down
   end if

 end do

end

