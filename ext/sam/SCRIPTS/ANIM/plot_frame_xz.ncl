; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin
;=============================================
; data processing
;=============================================

nframe = asciiread("number",1,"integer")

height_max=16.
;xmin =  0
;xmax = 2600
xmin =  1200
xmax = 1600

  wks  = gsn_open_wks("ncgm","")    
;  wks  = gsn_open_wks("ps","")   
;  wks  = gsn_open_wks("pdf","")  

 gsn_define_colormap(wks,"rainbow+white+gray")        ; choose color map

 los = (/0.,0.,-6./)
 his = (/3.,3.,6./)
 fields = (/"QN","QP","TABS"/)

 
 f     = addfile ("input3D.nc", "r")             ; add file

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
   fld = fld1(nframe,:,:)
   if(fields(i).eq."TABS") then
    tabs0 = f->TABS(0,:,0)
    nz = dimsizes(f->z)
    do k=0,nz-1
       fld(k,:) = fld(k,:) - tabs0(k)
    end do
   end if


   fld!0="z"
   fld!1="x"
   fld&z = f->z * 0.001
   fld&x = f->x * 0.001 

   res@cnLevelSelectionMode    = "ManualLevels"   ; manual levels
   res@cnMinLevelValF          =  los(i)             ; min level
   res@cnMaxLevelValF          = his(i)              ; max level
   res@cnLevelSpacingF         = (his(i)-los(i))/25. ; interval

   plot(i%nplots) = gsn_csm_contour(wks, fld({0:height_max},{xmin:xmax}), res) 
   print(" "+i%nplots)


   if((i+1)%nplots.eq.0.or.i.eq.dimsizes(fields)-1) then
     gsn_panel(wks,plot,(/nplotsy,nplotsx/),pres)           ; draw: 2-across, 2-down
   end if
 txres               = True                     ; text mods desired
 txres@txFontHeightF = 0.013                     ; font smaller. default big
 gsn_text_ndc(wks,""+nframe,0.94,0.97,txres)

 delete(fld1)
 delete(fld)

 end do

end

