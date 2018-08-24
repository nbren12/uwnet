; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"

begin
;=============================================
; data processing
;=============================================


height_max=16.
xmin =  300 ; number of gridpoints behind the squall head
xmax =  100 ; number of grid points in front
frame_start = 300
frame_end = 719


;  wks  = gsn_open_wks("ncgm","")    
;  wks  = gsn_open_wks("ps","")   
  wks  = gsn_open_wks("pdf","")  

 gsn_define_colormap(wks,"rainbow+white+gray")        ; choose color map

 los = (/0.,0.,-6.,0.,-12.,-10./)
 his = (/3.,3.,6.,15.,10.,10./)
 fields = (/"QN","QP","TABS","QV","U","W"/)

 
 f     = addfile ("input3D.nc", "r")             ; add file

 prec = f->QP(frame_start:frame_end,0,:)
 dims=dimsizes(prec)
 edge = new((/dims(0)/),integer)
 do k=0,dims(0)-1
   edge(k) = maxind(prec(k,:))
   print(max(prec(k,:))+"   "+edge(k))
 end do

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

   fld = new((/dimsizes(f->z),xmax+xmin+1/),float)
   fld = 0.
   fld1 = f->$fields(i)$
   do k=0,dims(0)-1
     fld(:,0:xmax+xmin) = fld(:,0:xmax+xmin)+fld1(frame_start+k,:,edge(k)-xmin:edge(k)+xmax)
   end do
   fld = fld/dims(0)
   print(min(fld)+"  "+max(fld))
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
   fld&x = ispan(-xmin,xmax,1) * 1.
   fld@long_name = fld1@long_name

;   res@cnLevelSelectionMode    = "ManualLevels"   ; manual levels
   res@cnLevelSelectionMode    = "AutomaticLevels"   ; manual levels
;   res@cnMinLevelValF          =  los(i)             ; min level
;   res@cnMaxLevelValF          = his(i)              ; max level
;   res@cnLevelSpacingF         = (his(i)-los(i))/25. ; interval

   plot(i%nplots) = gsn_csm_contour(wks, fld({0:height_max},:), res) 
   print(" "+i%nplots)


   if((i+1)%nplots.eq.0.or.i.eq.dimsizes(fields)-1) then
     gsn_panel(wks,plot,(/nplotsy,nplotsx/),pres)           ; draw: 2-across, 2-down
   end if

 delete(fld1)
 delete(fld)

 end do

end

