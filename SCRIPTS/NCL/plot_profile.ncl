; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "open_wks.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************
 filename = "../../OUT_STAT/GCSSFIRE_160x160x160_20m_10m_1s.nc"
; filename = "../../OUT_STAT/GCSSFIRE_64x64x64_25m_2s.nc"

zmax = 2000.


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

 res@xyExplicitLegendLabels = (/"1","2","3","4","5","6"/)

 times = (/0.05,0.5,1.,1.5,2.0,2.5/)

 fld = f->QT(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
print(fld&time)
 data = new((/dimsizes(times),dimsizes(fld&z)/),float)
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 res@xyLineColors    =   (/1,2,3,4,5,7,8/) 
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot
 print(fld&z*1000.+"  "+data(5,:))

 fld = f->QN(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->U(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->V(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot


 fld = f->W2(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

 fld = f->THETAL(:,{0:zmax})
 fld&z = fld&z * 0.001
 fld&time = fld&time * 24.
 do i=0,dimsizes(times)-1
   data(i,:) = fld({times(i)},:)
 end do
 plot  = gsn_csm_xy (wks,data,fld&z,res) ; create plot

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

 fld = f->RADQR(:,{0:zmax})
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

 fld = f->TVFLUX(:,{0:zmax})
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

