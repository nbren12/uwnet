; ***********************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "open_wks.ncl"

;************************************************
begin
;************************************************
; read in data
;************************************************

zmax = 20000.
time_start = 12. 
time_end = 24. 

 f = new((/10/),file)

; all data should have identical vertical grid

 f(0) = addfile("../../../OUT_STAT/GATE_IDEAL_S_2048x2048x256_100m_2s.nc","r")
 f(1) = addfile("../../../OUT_STAT/GATE_IDEAL_S_256x256x256_800m_2s.nc","r")
 f(2) = addfile("../../../OUT_STAT/GATE_IDEAL_S_256x256x64PB_800m_2s.nc","r")
 nfiles = 3


;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")
;************************************************
; plotting parameters
;************************************************

 res                   = True                       ; plot mods desired

; res@tiMainString      = fld@long_name             ; add title
 res@trYReverse        = False                       ; reverse Y-axis
; res@xyDashPatterns    = 1                         ; choose dash patterns
 res@pmLegendDisplayMode    = "Always"
 res@pmLegendSide           = "Top"                 ; Change location of 
 res@pmLegendParallelPosF   = .70                   ; move units right
 res@pmLegendOrthogonalPosF = -0.3                  ; more neg = down
 
 res@pmLegendWidthF         = 0.12                  ; Change width and
 res@pmLegendHeightF        = 0.15                  ; height of legend.
 res@lgLabelFontHeightF     = .02                   ; change font height
 res@lgPerimOn              = False                 ; no box around
 res@xyLineThicknessF = 4.0 

 res@xyExplicitLegendLabels = (/"100m","800m","800m L64"/)
 res@xyLineColors    =   (/1,2,4,5,6,7,8/) 
 ff = f(0)
 z = ff->z({0.:zmax})
 data = new((/nfiles,dimsizes(z)/),float)
 zz= new((/nfiles,dimsizes(z)/),float)
 data@_FillValue=-999.
 zz@_FillValue=-999.
 data = -999.
 zz = -999.


 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QT({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld) 
 end do
 data@long_name = ff->QT@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->RELH({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->RELH@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->MSE({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->MSE@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot



 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->CLD({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->CLD@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QN({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QN@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QP({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QP@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QC({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QC@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QI({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QI@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QR({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QR@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QS({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QS@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QG({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QG@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->QTFLUX({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->QTFLUX@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->PRECIP({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->PRECIP@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot


 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->MC({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->MC@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot


 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->W2({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->W2@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot


 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->W3({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->W3@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot


 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->CORECL({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->CORECL@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot



 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->UW({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->UW@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   n = dimsizes(ff->z({0:zmax}))
   fld = ff->VW({time_start/24.:time_end/24.},0:n-1)
   data(i,0:n-1) = dim_avg(fld(z|:,time|:))
   zz(i,0:n-1) = ff->z(0:n-1)
   delete(fld)
 end do
 data@long_name = ff->VW@long_name
 plot  = gsn_csm_xy (wks,data,zz,res) ; create plot





end

