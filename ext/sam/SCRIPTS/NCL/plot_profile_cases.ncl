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
time_start =  2.
time_end = 3. 

 f = new((/10/),file)

; all data should have identical vertical grid

 f(0) = addfile("../../OUT_STAT/GCSSFIRE_64x64x64_25m_2s.nc","r")
 nfiles = 1


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
 res@pmLegendOrthogonalPosF = -0.35                  ; more neg = down
 
 res@pmLegendWidthF         = 0.12                  ; Change width and
 res@pmLegendHeightF        = 0.20                  ; height of legend.
 res@lgLabelFontHeightF     = .02                   ; change font height
 res@lgPerimOn              = False                 ; no box around
 res@xyLineThicknessF = 4.0 

 res@xyExplicitLegendLabels = (/"500m","1000m","500mBUB4K","500mDGR","1000mHAIL","1000mQI0-5"/)
 ff = f(0)
 z = ff->z({0.:zmax})
 data = new((/nfiles,dimsizes(z)/),float)

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QT({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->RELH({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->MSE({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->CLD({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->PRECIP({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QN({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QC({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QI({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QR({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QS({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QG({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->QTFLUX({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->MC({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->W2({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->W3({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->CORECL({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->UW({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot

 do i=0,nfiles-1
   ff=f(i)
   fld = ff->VW({time_start/24.:time_end/24.},{0.:zmax})
   data(i,:) = dim_avg(fld(z|:,time|:))
 end do
 data@long_name = fld@long_name
 res@xyLineColors    =   (/1,2,3,4,6,7,8/) 
 plot  = gsn_csm_xy (wks,data,z,res) ; create plot



end

