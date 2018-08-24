
; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "skewt_func.ncl"
load "open_wks.ncl"

begin
;=============================================
; data processing
;=============================================
 filename = "./LBA_1536x1536x256.nc"

height_max=18000.

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")

 f     = addfile (filename , "r")             ; add file

; ===========================

; --- Create background skew-T and plot sounding + winds----------------
                                ; Create winds for demo
  skewtOpts                 = True
  skewtOpts@DrawColAreaFill = False    ; default is False
  skewtOpts@DrawColLine = False    ; default is False
  skewtOpts@DrawStandardAtm = False
  skewtOpts@DrawHeightScale=True
  skewtOpts@DrawHeightScaleFt=False
  skewtOpts@DrawFahrenheit = False
  skewtOpts@DrawWind     =  False
  skewtOpts@ThermoInfo     = True 
  skewtOpts@Cape     =  True

  
  time = (/0.01,0.086,0.17,0.25/)
  skewt_plot = new((/dimsizes(time)/),graphic)

  do itime =0,dimsizes(time)-1 

   time1 = time(itime)

  u = f->U({time1},{0:height_max}) 
  v = f->V({time1},{0:height_max}) 
  tc = f->TABS({time1},{0:height_max}) 
  rh = f->RELH({time1},{0:height_max}) 
  tdc = dewtemp_trh(tc,rh) - 273.16
  tc = tc - 273.16
  z = f->z({0:height_max}) 
  p = f->p({0:height_max}) 


  skewt_bkgd = skewT_BackGround (wks, skewtOpts)
  skewt_plot(itime) = skewT_PlotData   (wks, skewt_bkgd, p,tc,tdc,z,u,v,skewtOpts)
;  skewtOpts@DashPattern = 3
 

  draw (skewt_bkgd)
  draw (skewt_plot(itime))
  frame(wks)

  end do




end
