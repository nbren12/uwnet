
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
 filename = "./../../../OUT_STAT/GATE_IDEAL_S_2048x2048x256_100m_2s.nc"

height_max=18000.
time1 = 0.1/24.
time2 = 12./24.
time3 = 24./24.

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")

 f     = addfile (filename , "r")             ; add file

; ===========================

; --- Create background skew-T and plot sounding + winds----------------
                                ; Create winds for demo
  z = f->z({0:height_max}) 
  p = f->p({0:height_max}) 

  u = f->U({time1},{0:height_max}) 
  v = f->V({time1},{0:height_max}) 
  tc1 = f->TABS({time1},{0:height_max}) 
  rh1 = f->RELH({time1},{0:height_max}) 
  tdc1 = dewtemp_trh(tc1,rh1) - 273.16
  tc1 = tc1 - 273.16

  tc2 = f->TABS({time2},{0:height_max}) 
  rh2 = f->RELH({time2},{0:height_max}) 
  tdc2 = dewtemp_trh(tc2,rh2) - 273.16
  tc2 = tc2 - 273.16

  tc3= f->TABS({time3},{0:height_max}) 
  rh3 = f->RELH({time3},{0:height_max}) 
  tdc3 = dewtemp_trh(tc3,rh3) - 273.16
  tc3 = tc3 - 273.16

  skewtOpts                 = True
  skewtOpts@DrawColAreaFill = False    ; default is False
  skewtOpts@DrawColLine = False    ; default is False
  skewtOpts@DrawStandardAtm = False
  skewtOpts@DrawHeightScale=True
  skewtOpts@DrawHeightScaleFt=False
  skewtOpts@DrawFahrenheit = False
  skewtOpts@DrawWind     =  False
  skewtOpts@ThermoInfo     =  False
  skewtOpts@Cape     = False 
  

  skewt_bkgd = skewT_BackGround (wks, skewtOpts)
  skewtOpts@DashPattern = 0
  skewt_data1 = skewT_PlotData   (wks, skewt_bkgd, p,tc1,tdc1,z,u,v,skewtOpts)
  skewtOpts@DashPattern = 2
  skewt_data2 = skewT_PlotData   (wks, skewt_bkgd, p,tc2,tdc2,z,u,v,skewtOpts)
  skewtOpts@DashPattern = 5
  skewt_data3 = skewT_PlotData   (wks, skewt_bkgd, p,tc3,tdc3,z,u,v,skewtOpts)
  draw (skewt_bkgd)
  draw (skewt_data1)
  draw (skewt_data2)
  draw (skewt_data3)
  frame(wks)



end
