
; ===========================================
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
load "skewt_func_new.ncl"
load "open_wks.ncl"

begin
;=============================================
; data processing
;=============================================
filename = "./../../../OUT_STAT/GATE_IDEAL_S_2048x2048x256_100m_2s.nc"

height_max=18000.
time1 = 0.1/24.
time2 = 24./24.

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")

 f     = addfile (filename , "r")             ; add file

; ===========================

; --- Create background skew-T and plot sounding + winds----------------
                                ; Create winds for demo
  u = f->U({time1},{0:height_max}) 
  v = f->V({time1},{0:height_max}) 
  
  tc = f->TABS({time1},{0:height_max}) 
  rh = f->RELH({time1},{0:height_max}) 
  tdc = dewtemp_trh(tc,rh) - 273.16
  tc = tc - 273.16
  z = f->z({0:height_max}) 
  p = f->p({0:height_max}) 

  u1 = f->U({time2},{0:height_max}) 
  v1 = f->V({time2},{0:height_max}) 
  tc1 = f->TABS({time2},{0:height_max}) 
  rh1 = f->RELH({time2},{0:height_max}) 
  tdc1 = dewtemp_trh(tc1,rh1) - 273.16
  tc1 = tc1 - 273.16

  skewtOpts                 = True
  skewtOpts@DrawColAreaFill = False    ; default is False
  skewtOpts@DrawColLine = False    ; default is False
  skewtOpts@DrawStandardAtm = False
  skewtOpts@DrawHeightScale=True
  skewtOpts@DrawHeightScaleFt=False
  skewtOpts@DrawFahrenheit = False
  skewtOpts@ThermoInfo     = True 
  skewtOpts@Cape     =  True

  skewt_bkgd = skewT_BackGround (wks, skewtOpts)
  skewtOpts@DrawWind     = False 
  skewtOpts@WspdWdir  = False
  skewtOpts@Wthin     = 5
  skewtOpts@DashPattern = 2
  u =  -999.  
  u@_FillValue = -999.
  v =  -999.  
  v@_FillValue = -999.
  skewt_data = skewT_PlotData   (wks, skewt_bkgd, p,tc,tdc,z,u,v,skewtOpts)
  skewtOpts@DashPattern = 0
  skewtOpts@DrawWind     = False 
;  u1 =  -999.  
;  u1@_FillValue = -999.
;  v1 =  -999.  
;  v1@_FillValue = -999.
  skewt_data1 = skewT_PlotData   (wks, skewt_bkgd, p,tc1,tdc1,z,u1,v1,skewtOpts)
  draw (skewt_bkgd)
  draw (skewt_data)
  draw (skewt_data1)
  frame(wks)



end
