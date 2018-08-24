load "open_wks.ncl"
load "/disk2/CSUCEM/NCL/plotXYmarker.ncl"
load "/disk2/CSUCEM/NCL/frameit.ncl"


begin

wks = open_wks("NCGM")
;wks = open_wks("X11")
;wks = open_wks("PS")

dt=1.

filename="./LBA_1536x1536x256.nc"


file_in=addfile(filename,"r")

time_scale = 3.

;-------------------------------------------------------------------

period_begin = new((/50/),integer)
period_end = new((/50/),integer)
period_begin = 60
period_end = 80

nevents_max = 1

precip_min = 10.
time_min = 1.

day = file_in->time
precip = file_in->PRECIP(:,0)
w2max = dim_max(file_in->W2(:,20:))
w2mean = dim_avg(file_in->W2)*100.
cape =  file_in->CAPE
nday = dimsizes(day)
print(day+"  "+precip+"   "+w2max+"     "+w2mean)

;==========================================================
; select events:

nevents = 1

print(24.*(day(period_end(:nevents-1))-day(period_begin(:nevents-1)))+"   "+24*day(period_begin(:nevents-1))+"  "+24*day(period_end(:nevents-1)))
;==========================================================

z=file_in->z * 0.001
nz = dimsizes(z)
do i=0,nz-1
 if(z(i).le.18.) then
  nz1 = i
 end if
end do
z1 = z(:nz1-1)
dz=z1
do k=1,nz1-2
  dz(k)=0.5*(z1(k+1)-z1(k-1))
end do
dz(0)=2.*z1(0)
dz(nz1-1)=dz(nz1-2)

;--------------------------------------------

opt=0
opt@case_names = (/"A","B","C","D","E"/)

opt@item_count=dimsizes(opt@case_names)

opt@line_colors = (/2,3,4,6,7,8,1,1,1,1,1,1/)
;opt@line_colors = (/1,1,1,1,1,1,1,1,1,1,1,1/)
opt@dash_patterns = (/0,0,0,0,0,0,0,0,0,0,0,0/)
opt@line_thicknesses = (/1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5,0.5,0.5,0.5/)
opt@marker_colors = opt@line_colors
opt@marker_types = (/4,6,7,9,12,2,8,11,16,16/)
opt@marker_sizes = (/0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.005,0.004,0.004/)
opt@markers="True"

opt@legend = "False"
opt@reverse = "False"
opt@x_lab = "True"
text_letter = (/":I:a",":I:b",":I:c",":I:d",":I:e",":I:f",":I:g",":I:h",":I:i",":I:j",":I:k",":I:l",":I:m",":I:n",":I:o"/)
nplotx=4
nploty=2
xleft=0.04
ytop=0.86
dxx=0.19
dyy=0.32
xl=new((/nplotx*nploty+1/),float)
yt=xl
dx=xl
dy=xl
do j=0,nploty-1
 do i=0,nplotx-1
   k=i+nplotx*j+1
   xl(k)=xleft+dxx*i+0.02
   dx(k)=dxx-0.5*xleft
   yt(k)=ytop-dyy*j
   dy(k)=dyy-0.08
 end do
end do

nplot=0


rho=file_in->RHO(0,0:nz1-1)
bet=9.81/file_in->TABS(0,0:nz1-1)
fld = new((/30,nz1/),float)
res = new((/30,nz1/),float)
z2 = new((/30,nz1/),float)

;-----------------------------------------------------------
 nnn=floattointeger(time_scale/((day(2)-day(1))*24.))+1
 if(nnn.gt.1) then
    wgt=new((/nnn/),float)
    wgt=1./nnn
    precip=wgt_runave(precip,wgt,1)
    delete(wgt)
 end if


opt@markers="False"
opt@xl = 0.10
opt@dx = 0.90
opt@yt = 0.95
opt@dy = 0.3

opt@line_color = 0
opt@dash_pattern = 0
opt@line_thickness = 0

opt@xmin=min(day)
opt@xmax=max(day)
opt@ymin=0.
opt@ymax=100.

opt@xaxis = "Julian Day"
opt@yaxis = "mm/day"
opt@title = "Precipitation"

plotXY(wks, opt, precip,day)

opt@line_thicknesses(0) = 3.


opt@leg_xl=xl(2)+0.02
opt@leg_yt=yt(2)-0.02
opt@leg_dx=0.8*dx(2)
opt@leg_dy=0.02*nevents
opt@leg_fontsize=0.009
legendXY(wks,opt)

opt@yaxis = "km"

;----------------------------------------------------
; LW Pot Temp


do n=0,nevents-1
  arr = file_in->TL(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "h:B:L"
opt@xaxis = "K"

opt@xmin=0.95*min(fld(:nevents-1,:))
opt@xmax=1.05*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
; QT


do n=0,nevents-1
  arr = file_in->QT(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "q:B:T"
opt@xaxis = "g/kg"

opt@xmin=0.*min(fld(:nevents-1,:))
opt@xmax=1.05*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
; Temp Tendency


do n=0,nevents-1
  arr = file_in->TTEND(period_begin(n):period_end(n),:nz1-1)+\
        file_in->RADQR(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "L.S. Cooling"
opt@xaxis = "K/day"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
; Vapor Tendency


do n=0,nevents-1
  arr = file_in->QTEND(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "L.S. Moistening"
opt@xaxis = "g/kg/day"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
; Wind


do n=0,nevents-1
  arr = file_in->U(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  arr1= file_in->V(period_begin(n):period_end(n),:nz1-1)
  arr1!0 = "time"
  arr1!1 = "z"
  fld(n,:) = sqrt(dim_avg(arr(z|:,time|:))^2+dim_avg(arr1(z|:,time|:))^2)
  z2(n,:) = z1(:)
  delete(arr)
  delete(arr1)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "Wind"
opt@xaxis = "m/s"

opt@xmin=0.
opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
; Wind Direction


do n=0,nevents-1
  arr = file_in->U(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  arr1= file_in->V(period_begin(n):period_end(n),:nz1-1)
  arr1!0 = "time"
  arr1!1 = "z"
  u = dim_avg(arr(z|:,time|:))
  v = dim_avg(arr1(z|:,time|:))
  angle = atan(v/u)*180./3.1415
  do i=0,dimsizes(angle)-1
    if(u(i).lt.0) then
      angle(i) = angle(i)+180.
    end if
  end do
  delete(u)
  delete(v)
  fld(n,:) = angle(:)
  delete(angle)
  z2(n,:) = z1(:)
  delete(arr)
  delete(arr1)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "Wind Direction"
opt@xaxis = "Deg."

opt@xmin=-180.
opt@xmax=180.

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
;  CLD


do n=0,nevents-1
  arr = file_in->CLD(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "Cloud Cover"
opt@xaxis = ""

opt@xmin=1.0*min(fld(:nevents-1,:))
opt@xmax=1.0*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
;Precipitation flux


do n=0,nevents-1
  arr = file_in->PRECIP(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:)) 
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "Precip"
opt@xaxis = "mm/day"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

nplotx=4
nploty=2
xleft=0.04
ytop=0.86
dxx=0.19
dyy=0.32
delete(xl)
delete(dx)
delete(yt)
delete(dy)
xl=new((/nplotx*nploty+1/),float)
yt=xl
dx=xl
dy=xl
do j=0,nploty-1
 do i=0,nplotx-1
   k=i+nplotx*j+1
   xl(k)=xleft+dxx*i+0.02
   dx(k)=dxx-0.5*xleft
   yt(k)=ytop-dyy*j
   dy(k)=dyy-0.08
 end do
end do

nplot=0
frame(wks)

opt@leg_xl=xl(8)+0.02
opt@leg_yt=yt(8)-0.02
opt@leg_dx=0.8*dx(8)
opt@leg_dy=0.02*nevents
opt@leg_fontsize=0.009
legendXY(wks,opt)

;----------------------------------------------------
; buoyancy flux


do n=0,nevents-1
  arr = file_in->BUOYA(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z" 
  fld(n,:) = dim_avg(arr(z|:,time|:))/bet(:) 
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w':G:H:R::B:v:N:'>"
opt@xaxis = "Km/s"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
; vertical velocity variance


do n=0,nevents-1
  arr = file_in->W2(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z" 
  fld(n,:) = dim_avg(arr(z|:,time|:)) 
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w':S:2:N:>"
opt@xaxis = "m:S:2:N:/s:S:2"

;opt@xmin=0.
;opt@xmax=4.
opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
; vertical velocity third moment


do n=0,nevents-1
  arr = file_in->W3(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w':S:3:N:>"
opt@xaxis = "m:S:3:N:/s:S:3"

;opt@xmin=0.
;opt@xmax=50.
opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
;TKE 


do n=0,nevents-1
  arr = file_in->TKE(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))*rho(:) 
  z2(n,:) = z1(:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "CKE"
opt@xaxis = "kg/m/s:S:2"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=0.6*max(fld(:nevents-1,:))


plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
;  L W Pot Temp flux


do n=0,nevents-1
  arr = file_in->TLFLUX(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))/rho(:)/1000.
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w'h:B:L:N:'>"
opt@xaxis = "Km/s"

;opt@xmin=-3.0
;opt@xmax=0.0
opt@xmin=1.2*min(fld(:nevents-1,:))
opt@xmax=1.2*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
;  Total water flux


do n=0,nevents-1
  arr = file_in->QTFLUX(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))/rho(:)/2.5e3
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w'q:B:T:N:'>"
opt@xaxis = "m/s g/kg"

;opt@xmin=0.
;opt@xmax=0.8
opt@xmin=1.2*min(fld(:nevents-1,:))
opt@xmax=1.2*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)


;----------------------------------------------------
;  LWPT variance


do n=0,nevents-1
  arr = file_in->TL2(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<h:B:L:N:':S:2:N:>"
opt@xaxis = "K:S:2"

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
;  Total water variance


do n=0,nevents-1
  arr = file_in->QT2(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<q:B:T:N:':S:2:N:>"
opt@xaxis = "(g/kg):S:2"

opt@xmin=0.
opt@xmax=5.
;opt@xmin=1.1*min(fld(:nevents-1,:))
;opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)



;;;-----------------------------------------------------
;   U
do n=0,nevents-1
  arr = file_in->U(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  res(n,:) = res(n,:) + fld(n,:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<U>"
opt@xaxis = "m/s"

opt@xmin=1.2*min(fld(:nevents-1,5:))
opt@xmax=1.2*max(fld(:nevents-1,5:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;;;-----------------------------------------------------
;   V
do n=0,nevents-1
  arr = file_in->V(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  res(n,:) = res(n,:) + fld(n,:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<V>"
opt@xaxis = "m/s"

opt@xmin=1.2*min(fld(:nevents-1,5:))
opt@xmax=1.2*max(fld(:nevents-1,5:))

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;;;-----------------------------------------------------
;   uw
do n=0,nevents-1
  arr = file_in->UW(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  res(n,:) = res(n,:) + fld(n,:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<u'w'>"
opt@xaxis = "m:S:2:N:/s:S:2"

opt@xmin=1.2*min(fld(:nevents-1,5:))
opt@xmax=1.2*max(fld(:nevents-1,5:))
opt@xmin=-0.6
opt@xmax=0.2

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)


;;;-----------------------------------------------------
;   vw
do n=0,nevents-1
  arr = file_in->VW(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))
  res(n,:) = res(n,:) + fld(n,:)
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<v'w'>"
opt@xaxis = "m:S:2:N:/s:S:2"

;opt@xmin=1.2*min(fld(:nevents-1,5:))
;opt@xmax=1.2*max(fld(:nevents-1,5:))
opt@xmin=-0.2
opt@xmax=0.6

plotXYall1(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

nplot = nplotx*nploty
;----------------------------------------------------
; Scales:

z_star = new((/50/),float)
z_star0 = new((/50/),float)
w_star= new((/50/),float)
tke= new((/50/),float)
eps= new((/50/),float)
dissip= new((/50/),float)
dissip_prec= new((/50/),float)
buo= new((/50/),float)
shear= new((/50/),float)
resd= new((/50/),float)
v_star= new((/50/),float)
t_star= new((/50/),float)
q_star= new((/50/),float)
rho_star= new((/50/),float)

do n=0,nevents-1

  arr = file_in->BUOYA(period_begin(n):period_end(n),:nz1-1)
  buoy = dim_avg(arr(z|:,time|:)) 
  arr = file_in->SHEAR(period_begin(n):period_end(n),:nz1-1)
  shearz = dim_avg(arr(z|:,time|:)) 
  arr = file_in->DISSIP(period_begin(n):period_end(n),:nz1-1)+\
        file_in->DIFTR(period_begin(n):period_end(n),:nz1-1)
  dissipz = dim_avg(arr(z|:,time|:))
  arr = file_in->PRECIP(period_begin(n):period_end(n),:nz1-1)
  dissipz_prec = dim_avg(arr(z|:,time|:))/86400.*9.81
  arr = file_in->TKE(period_begin(n):period_end(n),:nz1-1)
  tkez = dim_avg(arr(z|:,time|:))*rho(:)
  arr = file_in->W2(period_begin(n):period_end(n),:nz1-1)
  w2z = dim_avg(arr(z|:,time|:))*rho(:)
  arr = file_in->ADVTR(period_begin(n):period_end(n),:nz1-1)+\
        file_in->PRESSTR(period_begin(n):period_end(n),:nz1-1)
  resz =  dim_avg(arr(z|:,time|:)) 


; Height of the most negative buoyancy flux:

  buoy_min=0.
  do k=nz1-1,0,1
   if(z(k).gt.3.and.buoy(k).lt.buoy_min) then 
      buoy_min = buoy(k)
      z_star(n)=z(k)
      k_star = k
   end if
  end do


;  Height where buoyancy flux becomes 0:

  do k=k_star-1,0,1
   if(buoy(k).gt.0.) then 
      k_star0 = k
      z_star0(n)=z(k)+buoy(k)/(buoy(k)-buoy(k+1))*(z(k+1)-z(k))
      break
   end if
  end do

; Height where buoyancy becomes positive after being negative near the PBL top:

  buoy_min=10000.
  do k=0,nz1-1
   if(z(k).le.5.and.buoy(k).lt.buoy_min) then 
      buoy_min = buoy(k)
      k_star1 = k
   end if
  end do
  do k=k_star1+1,nz1-1
   if(buoy(k).gt.0.) then 
      k_star1 = k
      break
   end if
  end do
  k_star1 = k_star1+1

  print(n+"  "+z(k_star1)+ "   "+z(k_star))

  
  z2(n,:) = z1(:)/z_star(n)

; scales:

  rho_star(n) = sum(rho(k_star1:k_star)*dz(k_star1:k_star))/sum(dz(k_star1:k_star))
  w_star(n) = (2.5*sum(rho(k_star1:k_star)*buoy(k_star1:k_star)*dz(k_star1:k_star)/rho_star(n))*1000.)^0.3333

  kr = nz1-1
  kr1 = 0
  tke(n) = sum(tkez(kr1:kr)*dz(kr1:kr))*1000.
  eps(n) = sum(w2z(kr1:kr)*dz(kr1:kr))*1000./ tke(n)
  dissip(n) = sum(dissipz(kr1:kr)*rho(kr1:kr)*dz(kr1:kr))*1000.
  dissip_prec(n) = sum(dissipz_prec(kr1:kr)*dz(kr1:kr))*1000.
  buo(n) = sum(rho(kr1:kr)*buoy(kr1:kr)*dz(kr1:kr))*1000.
  shear(n) = sum(rho(kr1:kr)*shearz(kr1:kr)*dz(kr1:kr))*1000.
  resd(n) = sum(rho(kr1:kr)*resz(kr1:kr)*dz(kr1:kr))*1000.

  arr = file_in->TLFLUX(period_begin(n):period_end(n),:nz1-1)
  wtl = dim_avg(arr(z|:,time|:))/1000. 
  t_star(n) =-sum(wtl(0:k_star)*dz(0:k_star))/w_star(n)/z_star(n)/rho_star(n)

  arr = file_in->QTFLUX(period_begin(n):period_end(n),:nz1-1)
  wqt = dim_avg(arr(z|:,time|:))/2.5e6 
  q_star(n) =sum(wqt(0:k_star)*dz(0:k_star))/w_star(n)/z_star(n)*1000./rho_star(n)

  print("z_star="+z_star(n)+"   rho_star="+rho_star(n)+"   w_star="+w_star(n)+"   t_star="+t_star(n)+"   q_star="+q_star(n)+" Precip"+avg(precip(period_begin(n):period_end(n)))+"cape "+avg(cape(period_begin(n):period_end(n)))+"cape_min "+min(cape(period_begin(n):period_end(n)))+"cape_max "+max(cape(period_begin(n):period_end(n))))

  delete(arr)
end do

print("----------------")
sc = rho_star(:)*w_star(:)^3

print("tke= "+tke(:n)/(rho_star(:n)*w_star(:n)^2*z_star(:n)*1000.)+"   res="+resd(:n)/sc(:n)+"   buo="+buo(:n)/sc(:n)+"   shear="+shear(:n)/sc(:n)+"   dissip"+dissip(:n)/sc(:n)+"   dissip_prec"+dissip_prec(:n)/dissip(:n)+"  a="+dissip(:n)*(z_star(:n)*1000.)^1.5*rho_star(:n)^0.5/tke(:n)^1.5+" taud="+tke(:n)/dissip(:n)+" tau="+tke(:n)/dissip(:n)*w_star(:n)/z_star(:n)/1000.+"eps="+eps(:n)+" dp="+dissip_prec(:n))

opt@yaxis = "z/z:B:*"


;----------------------------------------------------
; buoyancy flux


do n=0,nevents-1
  arr = file_in->BUOYA(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z" 
  fld(n,:) = dim_avg(arr(z|:,time|:))/(w_star(n)^3/z_star(n)/1000.) 
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<:G:b:R:w':G:H:R::B:v:N:'>/(w:B:*:N::S:3:N:/z:B:*:N:)"
opt@xaxis = ""

opt@xmin=-0.08
opt@xmax= 0.08
opt@xmin=1.2*min(fld(:nevents-1,:))
opt@xmax=1.2*max(fld(:nevents-1,:))
opt@ymin=0.
opt@ymax=1.5

plotXYall2(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
; vertical velocity variance


do n=0,nevents-1
  arr = file_in->W2(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z" 
  fld(n,:) = dim_avg(arr(z|:,time|:)) /w_star(n)^2
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w':S:2:N:>/w:B:*:N::S:2"
opt@xaxis = ""

opt@xmin=0.
opt@xmax=0.18
opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall2(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
; vertical velocity third moment


do n=0,nevents-1
  arr = file_in->W3(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:)) /w_star(n)^3
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w':S:3:N:>/w:B:*:N::S:3"
opt@xaxis = ""

opt@xmin=0.
opt@xmax=0.8
opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall2(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)

;----------------------------------------------------
;TKE 


do n=0,nevents-1
  arr = file_in->TKE(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))*rho(:)/rho_star(n)/w_star(n)^2 
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "CKE/:G:q:R::B:*:N:w:B:*:N::S:2"
opt@xaxis = ""

opt@xmin=0.
opt@xmax=5.


plotXYall2(wks,opt,z2,fld,nevents)

do n=0,nevents-1
  arr = file_in->TKES(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))*rho(:)/rho_star(n)/w_star(n)^2 
  delete(arr)
end do

puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)


;----------------------------------------------------
;  L W Pot Temp flux


do n=0,nevents-1
  arr = file_in->TLFLUX(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))/rho(:)/1000./(w_star(n)*t_star(n))
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w'h:B:L:N:'>/(w:B:*:N:T:B:*:N:)"
opt@xaxis = ""

opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))


plotXYall2(wks,opt,z2,fld,nevents)

puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)
;----------------------------------------------------
;  Total water flux


do n=0,nevents-1
  arr = file_in->QTFLUX(period_begin(n):period_end(n),:nz1-1)
  arr!0 = "time"
  arr!1 = "z"
  fld(n,:) = dim_avg(arr(z|:,time|:))/rho(:)/2.5e3/(w_star(n)*q_star(n))
  delete(arr)
end do

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frame(wks)
end if
opt@y_lab="False"
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if
opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)


opt@title = "<w'q:B:T:N:'>/(w:B:*:N:q:B:*:N:)"
opt@xaxis = ""

opt@xmin=0.
opt@xmax=1.6
opt@xmin=1.1*min(fld(:nevents-1,:))
opt@xmax=1.1*max(fld(:nevents-1,:))

plotXYall2(wks,opt,z2,fld,nevents)
puttext(wks,text_letter(nplot-1),opt@xl+0.85*opt@dx,opt@yt-0.1*opt@dy, 0.02)


frame(wks)

end
