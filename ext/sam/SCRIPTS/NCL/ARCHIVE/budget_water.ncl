load "../NCL/open_wks.ncl"
load "../NCL/plotXYmarker.ncl"
load "../NCL/frameit.ncl"

begin

wks = open_wks("NCGM")
;wks = open_wks("PS")

dt=1.

;file1="./ARM9707_128x128x64_new_X.nc"
;file1="../ARM0003/ARM0003_128x128_x.nc"
;file1="../../SAM/ARM9903/ARM9903_128x128x64.nc"
file1="../../SAM/ARM9901/ARM9901_128x128x64.nc"

text="ARM January 1999 IOP"

;day_start=196.3
;day_end=196.4
day_start=0.
day_end=365.

prec_thresh = 5.

fraction=0.7
;fraction=1.

file_in1=addfile(file1,"r")

caseid1=file1
print("caseid1 "+caseid1)

;-------------------------------------------------------------------
opt=0
opt@legend="True"
opt@markers="False"

nz1=floattointeger(fraction*dimsizes(file_in1->z))
print(dimsizes(file_in1->z))
z1=file_in1->z(0:nz1-1)*0.001
p1=file_in1->p(0:nz1-1)

dz = 0.5*(z1(0)+z1(1))
print(dz)
az=z1
do k=1,nz1-2
  az(k)=(z1(k+1)-z1(k-1))/(z1(0)+z1(1))
end do
az(0)=1.
az(nz1-1)=az(nz1-2)


opt@reverse="False"


;z1=file_in1->p(0:nz1-1)
;opt@reverse="True"

day1=file_in1->time
nt1=dimsizes(day1)


nt_start1=0
nt_end1=0
do i=0,nt1-1
  if(day1(i).lt.day_start) then
	nt_start1=i
  end if
  if(day1(i).le.day_end) then
	nt_end1=i
  end if
end do

nt = nt_end1 - nt_start1+1
nc = 0
nz = nz1

factor = new((/nt,nz/),float)
 prec = file_in1->PRECIP(nt_start1:nt_end1,0)
  do k=0,dimsizes(prec)-1
    if(prec(k).ge.prec_thresh) then
      factor(k,:) = 1.
      nc = nc+1
    else
      factor(k,:) = 0.
    end if
  end do
  factor = factor*nt/nc
  print(nt+"   "+nc)

opt@line_colors = (/2,3,4,26,6,7,8,12/)
opt@line_thicknesses = (/3.,3.,3.,3.,3.,3.,3.,3./)
opt@dash_patterns = (/0,1,3,2,4,0,0,0/)

nplotx=3
nploty=1
xleft=0.04
ytop=0.86
dxx=0.23
dyy=0.38
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
puttext(wks,text, xl(2)+0.5*dx(2), 0.93,0.015)

opt@yaxis = "km"

arr1=new((/nt_end1-nt_start1+1,nz1/),float)  
arr1!0="time" 
arr1!1="z"
prof1 = new((/12,nz1/),float)

rho=file_in1->RHO(0,0:nz1-1)

print(z1+"  "+az+" "+rho)

;-------------------------------------------------
; Total Water budget:

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frameit(wks,caseid1)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)

opt@case_names=(/"LS","Trans","Conv","Evap","Res","",""/)
opt@line_colors = (/2,3,4,26,7,6,8,12/)
opt@dash_patterns = (/0,0,0,0,0,0,0,0/)

opt@title = "Vapor+Cloud"
opt@xaxis = "g/m:S:3:N:/day"
opt@yaxis = "km"



arr1 = file_in1->QTEND(nt_start1:nt_end1,0:nz1-1)*factor
prof1(0,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QTADV(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QTDIFF(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)+prof1(1,:)
arr1 = file_in1->QTSINK(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(2,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QTSRC(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(3,:) = dim_avg(arr1(z|:,time|:))*rho(:)
prof1(4,:) = prof1(0,:)+prof1(1,:)+prof1(2,:)+prof1(3,:)

opt@xmin=-8.
opt@xmax= 8.

;opt@xmin=1.1*min(prof1(:4,2:))
;opt@xmax= 1.1*max(prof1(:4,2:))

plotXYall(wks,opt,z1,prof1,5)

;-------------------------------------------------
; Precipitating water budget:

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frameit(wks,caseid1)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)

opt@case_names=(/"Fall","Trans","Conv","Evap","Res","",""/)
opt@line_colors = (/8,3,4,26,7,6,8,12/)
opt@dash_patterns = (/0,0,0,0,0,0,0,0/)

opt@title = "Precipitation"
opt@xaxis = "g/m:S:3:N:/day"
opt@yaxis = ""

arr1 = file_in1->QPFALL(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(0,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QPADV(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QPDIFF(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)+prof1(1,:)
arr1 = file_in1->QPSRC(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(2,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QPEVP(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(3,:) = dim_avg(arr1(z|:,time|:))*rho(:)
prof1(4,:) = prof1(0,:)+prof1(1,:)+prof1(2,:)+prof1(3,:)


;opt@xmin=1.1*min(prof1(:3,2:))
;opt@xmax= 1.1*max(prof1(:3,2:))

plotXYall(wks,opt,z1,prof1,5)

print("Precip:")
print("sum_F="+sum(prof1(0,:)*az(:)*dz))
print("sum_T="+sum(prof1(1,:)*az(:)*dz))
print("sum_C="+sum(prof1(2,:)*az(:)*dz))
print("sum_E="+sum(prof1(3,:)*az(:)*dz))
print("sum_S="+sum(prof1(4,:)*az(:)*dz))
;-------------------------------------------------
; All water budget:

nplot=nplot+1
if(nplot.gt.nplotx*nploty) then
 nplot=1
 frameit(wks,caseid1)
end if
opt@y_lab="False"
if((nplot-1)%nplotx.eq.0) then
 opt@y_lab = "True"
end if

opt@xl = xl(nplot)
opt@dx = dx(nplot)
opt@yt = yt(nplot)
opt@dy = dy(nplot)

opt@case_names=(/"Fall","Trans","LS","Res","","",""/)
opt@line_colors = (/8,3,2,7,4,7,6,12/)
opt@dash_patterns = (/0,0,0,0,0,0,0,0/)

opt@title = "All-Water"
opt@xaxis = "g/m:S:3:N:/day"

arr1 = file_in1->QPFALL(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(0,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QPADV(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)
arr1 = file_in1->QPDIFF(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)+prof1(1,:)
arr1 = file_in1->QTADV(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)+prof1(1,:)
arr1 = file_in1->QTDIFF(nt_start1:nt_end1,0:nz1-1)/dt*factor
prof1(1,:) = dim_avg(arr1(z|:,time|:))*rho(:)+prof1(1,:)
arr1 = file_in1->QTEND(nt_start1:nt_end1,0:nz1-1)*factor 
prof1(2,:) = dim_avg(arr1(z|:,time|:))*rho(:)
prof1(3,:) = prof1(0,:)+prof1(1,:)+prof1(2,:)

;opt@xmin=1.1*min(prof1(:3,2:))
;opt@xmax= 1.1*max(prof1(:3,2:))

plotXYall(wks,opt,z1,prof1,4)

print("All water:")
print("sum_F="+sum(prof1(0,:)*az(:))*dz)
print("sum_T="+sum(prof1(1,:)*az(:))*dz)
print("sum_L="+sum(prof1(2,:)*az(:))*dz)
print("sum_S="+sum(prof1(3,:)*az(:))*dz)
;-------------------------------------------------


frame(wks)

end
