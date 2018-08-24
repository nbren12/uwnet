load "../../CSUCEM/NCL/open_wks.ncl"
load "../../CSUCEM/NCL/plotXYmarker.ncl"

begin

;wks = open_wks("NCGM")
;wks = open_wks("X11")
wks = open_wks("PS")

nens=0
nfiles_ens=20       ; number of files in the ensemble
nfiles_dat=1

nfiles=nens*nfiles_ens+nfiles_dat

filename=new((/100/),string)
filename_root=new((/100/),string)
file_in=new((/100/),file)

;====================================================================

filename_root(1)="."
filename(1) = "./ARM9707_128x128x64_new_X.evol"
;filename(1) = "../../SAM/ARM9903/ARM9903_128x128x64.evol"

;====================================================================
plot_ens = "False"

;filename_obs="/disk2/IOPs/GATE/GATE3h_new.nc"
;day0=242.
;filename_obs="/disk2/IOPs/TOGA/toga_coare.nc"
;day0=353.
filename_obs="/disk2/IOPs/9707/arm9707.nc"
day0=169.
;filename_obs="/disk2/IOPs/9903/arm9903.nc"
;day0=60.
;cldfrc_obs = "/disk2/IOPs/9707/cldamount_arm9707.nc"

time_scale = 3.

opt=0

opt@case_names = (/"3D","Obs"/)
opt@item_count=dimsizes(opt@case_names)

opt@line_colors = (/2,1,3,4,6,1,31,31,31,1,1,1/)
;opt@line_colors = (/2,1,1,4,1,31,31,31,1,1,1/)

opt@dash_patterns = (/0,0,0,0,0,0,0,0,0,0,0,0/)
opt@line_thicknesses = (/2.5,3.5,3.5,3.5,3.5,3.5,3.5,0.5,1.0,2.0,0.5/)
opt@marker_colors = opt@line_colors
opt@marker_types = (/1,4,5,7,1,9,8,12,15,1/)
opt@marker_sizes = (/0.008,0.008,0.008,0.008,0.008,0.008,0.004,0.004,0.004,0.001, 0.004/)
opt@markers="False"



opt@reverse = "False"
opt@legend="False"
opt@title=" "
opt@leg_fontsize=0.012
if(plot_ens.eq."True") then
  un=1
else
  un=0
end if
;---------------------------------------------------------------
plot_flag=(/1,12,19,23/)
;plot_flag=(/14,21,25/)
nplot_flag=dimsizes(plot_flag)
;-----------------------------------------------------------------

nplotx=1
nploty=4
xleft=0.1   
ytop=0.97
dxx=0.9
dyy=0.21
xl=new((/nplotx*nploty+1/),float)
yt=xl
dx=xl
dy=xl
do jj=0,nploty-1
 do ii=0,nplotx-1
   k=ii+nplotx*jj+1
   xl(k)=xleft+dxx*ii
   dx(k)=dxx-xleft
   yt(k)=ytop-dyy*jj
   dy(k)=dyy-2.*(1.-ytop)-0.01
 end do
end do

;-------------------------------------------------------------------


file_obs = addfile(filename_obs, "r")
day_obs = file_obs->tsec /3600./24.+ day0
;day_obs = day_obs - day_obs(0)
prec_obs = file_obs->Prec(:,0,0)*1000.*86400.
print(day_obs(:)+"  "+prec_obs(:))
lhf_obs =  file_obs->lhflx(:,0,0)
shf_obs =  file_obs->shflx(:,0,0)
cld_obs =  file_obs->totcld(:,0,0)
lw_obs = file_obs->TOA_LWup(:,0,0)
print(cld_obs)
;-------------------------------------------------------------------
nx = new((/nfiles+1/),integer)
day = new((/nfiles+1,10000/),float)
i = new((/nfiles+1/),integer)
day=0.
f=new((/nfiles+1,10000/),float)
f=0.
fn = new((/nfiles+1/),string)



do j=1,nens
 do l=1,nfiles_ens
  fn(l+(j-1)*nfiles_ens)=filename_root(j)+l+".evol"
  print(filename_root(j)+l+".evol")
 end do
end do

do l=1,nfiles_dat
 fn(nfiles_ens*nens+l)=filename(l)
end do




do n=1,nfiles

i(n)=0 
caseid = chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
print("caseid: "+caseid)
i(n)=i(n)+1 
nx(n) = fbinrecread(fn(n),i(n),1,"integer")  
print(nx(n)+" ")
i(n)=i(n)+1
nfield = fbinrecread(fn(n),i(n),1,"integer") 
i(n)=i(n)+1 
day(n,:nx(n)-1) = fbinrecread(fn(n),i(n),(/nx(n)/),"float")  
day(n,:nx(n)-1) = day(n,:nx(n)-1) - 0.5*( day(n,2)-day(n,1))

;day(n,:nx(n)-1)=day(n,:nx(n)-1)-day(n,0)

end do
print(day(1,:nx(1)-1))
lwp_obs = new((/nx(1)/),float)
;-------------------------------------------------------------------

iplot=0
iplot_flag=0
print("nfiled="+nfield)
opt@y_axis = "True"


do m=1,nfield


  print(m+" ")
 if(m.eq.plot_flag(iplot_flag)) then

  iplot=iplot+1

  if(iplot.eq.nplotx*nploty) then
    opt@xaxis = "Julian Day"
    opt@x_lab="True"
  else
    opt@xaxis = ""
    opt@x_lab="False"
  end if

 end if

  do n=1,nfiles

   i(n)=i(n)+1
   opt@title = chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
   i(n)=i(n)+1
   opt@yaxis=chartostring(fbinrecread(fn(n),i(n),-1,"character")) 
   i(n)=i(n)+1
   f(n,:nx(n)-1) = fbinrecread(fn(n),i(n),(/nx(n)/),"float")
   if(n.eq.1) then
     print(m+" "+ opt@title)
   end if

  end do

  if(m.eq.18) then
     lwp_obs = f(1,:nx(1)-1) 
  end if


  opt@xl = xl(iplot)
  opt@dx = dx(iplot)
  opt@yt = yt(iplot)
  opt@dy = dy(iplot)

  opt@xmin=min(day(1:nfiles,0))
  opt@xmax=max(day(1:nfiles,max(nx)-1))
  opt@ymin=0.8*min(f(1:nfiles,:min(nx)-1))
  opt@ymax=0.9*max(f(1:nfiles,:min(nx)-1))
  if(m.eq.1) then
    opt@ymin=0.
    opt@ymax=1.
  end if
  if(m.eq.12) then
    opt@ymin=140.
    opt@ymax=320.
  end if
  if(m.eq.19) then
    opt@ymin=0.
    opt@ymax=60.
  end if
  if(m.eq.21) then
    opt@ymin=0.
    opt@ymax=240.
  end if
  if(m.eq.24) then
    opt@ymin=0.
    opt@ymax=240.
  end if
  if(m.eq.23) then
    opt@ymin=0.
    opt@ymax=80.
  end if
  

  if(m.eq.plot_flag(iplot_flag)) then  

  opt@line_color = opt@item_count-1
  opt@dash_pattern =  opt@item_count-1
  opt@line_thickness =  opt@item_count-1
  opt@marker_no =  opt@item_count-1
  if(m.eq.1) then
;     plotXY(wks, opt, cld_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.12) then
     plotXY(wks, opt, lw_obs,day_obs)
     opt@legend="True"
  end if
  if(m.eq.23) then
     plotXY(wks, opt, prec_obs,day_obs)
     opt@legend="True"
  end if
 if(m.eq.20) then
     plotXY(wks, opt, lhf_obs,day_obs)
     opt@legend="false"
  end if
  if(m.eq.21) then
     plotXY(wks, opt, shf_obs,day_obs)
     opt@legend="false"
  end if

  if(m.eq.19) then
     plotXY(wks, opt, lwp_obs, day(1,:nx(1)-1)) 
     opt@legend="false"
  end if
  do j=0,nens-1
  do l=1,nfiles_ens

      opt@line_color = 0
      opt@dash_pattern = 0
      opt@line_thickness = 0
      opt@marker_no = 0
      n=j*nfiles_ens+l
      if(plot_ens.eq."True") then
       nnn=floattointeger(time_scale/((day(n,2)-day(n,1))*24.))+1
       if(nnn.gt.1) then
         wgt=new((/nnn/),float)
         wgt=1./nnn
         f(n,:nx(l)-1)=wgt_runave( f(n,:nx(l)-1),wgt,1)
         delete(wgt)
       end if
       plotXY(wks, opt, f(n,:nx(n)-1),day(n,:nx(n)-1))
      end if

  end do
  end do

  do j=0,nens-1
    opt@line_color = un+j 
    opt@dash_pattern = un+j
    opt@line_thickness = un+j
    opt@marker_no = un+j
    n=j*nfiles_ens+1
    f!0="ens"
    f!1="x"
    f1 = dim_avg(f(x|:nx(n)-1,ens|n:(j+1)*nfiles_ens))
    nnn=floattointeger(time_scale/((day(n,2)-day(n,1))*24.))+1
    if(nnn.gt.1) then
     wgt=new((/nnn/),float)
     wgt=1./nnn
     f1(:nx(l)-1)=wgt_runave( f1(:nx(l)-1),wgt,1)
     delete(wgt)
    end if
    plotXY(wks, opt, f1(:nx(n)-1:nnn),day(n,:nx(n)-1:nnn))
  end do

  do l=nens*nfiles_ens+1,nfiles

      opt@line_color = l - nfiles_ens*nens +un+(nens-1)
      opt@line_thickness = l - nfiles_ens*nens +un+(nens-1)
      opt@dash_pattern = l - nfiles_ens*nens +un+(nens-1)
      opt@marker_no = l - nfiles_ens*nens +un+(nens-1)

  nnn=floattointeger(time_scale/((day(l,2)-day(l,1))*24.))+1
  print(" "+nnn)
  if(nnn.gt.1) then
    wgt=new((/nnn/),float)
    wgt=1./nnn
    f(l,:nx(l)-1)=wgt_runave( f(l,:nx(l)-1),wgt,1)
    delete(wgt)
  end if

      plotXY(wks, opt, f(l,:nx(l)-1:nnn),day(l,:nx(l)-1:nnn))

  end do

  end if


  if(m.eq.plot_flag(iplot_flag)) then

  opt@legend="False"
  axisXY(wks, opt)
  if(iplot.eq.3) then
    item_max=5
    opt@leg_xl =  opt@xl+0.42*opt@dx
    opt@leg_yt =  opt@yt
    opt@leg_dx =  0.15*opt@dx
    opt@leg_dy =  0.10*opt@dy*opt@item_count
    if(opt@item_count.le.item_max) then
      legendXY(wks,opt)
    else
      item_count=opt@item_count
      opt@item_count = item_max
      opt@leg_dy =  0.12*opt@dy*opt@item_count
      legendXY(wks,opt)
      case_names = opt@case_names
      line_colors = opt@line_colors
      dash_patterns = opt@dash_patterns
      line_thicknesses = opt@line_thicknesses
      marker_colors = opt@marker_colors
      marker_types = opt@marker_types 
      marker_sizes = opt@marker_sizes 
      opt@case_names(0:item_count-item_max-1) = case_names(item_max:item_count-1)
      opt@line_colors(0:item_count-item_max-1) = line_colors(item_max:item_count-1)
      opt@dash_patterns(0:item_count-item_max-1) = dash_patterns(item_max:item_count-1)
      opt@line_thicknesses(0:item_count-item_max-1)=line_thicknesses(item_max:item_count-1)
      opt@marker_colors(0:item_count-item_max-1)=marker_colors(item_max:item_count-1)
      opt@marker_types(0:item_count-item_max-1)=marker_types(item_max:item_count-1)
      opt@marker_sizes(0:item_count-item_max-1)=marker_sizes(item_max:item_count-1)
      opt@item_count=item_count-item_max
      opt@leg_xl =  opt@leg_xl+1.1*opt@leg_dx
      opt@leg_dy =  0.12*opt@dy*opt@item_count
      legendXY(wks,opt)
      opt@item_count=item_count
      opt@case_names = case_names
      opt@line_colors = line_colors
      opt@dash_patterns = dash_patterns
      opt@line_thicknesses = line_thicknesses
      opt@marker_colors = marker_colors
      opt@marker_types = marker_types 
      opt@marker_sizes = marker_sizes


    end if
  end if


  iplot_flag=iplot_flag+1
  if(iplot_flag.ge.nplot_flag) then
   break
  end if

 end if

end do

frame(wks)

end
