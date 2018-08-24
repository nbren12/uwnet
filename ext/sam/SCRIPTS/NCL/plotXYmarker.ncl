
procedure plotXY(wks:graphic, opt, y[*]:numeric,x[*]:numeric)

begin


dataid = create "xyData" coordArraysClass noparent
    "caYArray" : y
    "caXArray" : x    
    "caXMissingV" : -999.
end create

 
plotid = create "xyPlot" xyPlotClass wks

	"xyCoordData"		:  dataid
	"vpXF"			:  opt@xl
	"vpYF"			:  opt@yt
	"vpWidthF"		:  opt@dx
	"vpHeightF"		:  opt@dy
	"pmTickMarkDisplayMode" : "Never"
	 "trYReverse"		: opt@reverse
	 "trXReverse"		: True
	 "trYMinF"            : opt@ymin
  	 "trYMaxF"            : opt@ymax
  	 "trXMinF"            : opt@xmin
  	 "trXMaxF"            : opt@xmax


end create

getvalues plotid
   "xyCoordDataSpec" : dspec
end getvalues

setvalues dspec
   	"xyLineColors" 		: opt@line_colors(opt@line_color)
   	"xyLineThicknesses" 	: opt@line_thicknesses(opt@line_thickness)
        "xyDashPatterns"        : opt@dash_patterns(opt@dash_pattern) 
end setvalues

if(opt@markers.eq."True") then
setvalues dspec
        "xyMarkLineMode"        :  "MARKLINES"
        "xyMonoMarkLineMode"    :  True
        "xyMarkerColor"         : opt@marker_colors(opt@marker_no)
        "xyMarkerSizeF"         : opt@marker_sizes(opt@marker_no)
        "xyMarker"              : opt@marker_types(opt@marker_no)
	"xyMarkerThicknesses"   : opt@line_thicknesses(opt@line_thickness)
 "xyLineLabelConstantSpacingF"  : 3. 
end setvalues
end if
draw(plotid)

end


procedure plotXYmarker(wks:graphic, opt, y[*]:numeric,x[*]:numeric)

begin


dataid = create "xyData" coordArraysClass noparent
    "caYArray" : y
    "caXArray" : x    
    "caXMissingV" : -999.
        "xyMarkLineMode"        :  "MARKERS"
        "xyMonoMarkLineMode"    :  True
end create

 
plotid = create "xyPlot" xyPlotClass wks

	"xyCoordData"		:  dataid
	"vpXF"			:  opt@xl
	"vpYF"			:  opt@yt
	"vpWidthF"		:  opt@dx
	"vpHeightF"		:  opt@dy
	"pmTickMarkDisplayMode" : "Never"
	 "trYReverse"		: opt@reverse
	 "trYMinF"            : opt@ymin
  	 "trYMaxF"            : opt@ymax
  	 "trXMinF"            : opt@xmin
  	 "trXMaxF"            : opt@xmax

end create

getvalues plotid
   "xyCoordDataSpec" : dspec
end getvalues

setvalues dspec
        "xyMarkLineMode"        :  "MARKERS"
        "xyMonoMarkLineMode"    :  True
   	"xyMarkerColor" 	: opt@marker_colors(opt@marker_color)
   	"xyMarkerSizeF" 	: opt@marker_sizes(opt@marker_size)
   	"xyMarker" 	        : opt@marker_types(opt@marker_type)
end setvalues

draw(plotid)

end



procedure stuffXY(wks:graphic, opt)

begin
font_max = 0.008
font_size = 0.065*opt@dx
if(font_size.gt.font_max) then
  font_size = font_max
end if


if(opt@legend.eq."True") then
legendid = create "Legend" legendClass wks
	"vpXF"			: opt@xl+0.6*opt@dx
	"vpYF"			: 0.99*opt@yt
	"vpWidthF"		: 0.25*opt@dx
	"vpHeightF"		: 0.02*opt@item_count
	"lgPerimOn"		: False	
	"lgLineColors"  	: opt@line_colors
	"lgLabelStrings"	: opt@case_names
	"lgLabelFontHeightF"	: 0.01
	"lgItemCount"		: opt@item_count
	"lgMonoLineThickness"	: False
	"lgLineThicknesses"	: opt@line_thicknesses
	"lgDashIndexes" 	: opt@dash_patterns
	"lgAutoManage"		: False
end create
draw(legendid)

end if

plotid = create "xyPlot" xyPlotClass wks

	"vpXF"			:  opt@xl
	"vpYF"			:  opt@yt
	"vpWidthF"		:  opt@dx
	"vpHeightF"		:  opt@dy
	"pmTickMarkDisplayMode" : "Always"
	"trYMinF" 		: opt@ymin
	"trYMaxF" 		: opt@ymax
	"trXMinF" 		: opt@xmin
	"trXMaxF" 		: opt@xmax
	"trYReverse"		: opt@reverse
	"trXReverse"		: True
        "tmXBLabelsOn"          :  opt@x_lab
        "tmYLLabelsOn"          :  opt@y_lab
        "tiXAxisOn"             :  opt@x_lab
        "tiYAxisOn"             :  opt@y_lab
	"tiXAxisString"		:  opt@xaxis
	"tiYAxisString"		:  opt@yaxis
	"tiMainString"		:   opt@title
	"tiXAxisFontHeightF"	:  font_size
	"tiYAxisFontHeightF"	:  font_size
	"tmXBLabelFontHeightF"	:  font_size
	"tmYLLabelFontHeightF"	:  font_size
;	"tmXMajorGrid"          :  True
  "tmXMajorGridLineDashPattern" :  2
  "tmXMajorGridThicknessF"	: 0.5 
;	"tmYMajorGrid"          :  True
  "tmYMajorGridLineDashPattern" :  2
  "tmYMajorGridThicknessF"	: 0.5 
	"tiMainFontHeightF"	:  font_size*1.2
                   "tmXBMode"   : "Explicit"
               "tmXBLabelsOn"   : "True"
                 "tmXBValues"   : (/-90., -60.,-30.,0.,30.,60.,90./)
                 "tmXBLabels"   : (/"90S","60S","30S","EQ","30N","60N","90N"/)

	
end create
draw(plotid)

end


procedure legendXY(wks:graphic, opt)

begin


if(opt@markers.eq."True") then

legendid = create "Legend" legendClass wks
	"vpXF"			: opt@leg_xl
	"vpYF"			: opt@leg_yt
	"vpWidthF"		: opt@leg_dx
	"vpHeightF"		: opt@leg_dy
	"lgPerimOn"		: False	
	"lgLineColors"  	: opt@line_colors
	"lgLabelStrings"	: opt@case_names
	"lgLabelFontHeightF"	: opt@leg_fontsize
	"lgItemCount"		: opt@item_count
	"lgMonoLineThickness"	: False
	"lgLineThicknesses"	: opt@line_thicknesses
	"lgDashIndexes" 	: opt@dash_patterns
	"lgAutoManage"		: False
        "lgMarkerColors"        : opt@marker_colors
        "lgMarkerSizes"         : opt@marker_sizes
        "lgMarkerIndexes"       : opt@marker_types
        "lgMonoMarkerIndex"     : False
        "lgItemType"            : "MARKLINES"
end create

else

legendid = create "Legend" legendClass wks
        "vpXF"                  : opt@leg_xl
        "vpYF"                  : opt@leg_yt
        "vpWidthF"              : opt@leg_dx
        "vpHeightF"             : opt@leg_dy
        "lgPerimOn"             : False
        "lgLineColors"          : opt@line_colors
        "lgLabelStrings"        : opt@case_names
        "lgLabelFontHeightF"    : opt@leg_fontsize
        "lgItemCount"           : opt@item_count
        "lgMonoLineThickness"   : False
        "lgLineThicknesses"     : opt@line_thicknesses
        "lgDashIndexes"         : opt@dash_patterns
        "lgAutoManage"          : False
end create
 
end if
draw(legendid)

end


procedure axisXY(wks:graphic, opt)

begin

font_max = 0.008
font_size = 0.065*opt@dx
if(font_size.gt.font_max) then
  font_size = font_max
end if

plotid = create "xyPlot" xyPlotClass wks

	"vpXF"			:  opt@xl
	"vpYF"			:  opt@yt
	"vpWidthF"		:  opt@dx
	"vpHeightF"		:  opt@dy
	"pmTickMarkDisplayMode" : "Always"
	"trYMinF" 		: opt@ymin
	"trYMaxF" 		: opt@ymax
	"trXMinF" 		: opt@xmin
	"trXMaxF" 		: opt@xmax
	 "trYReverse"		: opt@reverse
;	"trXReverse"		: True
	"tiXAxisString"		:  opt@xaxis
	"tiYAxisString"		:  opt@yaxis
	"tiXAxisFontHeightF"	:  font_size
	"tiYAxisFontHeightF"	:  font_size
	"tmXBLabelFontHeightF"	:  font_size
	"tmYLLabelFontHeightF"	:  font_size
;	"tmXMajorGrid"          :  True
	"tmXBLabelsOn"		:  opt@x_lab 
	"tmYLLabelsOn"		:  opt@y_lab 
	"tiXAxisOn"		:  opt@x_lab 
	"tiYAxisOn"		:  opt@y_lab 
  "tmXMajorGridLineDashPattern" :  2
  "tmXMajorGridThicknessF"	: 0.5 
;	"tmYMajorGrid"          :  True
  "tmYMajorGridLineDashPattern" :  2
  "tmYMajorGridThicknessF"	: 0.5 
	"tiMainFontHeightF"	:  font_size

	
end create
draw(plotid)
 txt = create "txt" textItemClass wks
                   "txString" : opt@title
                    "txPosXF" : opt@xl+0.5*opt@dx
                    "txPosYF" : opt@yt+0.02
                     "txJust" : "CenterCenter"
              "txFontHeightF" : font_size*1.2
 end create
 draw(txt)
end


procedure puttext(wks:graphic,text:string,x:numeric,y:numeric,fsize:numeric)
begin
 txt = create "txt" textItemClass wks
                   "txString" : text
                    "txPosXF" : x
                    "txPosYF" : y
                     "txJust" : "CenterCenter"
              "txFontHeightF" : fsize
 end create
 draw(txt)
end


procedure plotXYall(wks:graphic, opt,z[*]:numeric, f[*][*]:numeric,\
                    nplot:numeric)

begin

;opt@xmin=min(f(0:nplot,:)) 
;opt@xmax=max(f(0:nplot,:)) 
;dd = (opt@xmax-opt@xmin)*0.3
;opt@xmin=opt@xmin
;opt@xmax=opt@xmax+dd
opt@ymin=0.
opt@ymax=max(z)
opt@item_count = nplot

do m=0,nplot-1
 opt@line_color = m
 opt@line_thickness = m
 opt@dash_pattern = m
 plotXY(wks, opt, z, f(m,:))
end do
 stuffXY(wks, opt)

end

procedure plotXYall1(wks:graphic, opt,z[*][*]:numeric, f[*][*]:numeric,\
                    nplot:numeric)

begin

;opt@xmin=min(f(0:nplot,:))
;opt@xmax=max(f(0:nplot,:))
;dd = (opt@xmax-opt@xmin)*0.3
;opt@xmin=opt@xmin
;opt@xmax=opt@xmax+dd
opt@ymin=0.
opt@ymax=max(z)
opt@item_count = nplot

do m=0,nplot-1
 opt@line_color = m
 opt@line_thickness = m
 opt@dash_pattern = m
 opt@marker_no = m
 plotXY(wks, opt, z(m,:), f(m,:))
end do
 axisXY(wks, opt)

end

procedure plotXYall2(wks:graphic, opt,z[*][*]:numeric, f[*][*]:numeric,\
                    nplot:numeric)

begin

;opt@xmin=min(f(0:nplot,:))
;opt@xmax=max(f(0:nplot,:))
;dd = (opt@xmax-opt@xmin)*0.3
;opt@xmin=opt@xmin
;opt@xmax=opt@xmax+dd
;opt@ymin=0.
;opt@ymax=max(z)
opt@item_count = nplot

do m=0,nplot-1
 opt@line_color = m
 opt@line_thickness = m
 opt@dash_pattern = m
 opt@marker_no = m
 plotXY(wks, opt, z(m,:), f(m,:))
end do
 axisXY(wks, opt)
end






procedure profiles(wks:graphic, opt,z[*]:numeric, f[*][*]:numeric)

begin
opt@title = f@long_name
opt@xaxis = f@units
opt@ytitle = z@long_name
opt@xmin=1.e30
opt@xmax=-1.e30
nplot=opt@nplot
nt=opt@nt
do m=0,nplot-1
 n=m*(nt-1)/(nplot-1)
 mn=min(f(n,:))
 mx=max(f(n,:))
 if(mn.lt.opt@xmin) then
        opt@xmin=mn
 end if
 if(mx.gt.opt@xmax) then
        opt@xmax=mx
 end if
end do
dd=(opt@xmax-opt@xmin)*0.3
opt@xmin=opt@xmin
opt@xmax=opt@xmax+dd
do m=0,nplot-1
 n=m*(nt-1)/(nplot-1)
 opt@line_color = m
 opt@line_thickness = m
 opt@dash_pattern = m
 opt@case_names(m)=""+opt@day(n)
 plotXY(wks, opt, z, f(n,:))
end do
 stuffXY(wks, opt)

end


