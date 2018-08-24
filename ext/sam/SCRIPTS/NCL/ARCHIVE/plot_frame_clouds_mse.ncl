load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;=================================================;
; open file and read in data
;=================================================;
  nx = 2048
  ny = 2048

  NC = 125 ; number of levels for cloud field
  NM = 127 ; number of levels for MSE field

  nframe = asciiread("number",1,"integer")

  pixmap1=fbindirread("input.raw",nframe,(/ny,nx/),"byte")*1.
  pixmap1 = -1.+(pixmap1+1.)/(max(pixmap1)+1.)*4
  pixmap2=fbindirread("input1.raw",nframe,(/ny,nx/),"byte")*1.
  pixmap2 = 0.7+(pixmap2-0.7)/(max(pixmap2)-0.7)*2.3 
  pixmap2 = where(pixmap2.gt.2.0,pixmap2,pixmap2-1.7)
  pixmap2 = where(pixmap2.gt.2.0.and.pixmap2.lt.2.9,pixmap2-0.3,pixmap2)
  pixmap1 = where(pixmap1.lt.pixmap2,pixmap1,pixmap2)
  cloud1 = where(pixmap1.ge.2.0,0.,1.)
  pixmap1 = (pixmap1+1.)/4.*NC

  pixmap2=fbindirread("input3.raw",nframe,(/ny,nx/),"byte")*1. ; temp
  pixmap3=fbindirread("input4.raw",nframe,(/ny,nx/),"byte")*1. ; qv
  pixmap2 = NM*(5.*pixmap2/254. + 2.5*5*pixmap3/254.) /17.5
  pixmap2 = NC+1+pixmap2


;  pixmap = pixmap1(0:1199,800:1999)
;  pixmap = pixmap1(1024:,1024:ny-1)
  cloud_mask =  0.25*(cloud1(::2,::2)+cloud1(1::2,::2)+cloud1(::2,1::2)+cloud1(1::2,1::2))
  pixmap = cloud_mask*0.25*(pixmap1(::2,::2)+pixmap1(1::2,::2)+pixmap1(::2,1::2)+pixmap1(1::2,1::2))+ \
   (1.-cloud_mask)*0.25*(pixmap2(::2,::2)+pixmap2(1::2,::2)+pixmap2(::2,1::2)+pixmap2(1::2,1::2))
  print(min(pixmap1)+"   "+max(pixmap1))
  print(min(pixmap2)+"   "+max(pixmap2))
  print(min(pixmap)+"   "+max(pixmap))
  pixmap@units = ""+nframe

  wks  = gsn_open_wks("ncgm","")    ; open a ncgm file
;  wks  = gsn_open_wks("ps","")    ; open a ncgm file

; Cloud colors

cmap = new( (/2+NC+NM+1, 3/), float)
cmap( 1,:) = (/0.00, 0.00, 0.00/)
cmap( 0,:) = (/1.00, 1.00, 1.00/)
cmap(2:,:) = 0.

r1=0.13
r2=0.90
g1=0.16
g2=0.90
b1=0.41
b2=0.90
w = 1./(NC-1)
do i=0,NC-1
 cmap( 2+i,:) = (/r1+(r2-r1)*w*(NC-1-i), g1+(g2-g1)*w*(NC-1-i), b1+(b2-b1)*w*(NC-1-i)/)
; cmap( 2+i,:) = (/r1+(r2-r1)*w*i, g1+(g2-g1)*w*i, b1+(b2-b1)*w*i/)
end do


; MSE colors


r1=0.05
r2=0.99
g1=0.02
g2=0.99
b1=0.02
b2=0.06
w = 1./(NM-1)
do i=0,NM-1
; cmap( NC+i,:) = (/r1+(r2-r1)*w*(NM-1-i), g1+(g2-g1)*w*(NM-1-i), b1+(b2-b1)*w*(NM-1-i)/)
 cmap( NC+i,:) = (/r1+(r2-r1)*w*i, g1+(g2-g1)*w*i, b1+(b2-b1)*w*i/)
end do



  gsn_define_colormap(wks, cmap)   ; generate new color map

  res                     = True          ; plot mods desired
  res@gsnFrame        = False

  res@cnFillOn            = True          ; turn on color
  res@cnLinesOn            = False         ; turn on color
  res@gsnSpreadColors     = True          ; use full range of colormap
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
  res@cnLevels             = ispan(0,NC+NM-30,1)
  res@cnRasterModeOn = True
  res@gsnRightStringFontHeightF = 0.02

  res@vpWidthF            = 1.0           ; change aspect ratio of plot
  res@vpHeightF           = 1.0
  res@gsnPaperOrientation = "auto"
  res@gsnMaximize         = True        ; must include w/ Paper Orientation
  res@tmXBLabelsOn         = False              ; do not draw bottom labels
  res@tmXBOn               = False              ; no bottom tickmarks
  res@tmYLLabelsOn         = False              ; do not draw bottom labels
  res@tmYLOn               = False              ; no bottom tickmarks
  res@tmYROn               = False              ; no bottom tickmarks
  res@tmXTOn               = False              ; no bottom tickmarks
  res@lbLabelBarOn        = False
  res@cnLineLabelsOn   = False
  res@cnInfoLabelOn   = False

  plot = gsn_csm_contour(wks,pixmap,res)       ; contour the variable

 txres               = True                     ; text mods desired
 txres@txFontHeightF = 0.013                     ; font smaller. default big
gsn_text_ndc(wks,"System for Atmospheric Modeling SAM6.7, GATE_IDEAL 2048x2048x256 100m",0.44,0.97,txres)
gsn_text_ndc(wks,"(C) Marat Khairoutdinov, 2008",0.55,0.01,txres)

frame(wks)
  end





