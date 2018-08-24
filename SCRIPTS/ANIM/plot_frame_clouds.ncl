load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;=================================================;
; open file and read in data
;=================================================;
  nx = 64
  ny = 64

  title = "test"
;  title = "SAM6.7, SuperCell_WMO_Case4 1024x512x64 500m 4s"
;  title = "System for Atmospheric Modeling SAM6.7, GATE_IDEAL 2048x2048x256 100m"

  nframe = asciiread("number",1,"integer")

  pixmap1=fbindirread("cwp.raw",nframe,(/ny,nx/),"byte")*1.
  pixmap1 = -1.+(pixmap1+1.)/(max(pixmap1)+1.)*4
  pixmap2=fbindirread("iwp.raw",nframe,(/ny,nx/),"byte")*1.
  pixmap2 = 0.7+(pixmap2-0.7)/(max(pixmap2)-0.7)*2.3 
  pixmap2 = where(pixmap2.gt.2.0,pixmap2,pixmap2-1.7)
  pixmap2 = where(pixmap2.gt.2.0.and.pixmap2.lt.2.9,pixmap2-0.3,pixmap2)
;  pixmap1 = pixmap2
  pixmap1 = where(pixmap1.lt.pixmap2,pixmap1,pixmap2)
  print(min(pixmap1)+"   "+max(pixmap1))
  pixmap1 = (pixmap1+1.)/4*250.
;  pixmap = pixmap1(0:1800,200:2000)
;  pixmap = pixmap1(400:900,400:900)
;   pixmap = pixmap1(:,0:511)
   pixmap = pixmap1(:,:)
;  pixmap = 0.25*(pixmap1(::2,::2)+pixmap1(1::2,::2)+pixmap1(::2,1::2)+pixmap1(1::2,1::2))
  pixmap@units = ""+nframe

;  wks  = gsn_open_wks("ncgm","")    ; open a ncgm file
  wks  = gsn_open_wks("ps","")    ; open a ncgm file

NC = 250
cmap = new( (/2+NC, 3/), float)
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

  gsn_define_colormap(wks, cmap)   ; generate new color map

  res                     = True          ; plot mods desired
  res@gsnFrame        = False

  res@cnFillOn            = True          ; turn on color
  res@cnLinesOn            = False         ; turn on color
  res@gsnSpreadColors     = True          ; use full range of colormap
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
  res@cnLevels             = ispan(0,NC-1,1)
  res@cnRasterModeOn = True
  res@gsnRightStringFontHeightF = 0.02

  sizes = dimsizes(pixmap)
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
 gsn_text_ndc(wks,title,0.44,0.97,txres)
 gsn_text_ndc(wks,"(C) Marat Khairoutdinov, 2008",0.55,0.01,txres)

 frame(wks)

end





