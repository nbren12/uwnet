load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;=================================================;
; open file and read in data
;=================================================;
  nx = 2048
  ny = 2048

  nframe = asciiread("number",1,"integer")

  pixmap1=fbindirread("u.raw",nframe,(/ny,nx/),"byte")*1.
  pixmap2=fbindirread("v.raw",nframe,(/ny,nx/),"byte")*1.
  print(min(pixmap1)+"   "+max(pixmap1))
  pixmap = 5*(pixmap1(400:900,400:900)-pixmap1(400:900,399:899)+pixmap2(400:900,400:900)-pixmap2(399:899,400:900))
  print(min(pixmap)+"  "+max(pixmap))
  pixmap@units = ""+nframe

  wks  = gsn_open_wks("ncgm","")    ; open a ncgm file
;  wks  = gsn_open_wks("ps","")    ; open a ncgm file
NC = 250
cmap = new( (/2+NC, 3/), float)
cmap( 1,:) = (/0.00, 0.00, 0.00/)
cmap( 0,:) = (/1.00, 1.00, 1.00/)
cmap(2:,:) = 0.

r1=0.
g1=0.
b1=1.
r2=1.
g2=0.
b2=0.
w = 1./(NC-1)
do i=0,NC-1
; cmap( 2+i,:) = (/r1+(r2-r1)*w*(NC-1-i), g1+(g2-g1)*w*(NC-1-i), b1+(b2-b1)*w*(NC-1-i)/)
 cmap( 2+i,:) = (/r1+(r2-r1)*w*i, g1+(g2-g1)*w*i, b1+(b2-b1)*w*i/)
end do

  gsn_define_colormap(wks, cmap)   ; generate new color map

  res                     = True          ; plot mods desired

  res@cnFillOn            = True          ; turn on color
  res@cnLinesOn            = False         ; turn on color
  res@gsnSpreadColors     = True          ; use full range of colormap
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
  res@cnLevels             = ispan(-NC/2,NC/2,1)
  res@cnRasterModeOn = True

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

  end





