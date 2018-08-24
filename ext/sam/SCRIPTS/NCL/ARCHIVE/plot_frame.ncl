load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;=================================================;
; open file and read in data
;=================================================;
  nx = 2048
  ny = 2048

  nframe = asciiread("number",1,"integer")

  pixmap1=fbindirread("input.raw",nframe,(/ny,nx/),"byte")*1.
  print(min(pixmap1)+"   "+max(pixmap1))
;  pixmap = pixmap1(0:1199,800:1999)
;  pixmap = pixmap1(1024:,1024:ny-1)
  pixmap = 0.25*(pixmap1(::2,::2)+pixmap1(1::2,::2)+pixmap1(::2,1::2)+pixmap1(1::2,1::2))
  pixmap@units = ""+nframe

  wks  = gsn_open_wks("ncgm","")    ; open a ncgm file
;  wks  = gsn_open_wks("ps","")    ; open a ncgm file
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

  res@cnFillOn            = True          ; turn on color
  res@cnLinesOn            = False         ; turn on color
  res@gsnSpreadColors     = True          ; use full range of colormap
  res@cnLevelSelectionMode = "ExplicitLevels"   ; set explicit contour levels
  res@cnLevels             = ispan(0,NC-1,1)
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





