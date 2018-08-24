load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

begin
;=================================================;
; open file and read in data
;=================================================;
  nx = 256
  ny = 256

  nframe_start = 599
  nframe_end = 1439


  do nframe=nframe_start,nframe_end
    cwp=fbindirread("input.raw",nframe,(/ny*nx/),"byte")*1.
    cwp = 10.^(-1.+(cwp+1.)/(max(cwp)+1.)*4)    ; convert to normal units
    print(nframe+"   "+min(cwp)+"   "+max(cwp))
  end do

end

