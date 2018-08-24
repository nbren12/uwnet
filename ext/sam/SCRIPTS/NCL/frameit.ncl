
procedure  frameit(wks:graphic,title:string)
begin
txt = create "txt" textItemClass wks
                   "txString" : title
                     "txFont" : "duplex_roman"
                    "txPosXF" : 0.05
                    "txPosYF" : 0.01
                     "txJust" : "CenterLeft"
              "txFontHeightF" : 0.013
end create
draw(txt)
frame(wks)
end

