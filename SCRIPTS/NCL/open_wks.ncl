

;***********************************************************************;
; Function : gsn_open_wks                                               ;
;               type : type of workstation to open                      ;
;               name : name of workstation                              ;
;                                                                       ;
; This function opens either an X11 window, an NCGM file, or a          ;
; PostScript file depending on "type", which can be "x11", "ncgm", or   ;
; "ps". If "type" is a PS file or an NCGM, then it will be named        ;
; <name>.ps or <name>.ncgm respectively. This function also looks for a ;
; resource file called "name.res". If it exists, then it loads the      ;
; resources defined in that file. This function returns the workstation ;
; Id.                                                                   ;
;***********************************************************************;

function open_wks(type[1]:string)
local wks
begin
if (type.eq."NCGM") then
;
; Create an ncgm workstation.
;
   wks = create "ncgm" ncgmWorkstationClass defaultapp
   end create
else
  if (type.eq."X11") then
;
; Create an X workstation.
;
    wks = create "x11" xWorkstationClass defaultapp
      "wkPause" : "True"
    end create
  else
    if (type.eq."PS") then
;
; Create a PS workstation.
;
      wks = create "ps" psWorkstationClass defaultapp 
               "wkVisualType" : "color"
           "wkFullBackground" : "True"
      end create
    end if
  end if
end if

setvalues wks
          "wkBackgroundColor" : (/1.00,1.00,1.00/)
          "wkForegroundColor" : (/0.00,0.00,0.00/)
end setvalues

    return(wks)
end

