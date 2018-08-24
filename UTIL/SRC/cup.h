	real alpham		! used to relate CKE to mass flux
	real flim		! mass flux limit
	logical cupice		! logical switch for ice phase
	real tice		! freezing temperature of water 
	integer ncb		!
	integer ltpcup		! the highest level that can have a cloud top
	real taudis		! dissipation time scale for cke (sec)
	real cp
	real hlat	
	real grav

	parameter (alpham = 1.e8)
	parameter (flim = 765.)
	parameter (tice = 273.1)
	parameter (cupice = .true.)
	parameter (ncb = 4)
	parameter (ltpcup = 3)
	parameter (taudis = 600.)
	parameter (cp = 1004.)
	parameter (hlat = 2.5104e+06)
	parameter (grav = 9.81)
