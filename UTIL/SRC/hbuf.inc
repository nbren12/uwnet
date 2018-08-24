	
	integer hbuf_length, HBUF_MAX_LENGTH
	parameter(HBUF_MAX_LENGTH = 1000)
	
	real hbuf(HBUF_MAX_LENGTH*1000)	
	character *8 namelist(HBUF_MAX_LENGTH)	
	character *80 deflist(HBUF_MAX_LENGTH)	
	character *10 unitlist(HBUF_MAX_LENGTH)	
	integer status(HBUF_MAX_LENGTH)
	integer average_type(HBUF_MAX_LENGTH)

	common /hbuffer/ hbuf,namelist,deflist,unitlist,
     &			 status,average_type,hbuf_length
