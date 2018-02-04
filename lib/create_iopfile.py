"""See http://www.cesm.ucar.edu/models/atm-cam/docs/scam/ for a specification
of the fields contained in an IOP file.

Parameter settings for perpetual equinox
----------------------------------------

The most difficult thing to compute is the time, lat and lon which will make
the CAM radiation scheme match radiation used in SAM. I will need to set the
`solar_data_ymd` entry in CAM. See RAD_CAM/rad_full.f90:678 to see how it works
in SAM.

CAM Parameters
~~~~~~~~~~~~~~

Also see this discussion on perpetual equinox runs:
https://bb.cgd.ucar.edu/perpetual-solar-insolation-cam5

I will need to set the orbital parameters correctly. See the CAM namelist:
http://www.cesm.ucar.edu/cgi-bin/eaton/namelist/nldef2html-cam5_3

I will need to set this orbital parameters with SCAMs build-namelist.

    orb_obliq = 0.0
    orb_eccen = 0.0
    orb_mvelp = 0.0
    perpetual = .true.
    aqua_planet = .true. ! this implies that perpetual_ymd=March 21

`orb_mvelp` is the location of vernal equinox in longitude degrees, either the orb_iyear_AD must be set or the other three orb parameter must be set. default=unset

IOP File lat/lon/time
~~~~~~~~~~~~~~~~~~~~~

Because orb_mvelp = 0.0, we need the following setting in the IOPfile::

    lon = 0.0
    lat = dy*(j-ny/2)*2.5e-8 *360
    tsec = 'seconds since 1900-01-01T00:00:00'

After reading the SAM code carefully, I need to check the namelist entries for
day0, latitude0, and longitude0, do radlon and doradlat. Here is how the elements of these arrays are calculated::

    if (doradlat) then
    call task_rank_to_index(rank,it,jt)
    do j=1,ny
        latitude(:,j) = latitude0+dy*(j+jt-(ny_gl+YES3D-1)/2-1)*2.5e-8*360.
    end do
    else
    latitude(:,:) = latitude0
    end if

    if (doradlon) then
    call task_rank_to_index(rank,it,jt)
    do i=1,nx
        longitude(i,:) = longitude0+dx/cos(latitude0*pi/180.)* &
                                (i+it-nx_gl/2-1)*2.5e-8*360.
    end do
    else
    longitude(:,:) = longitude0
    end if

This means that::

    lat[j] = lat0 + dy*(j-ny/2)*2.5e-8 *360
    lon[i] = lon0 + dx*(i-nx/2)/cos(lat0 *pi/180) *i*2.5e-8 *360

2.5e-8 = 1/4e7 which is the circumference of the earth in meters. Therefore,
the factor 2.5-8 * 360 is the arclength of one degree.


Other Variables
---------------

For the SAM data, I will need to specify

- bdate
- tsec
- lev
- lat
- lon
- phis
- t
- q
- ps
- omega
- u
- v
- shflx
- lhflk
- divT
- vertdivT
- divq
- vertdivq



This is what a working IOP file looks like
netcdf epic_renamed {
dimensions:
	tsec = 88 ;
	lev = 23 ;
	lat = 1 ;
	lon = 1 ;
variables:
	int bdate ;
		bdate:long_name = "Base Date (year would be 2001 if 4 digit year were permitted)" ;
		bdate:units = "yymmdd" ;
	int tsec(tsec) ;
		tsec:long_name = "time after 00Z on nbdate" ;
		tsec:units = "s" ;
	int lev(lev) ;
		lev:long_name = "time after 00Z on nbdate" ;
		lev:units = "Pa" ;
	float lat(lat) ;
		lat:_FillValue = NaNf ;
		lat:long_name = "latitude" ;
		lat:units = "deg N" ;
	float lon(lon) ;
		lon:_FillValue = NaNf ;
		lon:long_name = "longitude" ;
		lon:units = "deg E" ;
	int year(tsec) ;
		year:units = "Year" ;
	int month(tsec) ;
		month:units = "Month" ;
	int day(tsec) ;
		day:units = "Day (UTC)" ;
	float hour(tsec) ;
		hour:_FillValue = NaNf ;
		hour:units = "Hour (UTC)" ;
	float calday(tsec) ;
		calday:_FillValue = NaNf ;
		calday:units = "Calday" ;
	float phis(lat, lon) ;
		phis:_FillValue = NaNf ;
		phis:long_name = "Surface Geopotential" ;
		phis:units = "m2/s2" ;
	float Ps(tsec, lat, lon) ;
		Ps:_FillValue = NaNf ;
		Ps:long_name = "Surface pressure (ERA-40)" ;
		Ps:units = "Pa" ;
	float Ptend(tsec, lat, lon) ;
		Ptend:_FillValue = NaNf ;
		Ptend:long_name = "Surface pressure tendency (ERA-40)" ;
		Ptend:units = "Pa/s" ;
	float Tg(tsec, lat, lon) ;
		Tg:_FillValue = NaNf ;
		Tg:long_name = "Sea Surface Temperature (ERA-40)" ;
		Tg:units = "K" ;
	float Ts(tsec, lat, lon) ;
		Ts:_FillValue = NaNf ;
		Ts:long_name = "Surface Air Temperature (ERA-40)" ;
		Ts:units = "K" ;
	float shflx(tsec, lat, lon) ;
		shflx:_FillValue = NaNf ;
		shflx:long_name = "Surface sensible heat flux (ERA-40)" ;
		shflx:units = "W/m2" ;
	float lhflx(tsec, lat, lon) ;
		lhflx:_FillValue = NaNf ;
		lhflx:long_name = "Surface sensible heat flux (ERA-40)" ;
		lhflx:units = "W/m2" ;
	float prec(tsec, lat, lon) ;
		prec:_FillValue = NaNf ;
		prec:long_name = "Total surface precipitation (ERA-40)" ;
		prec:units = "mm/d" ;
	float lwds(tsec, lat, lon) ;
		lwds:_FillValue = NaNf ;
		lwds:long_name = "surface longwave down (ERA-40)" ;
		lwds:units = "W/m2" ;
	float lwnt(tsec, lat, lon) ;
		lwnt:_FillValue = NaNf ;
		lwnt:long_name = "TOA net longwave up (ERA-40)" ;
		lwnt:units = "W/m2" ;
	float lwntc(tsec, lat, lon) ;
		lwntc:_FillValue = NaNf ;
		lwntc:long_name = "TOA clearsky net longwave up (ERA-40)" ;
		lwntc:units = "W/m2" ;
	float swds(tsec, lat, lon) ;
		swds:_FillValue = NaNf ;
		swds:long_name = "surface shortwave down (ERA-40)" ;
		swds:units = "W/m2" ;
	float swnt(tsec, lat, lon) ;
		swnt:_FillValue = NaNf ;
		swnt:long_name = "TOA net shortwave up (ERA-40)" ;
		swnt:units = "W/m2" ;
	float swntc(tsec, lat, lon) ;
		swntc:_FillValue = NaNf ;
		swntc:long_name = "TOA clearsky net shortwave up (ERA-40)" ;
		swntc:units = "W/m2" ;
	float u(tsec, lev, lat, lon) ;
		u:_FillValue = NaNf ;
		u:long_name = "Zonal wind (ERA-40)" ;
		u:units = "m/s" ;
	float v(tsec, lev, lat, lon) ;
		v:_FillValue = NaNf ;
		v:long_name = "Meridional wind (ERA-40)" ;
		v:units = "m/s" ;
	float omega(tsec, lev, lat, lon) ;
		omega:_FillValue = NaNf ;
		omega:long_name = "Vertical pressure velocity (ERA-40)" ;
		omega:units = "Pa/s" ;
	float T(tsec, lev, lat, lon) ;
		T:_FillValue = NaNf ;
		T:long_name = "Absolute Temperature (ERA-40)" ;
		T:units = "K" ;
	float divT(tsec, lev, lat, lon) ;
		divT:_FillValue = NaNf ;
		divT:long_name = "Horizontal Advective T tendency (ERA-40)" ;
		divT:units = "K/s" ;
	float vertdivT(tsec, lev, lat, lon) ;
		vertdivT:_FillValue = NaNf ;
		vertdivT:long_name = "Vertical Advective T tendency (ERA-40)" ;
		vertdivT:units = "K/s" ;
	float q(tsec, lev, lat, lon) ;
		q:_FillValue = NaNf ;
		q:long_name = "Specific humidity (ERA-40)" ;
		q:units = "kg/kg" ;
	float divq(tsec, lev, lat, lon) ;
		divq:_FillValue = NaNf ;
		divq:long_name = "Horizontal Advective q tendency (ERA-40)" ;
		divq:units = "kg/kg/s" ;
	float vertdivq(tsec, lev, lat, lon) ;
		vertdivq:_FillValue = NaNf ;
		vertdivq:long_name = "Vertical Advective q tendency (ERA-40)" ;
		vertdivq:units = "kg/kg/s" ;

// global attributes:
		:description = "Forcing dataset for EPIC ITCZ derived from ERA40" ;
		:author = "Peter Blossey" ;
		:date = "10-Aug-2006" ;

"""
