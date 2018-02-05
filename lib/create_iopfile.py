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
import numpy as np
import xarray as xr

from xnoah import swap_coord
from .thermo import omega_from_w
from .advection import vertical_advection, horizontal_advection


circumference_earth = 4.0075e7


def open_and_merge(file_2d, files_3d, stat_file):
    data_3d = xr.open_mfdataset(files_3d, preprocess=lambda x: x.drop('p'))
    data_2d = xr.open_dataset(file_2d)
    data_2d = data_2d.isel(time=np.argsort(data_2d.time.values))
    stat = xr.open_dataset(stat_file)
    return xr.merge((data_3d, data_2d, stat.p, stat.RHO, stat.Ps),
                    join='inner')


def compute_tendencies(data):
    return dict(
        divT=xr.DataArray(
            -horizontal_advection(data.U, data.V, data.TABS),
            attrs={'units': 'K/s'}),
        divq=xr.DataArray(
            -horizontal_advection(data.U, data.V, data.QV/1000),
            attrs={'units': 'kg/kg/s'}),
        vertdivT=xr.DataArray(
            -vertical_advection(data.W, data.TABS),
            attrs={'units': 'K/s'}),
        vertdivq=xr.DataArray(
            -vertical_advection(data.W, data.QV/1000),
            attrs={'units': 'kg/kg/s'}),
    )


def x_to_lon(x):
    return xr.DataArray(x/circumference_earth*360,
                        attrs={'units': 'deg E',
                               'long_name': 'longitude'})


def y_to_lat(y):
    n = len(y)
    if n % 2 == 1:
        raise ValueError("Y coordinate must have even number of points")
    else:
        n2 = n//2
    ymid = (y[n2-1] + y[n2])/2
    lat = (y-ymid)/circumference_earth*360
    return xr.DataArray(lat,
                        attrs={'units': 'deg N',
                               'long_name': 'latitude'})

vars_3d = ['q', 'T', 'u', 'v', 'divT', 'divq', 'vertdivT', 'vertdivq']
vars_2d = ['shflk', 'lhflx', 'Ptend', 'Ps', 'phis']

def expand_dims(x):
    if x.name in vars_3d + vars_2d:
        return x.expand_dims(['lat', 'lon'])
    else:
        return x


def prepare_iop_dataset(data):
    data_vars = dict(
        # time related variables
        bdate=xr.DataArray(990101, attrs={'units': "yymmdd"}),
        tsec=xr.DataArray(data.time*86400, attrs={'units': "s"}),
        # spatial coordinates
        lev=xr.DataArray(data.p*100, attrs={'units': 'Pa'}),
        lat=y_to_lat(data.y),
        lon=x_to_lon(data.x),
        # surface variables
        phis=0.0,
        Ps=xr.DataArray(data.Ps*100, attrs={'units': 'Pa'}),
        Ptend=data.Ps * 0,
        shflx=data.SHF,
        lhflx=data.LHF,
        # 3d variables
        q=xr.DataArray(data.QV/1000, attrs={'units': 'kg/kg'}),
        T=data.TABS,
        u=data.U,
        v=data.V,
        omega=xr.DataArray(omega_from_w(data.W, data.RHO[-1]),
                           attrs={'units': 'Pa/s'}),
    )

    data_vars.update(compute_tendencies(data))

    ds = xr.Dataset(data_vars)
    ds = (swap_coord(ds, {"time": "tsec", 'x': 'lon', 'y': 'lat', 'z': 'lev'})
          .transpose('tsec', 'lev', 'lat', 'lon')
          .sortby('lev'))

    return ds.drop('time')


namelist_template = """
&atm
    iopfile='iop.nc'
    nhtfrq=-1
    single_column=.true.
    scmlat= {lat:.4f}
    scmlon= {lon:.4f}
    aqua_planet = .true.
    orb_obliq = 0.0
    orb_eccen = 0.0
    orb_mvelp = {lon:.4f}
    orb_mode = 'fixed_parameters'
    perpetual = .true.
/
"""

def main():
    file_2d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/2d/all.nc"
    files_3d = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/coarse/3d/*.nc"
    stat = "data/raw/2/NG_5120x2560x34_4km_10s_QOBS_EQX/stat.nc"

    data = open_and_merge(file_2d, files_3d, stat)
    iop = prepare_iop_dataset(data)

    # just grab one column
    # SCAM needs lat and lon to be dimensions
    # not just coordinates
    loc = iop.isel(lon=0, lat=32)\
             .apply(expand_dims)\
             .transpose('tsec', 'lev', 'lat', 'lon')
    # for some reason SCAM dies when lon = 0
    # something to do with initializing the land vegetation array
    loc.lon[0] = 80

    print("Saving file to disk")
    loc.to_netcdf("data/processed/iop0x32/iop.nc")

    print("Saving namelist to disk")
    with open("data/processed/iop0x32/namelist.txt", "w") as f:
        f.write(namelist_template.format(
            lat=float(loc.lat),
            lon=float(loc.lon)))







if __name__ == '__main__':
    main()
