% constants from SAM's SRC/params.f90
Rd    = 287;
Cp    = 1004;

ncin = 'constrain_setup_forcing.nc';

wh = {'time','z','lat','lon','SST','LHF','SHF','wsubs','pressure','U','V','T','theta','theta_l','qt','qt_adj','qv','qv_adj','qc','RHw','RHi','o3','LWhr','SWhr'};
for m = 1:length(wh)
  in.(wh{m}) = double(ncread(ncin,wh{m}));
end

%% Harmonize time dimensions and delete duplicate times
calday = in.time/86400 + datenum('31-Jan-2010') - datenum('31-Dec-2009');
Ntime = length(calday);

Nlev = length(in.z);


for k = 1:Ntime
  in.Ps(k) = interp1(in.z,in.pressure(:,k),0,'linear','extrap');
  in.q_surf(k) = interp1(in.z,in.qt(:,k),0,'linear','extrap');
  in.T_surf(k) = interp1(in.z,in.T(:,k),0,'linear','extrap');
end

% choose last pressure sounding for pressure coordinate so that
%   the highest pressure in the sounding won't exceed the surface
%   pressure.
lev = in.pressure(end:-1:1,end);  % flip so that lev(end) is
                                  % closest to surface!!
Nlev = length(lev);

% we can use the auxillary variables q_surf and T_surf to
%  fill in the values of qt and T below the bottom pressure level.

% Form geostrophic wind (only in v)
in.Vg = (-15 -0.0024*in.z)*ones(1,Ntime);

% form density locally and use to compute omega.
in.rho = in.pressure./(Rd*in.T.*(1+0.61*in.qv));
in.omega = -g*in.rho.*in.wsubs;

in.Tl = in.theta_l.*(in.pressure/1e5).^(Rd/Cp);

% interpolate the time-height fields onto the new pressure grid
wh = {'wsubs','omega','U','V','T','Tl','theta','theta_l','qt','qt_adj','qv','qv_adj','qc','RHw','RHi','LWhr','SWhr','Vg'};
for m = 1:length(wh)
  tmp = in.(wh{m});
  clear tmp2
  for k = 1:Ntime
    tmp2(:,k) = interp1(in.pressure(:,k),tmp(:,k),lev,'linear', ...
                        'extrap');
  end
  out.(wh{m}) = tmp2;
end
  
% special treatment for ozone
out.o3mmr = in.o3(end:-1:1)*ones(1,Ntime);

%%%%% OPEN NETCDF FILE FOR FORCINGS %%%%%%%%%%

nc = ['CONSTRAIN_forcing_from_setup_' datestr(today) '.nc'];
comment = ['Forcings for CONSTRAIN adapted from constrain_setup_forcing.nc ' ...
           'file and converted to netcdf SCAM IOP format by ' ...
           'Peter Blossey (Univ of Washington).  The constrain_setup_forcing.nc file' ...
           ' was obtained from the intercomparison website ' ...
           'http://appconv.metoffice.com/cold_air_outbreak/' ...
            'constrain_case/home.html' ... 
          '  Note that this case is a GASS intercomparison.'];
iyear = 2010;
create_scam_netcdf_file(nc,comment,lev,-11,66,2010,calday,0)


%%%%% TIMESERIES OF SURFACE/TOA FIELDS (FLUXES, SURFACE TEMP, ETC.) %%%%%%%%%%
%% Note that all variables have dimensions {'time','lat','lon'}
%%   and are single precision.
Variables = { ...
    {'Ps','Pa','Surface Pressure','surface_air_pressure'}, ...
    {'Ptend','Pa/s','Surface Pressure Tendency','tendency_of_surface_air_pressure'}, ...
    {'Tg','K','Surface Temperature (SST if over water)','surface_temperature'}, ...
    {'shflx','W/m2','Surface Sensible Heat Flux','surface_upward_sensible_heat_flux'}, ...
    {'lhflx','W/m2','Surface Latent Heat Flux','surface_upward_latent_heat_flux'}, ...
    {'Tsair','K','Surface Air Temperature (extrapolated to z=0)',''}, ...
    {'qsrf','kg/kg','Surface Water Vapor Mass Mixing ratio (extrapolated to z=0)',''}, ...
    };
for n = 1:length(Variables)
  disp(Variables{n}{1})
  nccreate(nc,Variables{n}{1}, ...
           'Dimensions',{'lon','lat','time'}, ...
           'Datatype','single')
  ncwriteatt(nc,Variables{n}{1}, ...
             'units',Variables{n}{2})
  ncwriteatt(nc,Variables{n}{1}, ...
             'long_name',Variables{n}{3})
  ncwriteatt(nc,Variables{n}{1}, ...
             'standard_name',Variables{n}{4})
% $$$   ncwrite(nc,Variables{n}{1},...
% $$$           reshape(Variables{n}{5},[Ntime 1 1]));
end

ncwrite(nc,'Ps',reshape(in.Ps,[1 1 Ntime]));
ncwrite(nc,'Ptend',reshape(deriv_nonuniform(in.time,in.Ps),[1 ...
                    1 Ntime]));
ncwrite(nc,'Tg',reshape(in.SST,[1 1 Ntime]));
ncwrite(nc,'shflx',reshape(in.SHF,[1 1 Ntime]));
ncwrite(nc,'lhflx',reshape(in.LHF,[1 1 Ntime]));
ncwrite(nc,'Tsair',reshape(in.T_surf,[1 1 Ntime]));
ncwrite(nc,'qsrf',reshape(in.q_surf,[1 1 Ntime]));

%%%%% TIMESERIES OF VERTICALLY-VARYING FIELDS (T,q,etc.) %%%%%%%%%%
%% Note that all variables have dimensions {'time','lev','lat','lon'}
%%   and are single precision.
Variables = { ...
    {'u','m/s','Zonal Wind','eastward_wind'}, ...
    {'v','m/s','Meridional Wind','northward_wind'}, ...
    {'ug','m/s','Geostrophic Zonal Wind','geostrophic_eastward_wind'}, ...
    {'vg','m/s','Geostrophic Meridional Wind','geostrophic_northward_wind'}, ...
    {'omega','Pa/s','Vertical Pressure Velocity','lagrangian_tendency_of_air_pressure'}, ...
    {'T','K','Liquid Water Temperature (converted from theta_l, for initialization)','air_temperature'}, ...
    {'q','kg/kg','Water Vapor Mass Mixing Ratio (Adjusted to give saturated layer at time=0)',''}, ...
    {'divT','K/s','Large-scale Horizontal Temperature Advection',''}, ...
    {'divq','K/s','Large-scale Horizontal Advection of Water Vapor Mass Mixing Ratio',''}, ...
    {'Tref','K','Absolute Temperature from Unified Model Simulation','air_temperature'}, ...
    {'qref','kg/kg','Water Vapor Mass Mixing Ratio from Unified Model Simulation',''}, ...
    {'o3mmr','kg/kg','Ozone Mass Mixing Ratio from Unified Model/McClatchey Sounding',''}, ...
    };
for n = 1:length(Variables)
  disp(Variables{n}{1})
  nccreate(nc,Variables{n}{1}, ...
           'Dimensions',{'lon','lat','lev','time'}, ...
           'Datatype','single')
  ncwriteatt(nc,Variables{n}{1}, ...
             'units',Variables{n}{2})
  ncwriteatt(nc,Variables{n}{1}, ...
             'long_name',Variables{n}{3})
  ncwriteatt(nc,Variables{n}{1}, ...
             'standard_name',Variables{n}{4})
% $$$   ncwrite(nc,Variables{n}{1},...
% $$$           reshape(Variables{n}{5},[Ntime 1 1]));
end

ncwrite(nc,'u',reshape(out.U,[1 1 Nlev Ntime]))
ncwrite(nc,'v',reshape(out.V,[1 1 Nlev Ntime]))
ncwrite(nc,'ug',reshape(zeros(size(out.U)),[1 1 Nlev Ntime]))
ncwrite(nc,'vg',reshape(out.Vg,[1 1 Nlev Ntime]))
ncwrite(nc,'omega',reshape(out.omega,[1 1 Nlev Ntime]))

ncwrite(nc,'T',reshape(out.Tl,[1 1 Nlev Ntime]))
ncwrite(nc,'q',reshape(out.qt_adj,[1 1 Nlev Ntime]))
ncwrite(nc,'divT',reshape(zeros(size(out.T)),[1 1 Nlev Ntime]))
ncwrite(nc,'divq',reshape(zeros(size(out.qt)),[1 1 Nlev Ntime]))

ncwrite(nc,'Tref',reshape(out.T,[1 1 Nlev Ntime]))
ncwrite(nc,'qref',reshape(out.qt,[1 1 Nlev Ntime]))

ncwrite(nc,'o3mmr',reshape(out.o3mmr,[1 1 Nlev Ntime]))


