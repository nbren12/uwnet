function [out] = create_scam_netcdf_file(ncfile,comment,...
                                         lev,lon,lat,iyear,calday, ...
                                         phis)
% $$$                                          

%function [out] = create_scam_netcdf_file(ncfile,comment,...
%                                         time,lev,lon,lat, ...
%                                         nbdate,calday,phis);
% 
% creates a netcdf file suitable for using as an IOP forcing
% dataset for SCAM, the single column version of the Community
% Atmosphere Model (CAM).  Uses SNCTOOLS netcdf interface.

mySchema.Name   = '/';
mySchema.Format = 'classic';

% cell array (length # dimensions) of smaller cell arrays that hold
%   dimension names and lengths
Dimensions = {{'lat',1}, ...
              {'lon',1}, ...
              {'lev',length(lev)}, ...
              {'time',length(calday)}};

for n = 1:length(Dimensions)
  mySchema.Dimensions(n).Name = Dimensions{n}{1};
  mySchema.Dimensions(n).Length = Dimensions{n}{2};
end

% cell array (length # attributes) of smaller cell arrays that hold
%   attribute names and values.
Attributes = {{'author','Peter Blossey, Email: pblossey@uw.edu'}, ...
              {'institution','University of Washington, Seattle'}, ...
              {'Conventions','CF-1.3'}, ...
              {'date',date}, ...
              {'comment',comment}};

for n = 1:length(Attributes)
  mySchema.Attributes(n).Name = Attributes{n}{1};
  mySchema.Attributes(n).Value = Attributes{n}{2};
end

% TODO: Add these attributes in the future.
% $$$ nc_attput(ncfile, nc_global, 'source', '');
% $$$ nc_attput(ncfile, nc_global, 'references', '');


% $$$ for n = 1:length(Variables)
% $$$   mySchema.Variables(n).Name = Variables{n}{1};
% $$$   mySchema.Variables(n).Dimensions = Variables{n}{2};
% $$$   mySchema.Variables(n).Datatype = Variables{n}{3};  
% $$$   mySchema.Variables(n).Attributes(1).Name = 'units';
% $$$   mySchema.Variables(n).Attributes(1).Value = Variables{n}{4};
% $$$   mySchema.Variables(n).Attributes(2).Name = 'long_name';
% $$$   mySchema.Variables(n).Attributes(2).Value = Variables{n}{5};
% $$$   mySchema.Variables(n).Attributes(3).Name = 'standard_name';
% $$$   mySchema.Variables(n).Attributes(3).Value = Variables{n}{6};
% $$$ end

ncwriteschema(ncfile, mySchema);
ncdisp(ncfile);

Variables = { ...
    {'lat',{'lat'},'single','degree_north','Latitude','latitude'}, ...
    {'lon',{'lon'},'single','degree_east','Longitude','longitude'}, ...
    {'lev',{'lev'},'single','Pa','Pressure','air_pressure'}, ...
    {'tsec',{'time'},'int32','s','Time in seconds after 00Z on nbdate','time'}, ...
    {'calday',{'time'},'single','d',sprintf('Time in days after 00Z on Dec. 31, %d',iyear),''}, ...
    {'year',{'time'},'int32','year','Year',''}, ...
    {'month',{'time'},'int32','month','Month',''}, ...
    {'day',{'time'},'int32','day','Day',''}, ...
    {'hour',{'time'},'single','hour','Hour',''}, ...
    {'nbdate',{},'int32','yymmdd','Base Date',''}, ...
    {'bdate',{},'int32','yymmdd','Base Date',''}, ...
    {'phis',{'lat','lon'},'single','m2/s2',...
     'Surface Geopotential','surface_geopotential'}};
for n = 1:length(Variables)
  disp(Variables{n}{1})
  nccreate(ncfile,Variables{n}{1}, ...
           'Dimensions',Variables{n}{2}, ...
           'Datatype',Variables{n}{3})
  ncwriteatt(ncfile,Variables{n}{1}, ...
             'units',Variables{n}{4})
  ncwriteatt(ncfile,Variables{n}{1}, ...
             'long_name',Variables{n}{5})
  ncwriteatt(ncfile,Variables{n}{1}, ...
             'standard_name',Variables{n}{6})
end
ncwriteatt(ncfile,'nbdate','comment', ...
           ['Note that only two digit year is permitted.']);


ncwrite(ncfile,'lat',lat) %%%%% ADD LATITUDE
ncwrite(ncfile,'lon',lon) %%%%% ADD LONGITUDE
ncwrite(ncfile,'lev',lev) %%%%% ADD LEV/PRESSURE COORDINATE
ncwrite(ncfile,'phis',phis) %%%%% ADD SURFACE GEOPOTENTIAL

tsec = int32(86400*(calday - floor(calday(1))));
ncwrite(ncfile,'tsec',tsec) %%%%% ADD TIME COORDINATE (INTEGER SECONDS)

%%%%% ADD OTHER TIME-RELATED QUANTITIES %%%%%%%%%%
ncwrite(ncfile,'calday',calday) %%%%% ADD CALENDAR DAY

% create date vector.
dv = datevec(double(calday+datenum(iyear-1,12,31)))

% Add date-related quantites
ncwrite(ncfile,'year',int32(dv(:,1))) %%%%% ADD YEAR
ncwrite(ncfile,'month',int32(dv(:,2))) %%%%% ADD MONTH
ncwrite(ncfile,'day',int32(dv(:,3))) %%%%% ADD DAY
ncwrite(ncfile,'hour',dv(:,4)+dv(:,5)/60+dv(:,6)/3600) %%%%% ADD HOUR

% Set nbdate
yy = mod(dv(1,1),100);
mm = dv(1,2);
dd = dv(1,3);
nbdate = int32(1e4*yy + 1e2*dv(1,2) + dv(1,3));

ncwrite(ncfile,'nbdate',nbdate)
ncwrite(ncfile,'bdate',nbdate)


