.. _dataheader
Training Data Header
====================

This is a valid training data header::

  netcdf \2018-10-02-ngaqua-subset {
  dimensions:
    time = UNLIMITED ; // (156 currently)
    z = 34 ;
    y = 1 ;
    x = 128 ;
  variables:
    float FQT(time, z, y, x) ;
      FQT:_FillValue = NaNf ;
      string FQT:units = "g/kg/s" ;
    float FSLI(time, z, y, x) ;
      FSLI:_FillValue = NaNf ;
      string FSLI:units = "K/s" ;
    float FU(time, z, y, x) ;
      FU:_FillValue = NaNf ;
      string FU:units = "m/s^2" ;
    float FV(time, z, y, x) ;
      FV:_FillValue = NaNf ;
      string FV:units = "m/s^2" ;
    float LHF(time, y, x) ;
      LHF:_FillValue = NaNf ;
      string LHF:long_name = "Latent Heat Flux" ;
      string LHF:units = "W/m2" ;
    float Prec(time, y, x) ;
      Prec:_FillValue = NaNf ;
      string Prec:long_name = "Surface Precip. Rate" ;
      string Prec:units = "mm/day" ;
    float QN(time, z, y, x) ;
      QN:_FillValue = NaNf ;
      string QN:long_name = "Non-precipitating Condensate (Water+Ice)                                        " ;
      string QN:units = "g/kg      " ;
    float QP(time, z, y, x) ;
      QP:_FillValue = NaNf ;
      string QP:long_name = "Precipitating Water (Rain+Snow)                                                 " ;
      string QP:units = "g/kg      " ;
    float QT(time, z, y, x) ;
      QT:_FillValue = NaNf ;
      string QT:long_name = "Total non-precipitating water                                                   " ;
      string QT:units = "g/kg      " ;
    float RADSFC(time, y, x) ;
      RADSFC:_FillValue = NaNf ;
    float RADTOA(time, y, x) ;
      RADTOA:_FillValue = NaNf ;
    float SHF(time, y, x) ;
      SHF:_FillValue = NaNf ;
      string SHF:long_name = "Sensible Heat Flux" ;
      string SHF:units = "W/m2" ;
    float SLI(time, z, y, x) ;
      SLI:_FillValue = NaNf ;
      string SLI:long_name = "Liquid/Ice Static Energy                                                        " ;
      string SLI:units = "K         " ;
    float SOLIN(time, y, x) ;
      SOLIN:_FillValue = NaNf ;
      string SOLIN:long_name = "Solar TOA insolation" ;
      string SOLIN:units = "W/m2" ;
    float SST(time, y, x) ;
      SST:_FillValue = NaNf ;
      string SST:long_name = "Sea Surface Temperature" ;
      string SST:units = "K" ;
    float U(time, z, y, x) ;
      U:_FillValue = NaNf ;
      string U:long_name = "X Wind Component                                                                " ;
      string U:units = "m/s       " ;
    float V(time, z, y, x) ;
      V:_FillValue = NaNf ;
      string V:long_name = "Y Wind Component                                                                " ;
      string V:units = "m/s       " ;
    float W(time, z, y, x) ;
      W:_FillValue = NaNf ;
      string W:long_name = "Z Wind Component                                                                " ;
      string W:units = "m/s       " ;
    double layer_mass(z) ;
      layer_mass:_FillValue = NaN ;
      string layer_mass:units = "kg/m2" ;
    float rho(z) ;
      rho:_FillValue = NaNf ;
      string rho:long_name = "Air density" ;
      string rho:units = "kg/m3" ;
    double time(time) ;
      time:_FillValue = NaN ;
      string time:units = "day" ;
      string time:long_name = "time" ;
    float x(x) ;
      x:_FillValue = NaNf ;
      string x:units = "m" ;
    float y(y) ;
      y:_FillValue = NaNf ;
      string y:units = "m" ;
    float z(z) ;
      z:_FillValue = NaNf ;
      string z:units = "m" ;
      string z:long_name = "height" ;
