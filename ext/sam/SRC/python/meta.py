"""Dimension information for different variables"""

vars_3d = 'U V W qt sl FQT FSL TABS'.split()
vars_2d = 'SST SOLIN RADSFC RADTOA SHF LHF Prec lat lon'.split()
vars_z = 'p layer_mass pi'.split()
vars_scalar = 'p0'.split()

constant_vars = ['p', 'layer_mass', 'pi']

dims = {}

for name in vars_2d:
    dims[name] = ['time', 'null', 'y', 'x']

for name in vars_3d:
    dims[name] = ['time', 'z', 'y', 'x']

dims['pi'] = ['time', 'zs']
dims['layer_mass'] = ['z']
dims['p'] = ['z']
dims['p0'] = ['null']
dims['day'] = ['time']
dims['dt'] = ['time']
dims['nstep'] = ['time']
dims['time'] = ['time']
dims['Q1NN'] = ['time', 'z', 'y', 'x']
dims['Q2NN'] = ['time', 'z', 'y', 'x']

for name in constant_vars:
    dims[name] = [dim for dim in dims[name] if dim != 'time']
