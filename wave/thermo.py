import numpy as np

grav = 9.81
R = 287.058
cp = 1004
kappa = R/cp
Lc = 2.5104e6
rho0 = 1.19
sec_in_day = 86400
liquid_water_density = 1000.0


def interface_heights(z):
    zext = np.hstack((-z[0], z, 2.0 * z[-1] - 1.0 * z[-2]))
    return .5 * (zext[1:] + zext[:-1])
