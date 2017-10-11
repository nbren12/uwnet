import numpy as np
import holoviews as hv

def extrap_to_interface(p):
    pextrap = np.hstack((2 * p[0] - p[1], p, 2 * p[-1] - p[-2]))
    pint = .5 * (pextrap[1:] + pextrap[:-1])

    return pint


def quadmesh(arg, **kwargs):
    """Wrapper to QuadMesh with pcolormesh like behavior
    """

    x, y, z = arg
    if len(x) == z.shape[1]:
        # need to construct interface values
        x = extrap_to_interface(x)
        y = extrap_to_interface(y)

    if x[0] > x[1]:
        x = x[::-1]
        z = z[:, ::-1]
    if y[0] > y[1]:
        y = y[::-1]
        z = z[::-1, :]

    return hv.QuadMesh((x, y, z), **kwargs)
