"""Holoviews operations
"""
import numpy as np
import param
import holoviews as hv
from holoviews.operation import Operation
from holoviews.operation.datashader import (aggregate, datashade, dynspread,
                                            shade)


class percent(Operation):
    """
    Subtracts two curves from one another.
    """

    label = param.String(
        default='Percent',
        doc="""
        Normalizes data between 0 and 1.
        """)

    def normalize(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def _process(self, element, key=None):

        # Get x-values and y-values of curves
        x = element.dimension_values(0)
        y = element.dimension_values(1)
        M, m = y.max(), y.min()

        x = (x - m) / (M - m)
        y = (y - m) / (M - m)

        # Return new Element with subtracted y-values
        # and new label
        #         print("Scatter")
        return element.clone((x, y))


def shade_points(scat):
    return datashade(percent(scat), cmap="blue")\
        .redim.range(true=(0,1), lm=(0,1)) \
        .redim.unit(lm="%", true="%")\
        * hv.Curve(((0,1),(0,1))).opts(style=dict(color='black'))


def ndoverlay_to_contours(ndoverlay):
    """Turn an NDOverlay into a contour

    Notes
    -----
    Provided by Phillip Rudiger on the gitter channel

    Examples
    --------
    >>>     def ndoverlay_to_contours(ndoverlay):
    ...         dims = ndoverlay.dimensions(label='name')
    ...         return hv.Contours([dict(v.columns(), **dict(zip(dims, k))) for k, v in ndoverlay.data.items()],
    ...                         kdims=ndoverlay.last.dimensions()[:2], vdims=ndoverlay.kdims)
    ...
    ...     curves = hv.HoloMap({(chr(n+65), n): hv.Curve(np.random.rand(10)) for n in range(10)}, kdims=['variable', 'n'])
    ...     curves.overlay('n').grid('variable').map(ndoverlay_to_contours, hv.NdOverlay)
    ...
    """
    dims = ndoverlay.dimensions(label='name')
    return hv.Contours(
        [
            dict(v.columns(), **dict(zip(dims, k)))
            for k, v in ndoverlay.data.items()
        ],
        kdims=ndoverlay.last.dimensions()[:2],
        vdims=ndoverlay.kdims)


def extrap_1d(xc):
    return np.hstack((2 * xc[0] - xc[1], xc, 2 * xc[-1] - xc[-2]))


def edges_1d(xc):
    xc = extrap_1d(xc)

    return .5 * (xc[1:] + xc[:-1])


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
